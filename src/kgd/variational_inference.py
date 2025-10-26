from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import grad, jit, random, vmap
from jaxtyping import Array, Float
from tqdm import tqdm

from . import kgd_functions as f
from .kgd_functions import GradientKernel, KernelGradientDiscrepancy


class Model(nn.Module):
    """
    Pushforward map.
    """

    hidden_dim: int
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: Float[Array, "d"]) -> Float[Array, "1"]:
        """
        Compute the pushforward map.

        Args:
            x (Float[Array, "d"]): Input array of shape (d,).

        Returns:
            Float[Array, "1"]: Output array of shape (1,).
        """
        y = nn.Dense(self.hidden_dim)(x)
        y = nn.elu(y)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.elu(y)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.elu(y)
        y = nn.Dense(self.output_dim)(y)
        return y.squeeze()


class VariationalInference:
    """
    Variational Inference via Kernel Gradient Discrepancy.
    """

    def __init__(
        self,
        grad_log_q0: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
        L: Callable[[Float[Array, "n d"]], Float[Array, "n"]],
        k: Callable[[Float[Array, "d"], Float[Array, "d"]], Float[Array, "1"]],
        d: int,
        layers_dim: int = 128,
        learning_rate: float = 1e-3,
    ) -> None:
        """
        Variational Inference via Kernel Gradient Discrepancy.

        Args:
            grad_log_q0 (Callable[[Float[Array, "n d"]], Float[Array, "n d"]]): Function computing the gradient of the log-density of the initial distribution q0.
            L (Callable[[Float[Array, "n d"]], Float[Array, "n"]]): Function computing the interaction potential.
            k (Callable[[Float[Array, "d"], Float[Array, "d"]], Float[Array, "1"]]): Kernel function.
            d (int): Dimensionality of the data.
            layers_dim (int): Dimension of the hidden layers in the neural network (default is 128).
            learning_rate (float): Learning rate for the optimizer (default is 1e-3).
        """
        self.d = d
        self.S_q0 = grad_log_q0
        self.L = L
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k = k
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)

        self.learning_rate = learning_rate
        self.model_T = Model(layers_dim, self.d)
        self.tx = optax.adam(learning_rate=learning_rate)

    def noise_fn(
        self,
        key: Float[Array, "key_dim"],
        shape: tuple[int, ...],
    ) -> Float[Array, "..."]:
        """
        Generate noise samples from a uniform distribution. (mu_0)

        Args:
            key (Float[Array, "key_dim"]): JAX random key.
            shape (tuple[int, ...]): Shape of the output array.

        Returns:
            Float[Array, "..."]: Array of shape `shape` with uniform noise samples.
        """
        return random.uniform(key, shape, minval=-3, maxval=3)

    def pretraining(
        self,
        key: Float[Array, "key_dim"],
        T_pretrain: int = 1000,
    ) -> dict[str, Any]:
        """
        Pretrain the neural network to match the initial distribution q0.

        Args:
            key (Float[Array, "key_dim"]): JAX random key.
            T_pretrain (int): Number of pretraining steps.

        Returns:
            dict[str, Any]: Pretrained model parameters.
        """

        def pretraining_loss(
            key: Float[Array, "key_dim"],
            params: dict[str, Any],
            n_samples: int = 100,
            init_std: float = 3.0,
        ) -> Float[Array, "1"]:
            """
            Compute the pretraining loss.

            Args:
                key (Float[Array, "key_dim"]): JAX random key.
                params (dict[str, Any]): Model parameters.
                n_samples (int): Number of samples to generate (default is 100).
                init_std (float): Standard deviation of the initial distribution (default is 3.0

            Returns:
                Float[Array, "1"]: Pretraining loss value.
            """
            key1, key2 = random.split(key)
            noise = self.noise_fn(key1, (n_samples, self.d))
            X = self.model_T.apply(params, noise)
            target_samples = init_std * random.normal(key2, X.shape)
            assert target_samples.shape == X.shape
            k = lambda x, y: f.imq_scale_mixtures(x, y, jnp.logspace(-2, 2, 5))
            Kxx = vmap(lambda x: vmap(lambda y: k(x, y))(X))(X)
            Kxy = vmap(lambda x: vmap(lambda y: k(x, y))(target_samples))(X)
            Kyy = vmap(lambda x: vmap(lambda y: k(x, y))(target_samples))(
                target_samples
            )

            return (Kxx - 2 * Kxy + Kyy).mean()

        key1, key2, *(keys) = random.split(key, T_pretrain + 2)
        x = random.normal(key1, (self.d,))
        params = self.model_T.init(key2, x)
        jax.tree_util.tree_map(lambda x: x.shape, params)  # Checking output shapes
        self.model_T.apply(params, x)
        opt_state = self.tx.init(params)

        for step in tqdm(range(T_pretrain)):
            loss_grad_fn = jax.value_and_grad(
                lambda params: pretraining_loss(keys[step], params)
            )
            loss_value, grads = loss_grad_fn(params)
            updates, opt_state = self.tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        return params

    def l2_norm(self, params: dict[str, Any]) -> Float[Array, "1"]:
        """
        Compute the L2 norm of the model parameters.

        Args:
            params (dict[str, Any]): Model parameters.

        Returns:
            Float[Array, "1"]: L2 norm of the parameters.
        """
        return sum([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)])

    def loss(
        self,
        key: Float[Array, "key_dim"],
        params: dict[str, Any],
        num_samples: int = 100,
        d_noise: int = 4,
        reg: float = 0.0,
    ) -> Float[Array, "1"]:
        """
        Compute the loss function.

        Args:
            key (Float[Array, "key_dim"]): JAX random key.
            params (dict[str, Any]): Model parameters.
            num_samples (int): Number of samples to generate (default is 100).
            d_noise (int): Dimensionality of the noise input (default is 4).
            reg (float): Regularization coefficient (default is 0.0).

        Returns:
            Float[Array, "1"]: Loss value.
        """
        noise = self.noise_fn(key, (num_samples, d_noise))
        X = self.model_T.apply(params, noise)
        return self.KGD.square_kgd(X) + reg * self.l2_norm(params)

    def run_particles(
        self,
        key: Float[Array, "key_dim"],
        T: int,
        n: int,
        N: int = 2000,
        gamma: int = 300,
    ) -> list[Float[Array, "n d"]]:
        """
        Run the Variational Inference via Kernel Gradient Discrepancy.

        Args:
            key (Float[Array, "key_dim"]): JAX random key.
            T (int): Number of iterations.
            n (int): Number of particles.
            N (int): Number of samples for loss computation (default is 2000).
            gamma (int): Regularization coefficient (default is 300).

        Returns:
            list[Float[Array, "n d"]]: List containing the particles at each iteration.
        """
        keys = random.split(key, T + 2)

        params = self.pretraining(keys[-2])

        losses = []
        all_particles = []

        noise = self.noise_fn(keys[-1], (n, self.d))
        X_kgd = self.model_T.apply(params, noise)

        schedule = optax.exponential_decay(
            init_value=self.learning_rate, transition_steps=100, decay_rate=0.99
        )
        self.tx = optax.chain(
            optax.adam(learning_rate=self.learning_rate),
        )
        opt_state = self.tx.init(params)

        loss_ = jit(
            lambda key, params: self.loss(
                key, params, num_samples=n, d_noise=self.d, reg=2.0
            )
        )

        for step in tqdm(range(T)):
            key1, key2 = random.split(keys[step])

            all_particles.append(X_kgd)

            loss_grad_fn = jax.value_and_grad(lambda params: loss_(key1, params))
            loss_value, grads = loss_grad_fn(params)
            losses.append(loss_value)

            updates, opt_state = self.tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            noise_kgd = self.noise_fn(key2, (n, self.d))
            X_kgd = self.model_T.apply(params, noise_kgd)

            if step % 1000 == 0:
                X = self.model_T.apply(params, noise)
                mse = self.L(X) / gamma
                gradsize = (self.gradL(X) ** 2).sum() / gamma**2
                print(f"Step {step}, Loss: {loss_value}, MSE {mse}, Grad {gradsize}")

        return all_particles
