from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jaxtyping import Array, Float


class MeanFieldLangevinDynamics:
    """
    Mean-Field Langevin Dynamics sampler.
    """

    def __init__(
        self,
        grad_log_q0: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
        L: Callable[[Float[Array, "n d"]], Float[Array, "n"]],
        gradL: Optional[Callable[[Float[Array, "n d"]], Float[Array, "n d"]]] = None,
    ) -> None:
        """
        Mean-Field Langevin Dynamics sampler.

        Args:
            grad_log_q0 (Callable[[Float[Array, "n d"]], Float[Array, "n d"]]): Function computing the gradient of the log-density of the initial distribution q0.
            L (Callable[[Float[Array, "n d"]], Float[Array, "n"]]): Function computing the interaction potential.
            gradL (Optional[Callable[[Float[Array, "n d"]], Float[Array, "n d"]]]): Optional function computing the gradient of the interaction potential. If None, it will be computed using automatic differentiation.
        """
        self.S_q0 = jit(grad_log_q0)
        self.L = L
        if gradL is None:
            self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        else:
            self.gradL = gradL
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))

    def run_particles(
        self,
        eta: float,
        T: int,
        X0: Float[Array, "n d"],
        key: Float[Array, "key_dim"],
        noise: float = 0,
    ) -> Float[Array, "T n d"]:
        """
        Run the Mean-Field Langevin Dynamics sampler.

        Args:
            eta (float): Step size.
            T (int): Number of iterations.
            X0 (Float[Array, "n d"]): Initial particles of shape (n, d).
            key (Float[Array, "key_dim"]): JAX random key.
            noise (float): Additional noise term (default is 0).

        Returns:
            Float[Array, "T n d"]: Array of shape (T, n, d) containing the particles at each iteration.
        """
        n, d = X0.shape

        device = X0.device
        all_particles = jax.device_put(jnp.zeros((T, n, d)), device)
        X = X0.copy()

        keys_tab = random.split(key, T)
        for it in range(T):
            key, subkey = random.split(keys_tab[it])
            Z = random.normal(subkey, shape=(n, d))
            X = X + eta * self.S_PQ(X) + jnp.sqrt(2 * eta) * Z
            all_particles = all_particles.at[it].set(X)

        return all_particles
