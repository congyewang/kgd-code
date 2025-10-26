from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax
from jax import grad, jit
from jaxtyping import Array, Float


class VariationalGradientDescent:
    """
    Variational Gradient Descent sampler.
    """

    def __init__(
        self,
        grad_log_q0: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
        L: Callable[[Float[Array, "n d"]], Float[Array, "n"]],
        m: Callable[[Float[Array, "n d"], Float[Array, "n d"]], Float[Array, "n d"]],
        dm: Callable[[Float[Array, "n d"], Float[Array, "n d"]], Float[Array, "n d"]],
        gradL: Optional[Callable[[Float[Array, "n d"]], Float[Array, "n d"]]] = None,
    ) -> None:
        """
        Variational Gradient Descent sampler.

        Args:
            grad_log_q0 (Callable[[Float[Array, "n d"]], Float[Array, "n d"]]): Function computing the gradient of the log-density of the initial distribution q0.
            L (Callable[[Float[Array, "n d"]], Float[Array, "n"]]): Function computing the interaction potential.
            m (Callable[[Float[Array, "n d"], Float[Array, "n d"]], Float[Array, "n d"]]): Function computing the mean-field interaction term.
            dm (Callable[[Float[Array, "n d"], Float[Array, "n d"]], Float[Array, "n d"]]): Function computing the gradient of the mean-field interaction term.
            gradL (Optional[Callable[[Float[Array, "n d"]], Float[Array, "n d"]]]): Optional function computing the gradient of the interaction potential. If None, it will be computed using automatic differentiation.
        """
        self.S_q0 = jit(grad_log_q0)
        self.L = L
        if gradL is None:
            self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        else:
            self.gradL = gradL
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))  #
        self.m = m
        self.dm = dm

    def run_particles(
        self,
        eta: float,
        T: int,
        X0: Float[Array, "n d"],
        optimise_stepsize: bool = False,
    ) -> Float[Array, "T n d"]:
        """
        Run the Variational Gradient Descent sampler.

        Args:
            eta (float): Step size.
            T (int): Number of iterations.
            X0 (Float[Array, "n d"]): Initial particles of shape (n, d).

        Returns:
            Float[Array, "T n d"]: Array of shape (T, n, d) containing the particles at each iteration.
        """
        n, d = X0.shape

        device = X0.device
        all_particles = jax.device_put(jnp.zeros((T, n, d)), device)
        X = X0.copy()
        if optimise_stepsize:
            optimizer = optax.adam(eta)
            opt_state = optimizer.init(X)

        for it in range(T):
            phi = 1 / n * (self.m(X, X) @ self.S_PQ(X) + jnp.sum(self.dm(X, X), axis=0))
            if optimise_stepsize:
                updates, opt_state = optimizer.update(-phi, opt_state)
                X = optax.apply_updates(X, updates)
            else:
                X = X + eta * phi
            all_particles = all_particles.at[it].set(X)

        return all_particles
