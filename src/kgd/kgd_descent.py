from typing import Callable

import jax
import optax
from jax import grad, jit
from jaxtyping import Array, Float

from .kgd_functions import GradientKernel, KernelGradientDiscrepancy


class KGD_Descent:
    """
    KGD Descent class that performs optimization to minimize the Kernel Gradient Discrepancy (KGD) using gradient descent.
    """

    def __init__(
        self,
        grad_log_q0: Callable[[Float[Array, "sample"]], Float[Array, "gradient"]],
        L: Callable[[Float[Array, "sample"]], Float[Array, "loss"]],
        k: Callable[[Float[Array, "x1"], Float[Array, "x2"]], Float[Array, "kernel"]],
    ) -> None:
        """
        Initializes the KGD_Descent class.

        Args:
            grad_log_q0 (Callable[[Float[Array, "sample"]], Float[Array, "gradient"]]): Function to compute the gradient of the log density of the base distribution
            L (Callable[[Float[Array, "sample"]], Float[Array, "loss"]]): Loss function
            k (Callable[[Float[Array, "x1"], Float[Array, "x2"]], Float[Array, "kernel"]]): Kernel function
        """
        self.S_q0 = grad_log_q0
        self.L = L
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k = k
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)

    def run_particles(
        self,
        X0: Float[Array, "n d"],
        T: int,
        learning_rate: float = 1e-2,
        gamma: float = 300,
    ) -> list[Float[Array, "n d"]]:
        """
        Runs the KGD descent optimization.

        Args:
            X0 (Float[Array, "n d"]): Initial particles
            T (int): Number of optimization steps
            learning_rate (float): Learning rate for the optimizer
            gamma (float): Scaling factor for the loss computation

        Returns:
            list[Float[Array, "n d"]]: List of particles at each optimization step
        """
        tx = optax.adam(learning_rate)
        opt_state = tx.init(X0)
        X = X0.copy()

        loss_and_grad = jax.jit(jax.value_and_grad(self.KGD.evaluate))
        all_particles = [X]

        losses = []
        for step in range(T):
            loss_val, g = loss_and_grad(X)
            updates, opt_state = tx.update(g, opt_state)
            X = optax.apply_updates(X, updates)
            all_particles.append(X)

            if step % 10 == 0:
                mse = self.L(X) / gamma
                print(f"Step {step}, loss = {loss_val}, mse = {mse}")
                losses.append(loss_val)
        return all_particles
