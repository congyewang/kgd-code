from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jaxtyping import Array, Float

from .kgd_functions import GradientKernel, KernelGradientDiscrepancy


class ExtensibleSampling:
    """
    Extensible Sampling class implementing various search methods to find new sample points that minimize the Kernel Gradient Discrepancy (KGD).
    """

    def __init__(
        self,
        grad_log_q0: Callable[[Float[Array, "sample"]], Float[Array, "gradient"]],
        L: Callable[[Float[Array, "sample"]], Float[Array, "loss"]],
        k: Callable[[Float[Array, "x1"], Float[Array, "x2"]], Float[Array, "kernel"]],
        gradL: Optional[
            Callable[[Float[Array, "sample"]], Float[Array, "gradient"]]
        ] = None,
    ) -> None:
        """
        Initializes the ExtensibleSampling class.

        Args:
            grad_log_q0 (Callable[[Float[Array, "sample"]], Float[Array, "gradient"]]): Function to compute the gradient of the log density of the base distribution
            L (Callable[[Float[Array, "sample"]], Float[Array, "loss"]]): Loss function
            k (Callable[[Float[Array, "x1"], Float[Array, "x2"]], Float[Array, "kernel"]]): Kernel function
            gradL (Optional[Callable[[Float[Array, "sample"]], Float[Array, "gradient"]]]): Optional function to compute the gradient of the loss function. If None, it is computed using automatic differentiation.
        """
        self.S_q0 = jit(grad_log_q0)
        self.L = L
        self.k = k
        if gradL == None:
            self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        else:
            self.gradL = gradL
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)

    def grid_search(
        self,
        key: Float[Array, "key_dim"],
        X: Float[Array, "n d"],
        l: float,
        u: float,
        noise_std: float,
        delta: float,
        n_grid: int,
        it: int,
    ) -> Float[Array, "d"]:
        """
        Performs a grid search to find the point that minimizes the KGD.

        Args:
            key (Float[Array, "key_dim"]): JAX random key
            X (Float[Array, "n d"]): Current set of particles
            l (float): Lower bound of the grid
            u (float): Upper bound of the grid
            noise_std (float): Standard deviation of the noise added to grid points
            delta (float): Variance for sampling in higher dimensions
            n_grid (int): Number of grid points per dimension
            it (int): Current iteration number

        Returns:
            Float[Array, "d"]: The point that minimizes the KGD.
        """
        n, d = X.shape
        if it <= 30:
            axis = jnp.linspace(l, u, n_grid)
            meshes = jnp.meshgrid(*[axis] * d, indexing="ij")
            grid_points = jnp.stack(meshes, axis=-1).reshape(
                -1, d
            ) + noise_std * random.normal(
                key, shape=(n_grid**d, d)
            )  # Adding small noise to avoid exact duplicates
            KGD_grid = jnp.zeros(n_grid**d)
            for i in range(n_grid**d):
                KGD_grid = KGD_grid.at[i].set(
                    self.KGD.evaluate(
                        jnp.concatenate((X, grid_points[i][None, :]), axis=0)
                    )
                )
            min_idx = jnp.argmin(KGD_grid)
        else:
            # From number of particles > 30 we sample from a mixture of gaussian whose means are particles already sampled

            n_samples = 10**d
            key, subkey = random.split(key)

            comp_idx = random.randint(key, (n_samples,), minval=0, maxval=n)
            means = X[comp_idx]

            noise = jnp.sqrt(delta) * random.normal(subkey, (n_samples, d))

            grid_points = means + noise  # (n_samples, d)

            KGD_grid = jnp.zeros(n_samples)
            for i in range(n_samples):
                KGD_grid = KGD_grid.at[i].set(
                    self.KGD.evaluate(
                        jnp.concatenate((X, grid_points[i][None, :]), axis=0)
                    )
                )
            min_idx = jnp.argmin(KGD_grid)
        return grid_points[min_idx]

    def prior_search(
        self,
        key: Float[Array, "key_dim"],
        X: Float[Array, "n d"],
        mean: Float[Array, "d"],
        cov: Float[Array, "d d"],
        n_samples: int,
        it: int,
    ) -> Float[Array, "d"]:
        """
        Performs a prior-based search to find the point that minimizes the KGD.

        Args:
            key (Float[Array, "key_dim"]): JAX random key
            X (Float[Array, "n d"]): Current set of particles
            mean (Float[Array, "d"]): Mean of the prior distribution
            cov (Float[Array, "d d"]): Covariance matrix of the prior distribution
            n_samples (int): Number of samples to draw from the prior
            it (int): Current iteration number

        Returns:
            Float[Array, "d"]: The point that minimizes the KGD.
        """
        d = X.shape[1]
        points = cov * random.normal(key, (n_samples, d)) + mean
        KGD_points = jnp.zeros(n_samples)
        for i in range(n_samples):
            KGD_points = KGD_points.at[i].set(
                self.KGD.evaluate(jnp.concatenate((X, points[i][None, :]), axis=0))
            )
        min_idx = jnp.argmin(KGD_points)
        return points[min_idx]

    def gd_search(
        self,
        X: Float[Array, "n d"],
        N_it: int = 5000,
        step_size: float = 0.01,
    ) -> Float[Array, "d"]:
        """
        Performs a gradient descent search to find the point that minimizes the KGD.

        Args:
            X (Float[Array, "n d"]): Current set of particles
            N_it (int): Number of gradient descent iterations
            step_size (float): Step size for gradient descent

        Returns:
            Float[Array, "d"]: The point that minimizes the KGD.
        """
        Loss = lambda x: self.KGD.evaluate(jnp.concatenate((X, x[None, :]), axis=0))

        DLoss = jit(grad(Loss))
        x0 = jnp.mean(X, axis=0)
        x = x0
        for it in range(N_it):
            x = x - step_size * DLoss(x)
        return x

    def run_particles(
        self,
        key: Float[Array, "key_dim"],
        x0: Float[Array, "d"],
        T: int,
        min_search_method: str = "grid_search",
        settings: list[float] = [-5, 5, 0.1, 0.5, 20],
    ) -> Float[Array, "T d"]:
        """
        Runs the extensible sampling process to find T new sample points.

        Args:
            key (Float[Array, "key_dim"]): JAX random key
            x0 (Float[Array, "d"]): Initial point
            T (int): Number of new sample points to find
            min_search_method (str): Method to use for searching new points ("grid_search", "GD search", "prior_search")
            settings (list[float]): Settings for the search methods

        Returns:
            Float[Array, "T d"]: Array of new sample points found.
        """
        d = len(x0)
        X = jnp.zeros((T, d))
        X = X.at[0].set(x0)

        keys = random.split(key, T)
        for it in range(1, T):
            if it % 10 == 0:
                print("Iteration:", it)
            if min_search_method == "grid_search":
                X = X.at[it].set(
                    self.grid_search(
                        keys[it],
                        X[:it],
                        settings[0],
                        settings[1],
                        settings[2],
                        settings[3],
                        settings[4],
                        it,
                    )
                )
            if min_search_method == "GD search":
                X = X.at[it].set(self.gd_search(X[:it]))
            if min_search_method == "prior_search":
                X = X.at[it].set(
                    self.prior_search(
                        keys[it], X[:it], settings[0], settings[1], settings[2], it
                    )
                )
        return X
