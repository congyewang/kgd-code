from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit, random, vmap
from jaxtyping import Array, Float


class GradientKernel:
    """
    Gradient Kernel class to compute the gradient-based kernel matrix.
    """

    def __init__(
        self,
        S_PQ: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
        k: Callable[
            [Float[Array, "x1"], Float[Array, "x2"]],
            Float[Array, "kernel"],
        ],
    ) -> None:
        """
        Initializes the GradientKernel class.

        Args:
            S_PQ (Callable[[Float[Array, "n d"]], Float[Array, "n d"]]): Function to compute the score function S_PQ
            k (Callable[[Float[Array, "x1"], Float[Array, "x2"]], Float[Array, "kernel"]]): Kernel function
        """
        self.S_PQ = S_PQ
        self.k = jit(k)
        self.dkx = jit(jacrev(self.k, argnums=0))
        self.dky = jit(jacrev(self.k, argnums=1))
        self.d2k = jit(jacfwd(self.dky, argnums=0))

        self.K = lambda X: vmap(lambda x: vmap(lambda y: self.k(x, y))(X))(X)
        self.dK1 = lambda X: vmap(lambda x: vmap(lambda y: self.dkx(x, y))(X))(X)
        self.d2K = lambda X: vmap(
            lambda x: vmap(lambda y: jnp.trace(self.d2k(x, y)))(X)
        )(X)

        # for extensible sampling
        self.Kx = lambda X, x: vmap(lambda y: self.k(x, y))(X)
        self.dK1x = lambda X, x: vmap(lambda y: self.dkx(x, y))(X)
        self.dK2x = lambda X, x: vmap(lambda y: self.dky(x, y))(X)
        self.d2Kx = lambda X, x: vmap(lambda y: jnp.trace(self.d2k(x, y)))(X)

    def gram_matrix(
        self,
        X: Float[Array, "n d"],
    ) -> Float[Array, "n n"]:
        """
        Computes the gradient-based kernel Gram matrix.

        Args:
            X (Float[Array, "n d"]): Input particles

        Returns:
            Float[Array, "n n"]: Gradient-based kernel Gram matrix
        """
        K = self.K(X)
        dK = self.dK1(X)
        d2K = self.d2K(X)
        S_PQ = self.S_PQ(X)
        S_dK = jnp.einsum("ijk, ijk -> ij", dK, (S_PQ[None, :, :]))
        k_pq = d2K + S_dK + S_dK.T + K * jnp.dot(S_PQ, S_PQ.T)

        return k_pq


class KernelGradientDiscrepancy:
    """
    Kernel Gradient Discrepancy (KGD) class to evaluate
    the discrepancy using the gradient-based kernel.
    """

    def __init__(
        self,
        k_pq: GradientKernel,
    ) -> None:
        """
        Initializes the KernelGradientDiscrepancy class.

        Args:
            k_pq (GradientKernel): GradientKernel instance
        """
        self.K_pq = jit(k_pq.gram_matrix)
        self.k_pq = k_pq

    def evaluate(
        self,
        X: Float[Array, "n d"],
    ) -> Float[Array, ""]:
        """
        Evaluates the Kernel Gradient Discrepancy (KGD) for the given particles.

        Args:
            X (Float[Array, "n d"]): Input particles

        Returns:
            Float[Array, ""]: KGD value
        """
        n = len(X)
        K_pq = self.K_pq(X)
        sum = 1 / n * jnp.sqrt(jnp.sum(K_pq))

        return sum

    def square_kgd(
        self,
        X: Float[Array, "n d"],
    ) -> Float[Array, ""]:
        """
        Evaluates the squared Kernel Gradient Discrepancy (KGD) for the given particles.

        Args:
            X (Float[Array, "n d"]): Input particles

        Returns:
            Float[Array, ""]: Squared KGD value
        """
        n = len(X)
        K_pq = self.K_pq(X)
        sum = jnp.mean(K_pq)

        return sum

    def kde_KGD(
        self,
        X: Float[Array, "n d"],
        num_samples: int = 100,
    ) -> Float[Array, ""]:
        """
        Evaluates the KGD using Kernel Density Estimation (KDE) for the given particles.

        Args:
            X (Float[Array, "n d"]): Input particles
            num_samples (int): Number of samples for KDE

        Returns:
            Float[Array, ""]: KGD value using KDE
        """
        n, d = X.shape
        bandwidth = 1 / jnp.sqrt(n)
        key = random.PRNGKey(0)
        key, key_idx, key_noise = random.split(key, 3)
        indices = random.choice(key_idx, n, shape=(num_samples,), replace=True)
        samples = random.normal(key_noise, (num_samples, d)) * bandwidth + X[indices]
        samples = jax.device_put(samples, X.device)
        return self.evaluate(samples)


class F_P:
    """
    Variational objective function class that combines a loss function with a KL divergence term estimated via KDE.
    """

    def __init__(
        self,
        q0: Callable[[Float[Array, "d"]], Float[Array, ""]],
        L: Callable[[Float[Array, "n d"], Float[Array, "n d"]], Float[Array, ""]],
    ) -> None:
        """
        Initializes the F_P class.

        Args:
            q0 (Callable[[Float[Array, "d"]], Float[Array, ""]]): Base distribution density function
            L (Callable[[Float[Array, "n d"], Float[Array, "n d"]], Float[Array, ""]]): Loss function
        """
        self.q0 = q0
        self.Q_0 = jit(vmap(self.q0))
        self.L = L

    def kl_divergence_kde(
        self,
        X: Float[Array, "n d"],
        num_samples: int,
    ) -> Float[Array, ""]:
        """
        Estimates the KL divergence between the empirical distribution of particles and the base distribution using Kernel Density Estimation (KDE).

        Args:
            X (Float[Array, "n d"]): Input particles
            num_samples (int): Number of samples for KDE

        Returns:
            Float[Array, ""]: Estimated KL divergence
        """
        n, d = X.shape

        def kde_density(
            data: Float[Array, "n d"],
            points: Float[Array, "m d"],
            bandwidth: float,
        ) -> Float[Array, "m"]:
            """
            KDE density estimation.

            Args:
                data (Float[Array, "n d"]): Data points
                points (Float[Array, "m d"]): Points where density is estimated
                bandwidth (float): Bandwidth for the kernel

            Returns:
                Float[Array, "m"]: Estimated log densities at the given points
            """
            diff = jnp.expand_dims(points, axis=1) - jnp.expand_dims(data, axis=0)
            dist = jnp.linalg.norm(diff, axis=2)
            weights = jnp.exp(-(dist**2) / (2 * bandwidth**2))
            density = jnp.sum(weights, axis=1) / (
                n * (bandwidth * jnp.sqrt(2 * jnp.pi)) ** d
            )

            return jnp.log(jnp.maximum(density, 1e-10))  # Sécurisé contre log(0)

        bandwidth = 1 / jnp.sqrt(n)

        key = random.PRNGKey(0)
        key, key_idx, key_noise = random.split(key, 3)
        indices = random.choice(key_idx, n, shape=(num_samples,), replace=True)
        samples = random.normal(key_noise, (num_samples, d)) * bandwidth + X[indices]
        samples = jax.device_put(samples, X.device)

        log_qn = kde_density(X, samples, bandwidth)
        log_q0 = jnp.log(jnp.maximum(self.Q_0(samples), 1e-10))

        kl_estimate = jnp.mean(log_qn - log_q0)

        return kl_estimate

    def evaluate(
        self,
        X: Float[Array, "n d"],
        num_samples: int = 50,
    ) -> Float[Array, ""]:
        """
        Evaluates the variational objective function F_P.

        Args:
            X (Float[Array, "n d"]): Input particles
            num_samples (int): Number of samples for KL divergence estimation

        Returns:
            Float[Array, ""]: Value of the variational objective function F_P
        """
        KL = self.kl_divergence_kde(X, num_samples)

        return self.L(X) + KL


def MMD(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    k: Callable[[Float[Array, "d"], Float[Array, "d"]], Float[Array, ""]],
) -> Float[Array, ""]:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        x (Float[Array, "n d"]): First set of samples
        y (Float[Array, "m d"]): Second set of samples
        k (Callable[[Float[Array, "d"], Float[Array, "d"]], Float[Array, ""]]): Kernel function

    Returns:
        Float[Array, ""]: MMD value
    """
    n = len(x)
    K = lambda X, Y: vmap(lambda x1: vmap(lambda x2: k(x1, x2))(X))(Y)
    Kxx = K(x, x)
    Kyy = K(y, y)
    Kxy = K(x, y)
    A = 1 / ((n - 1) * n) * (np.sum(Kxx) - np.sum(np.diag(Kxx)))
    C = 1 / ((n - 1) * n) * (np.sum(Kyy) - np.sum(np.diag(Kyy)))
    B = 1 / n**2 * np.sum(Kxy)

    return A - B + C


############################
######## KERNELS ###########
############################


def k_imq(
    x: Float[Array, "d"], y: Float[Array, "d"], c: float, b: float, scale: float = 1.0
) -> float:
    """
    Inverse Multi-Quadratic (IMQ) kernel function.

    Args:
        x (Float[Array, "d"]): First input vector
        y (Float[Array, "d"]): Second input vector
        c (float): Constant parameter
        b (float): Exponent parameter (must be positive)
        scale (float): Scale parameter

    Returns:
        float: IMQ kernel value
    """
    if b <= 0:
        raise ValueError("Parameter 'b' must be positive.")

    res = (c**2 + (x - y).dot(x - y) / scale**2) ** (-b)

    return res


def imq_scale_mixtures(
    x: Float[Array, "d"],
    y: Float[Array, "d"],
    scales: Float[Array, "s"],
) -> float:
    """
    Computes the IMQ kernel between two vectors with varying scales.

    Args:
        x (Float[Array, "d"]): First input vector
        y (Float[Array, "d"]): Second input vector
        scales (Float[Array, "s"]): Array of scale parameters

    Returns:
        float: Averaged IMQ kernel value over the provided scales
    """
    return vmap(lambda scale: k_imq(x, y, 1, 0.5, scale))(scales).mean()


def gaussian_kernel(
    x: Float[Array, "d"],
    y: Float[Array, "d"],
    sigma: float,
) -> float:
    """
    Computes the Gaussian kernel between two vectors.

    Args:
        x (Float[Array, "d"]): First input vector
        y (Float[Array, "d"]): Second input vector
        sigma (float): Standard deviation parameter

    Returns:
        float: Gaussian kernel value
    """
    return jnp.exp(-(x - y).dot(x - y) / (2 * sigma**2))


def matern_kernel(
    x: Float[Array, "d"],
    y: Float[Array, "d"],
    length_scale: float,
) -> float:
    """
    Matérn kernel function with nu=3/2.

    Args:
        x (Float[Array, "d"]): First input vector
        y (Float[Array, "d"]: Second input vector
        length_scale (float): Length scale parameter

    Returns:
        float: Matérn kernel value
    """
    eps = 1e-14
    d = ((x - y).dot(x - y) + eps) ** 0.5
    return (1 + jnp.sqrt(3) * d / length_scale) * jnp.exp(
        -jnp.sqrt(3) * d / length_scale
    )


######## Recommended kernel #########


def k_lin(
    x: Float[Array, "d"],
    y: Float[Array, "d"],
    c: float,
) -> float:
    """
    Computes the linear kernel between two vectors.

    Args:
        x (Float[Array, "d"]): First input vector
        y (Float[Array, "d"]): Second input vector
        c (float): Constant parameter

    Returns:
        float: Linear kernel value
    """
    return jnp.dot(x, y) + c**2


def a_(
    x: Float[Array, "d"],
    s: float,
    c: float,
) -> float:
    """
    Computes the scaling function a_s(x).

    Args:
        x (Float[Array, "d"]): Input vector
        s (float): Exponent parameter
        c (float): Constant parameter

    Returns:
        float: Value of the scaling function a_s(x)
    """
    return (c**2 + jnp.sum((x) ** 2)) ** (s / 2)


def recommended_kernel(
    x: Float[Array, "d"],
    y: Float[Array, "d"],
    L: Callable[[Float[Array, "d"], Float[Array, "d"]], float],
    alpha: float,
    beta: float,
    c: float,
) -> float:
    """
    Computes the recommended kernel between two vectors.

    Args:
        x (Float[Array, "d"]): First input vector
        y (Float[Array, "d"]): Second input vector
        L (Callable[[Float[Array, "d"], Float[Array, "d"]], float]): Loss function
        alpha (float): Exponent parameter for scaling function
        beta (float): Exponent parameter for scaling function
        c (float): Constant parameter

    Returns:
        float: Recommended kernel value
    """
    a_s_x = a_(x, alpha - beta, c)
    a_s_y = a_(y, alpha - beta, c)
    k_lin_xy = k_lin(x, y, c) / (k_lin(x, x, c) * k_lin(y, y, c)) ** 0.5
    return a_s_x * (L(x, y) + k_lin_xy) * a_s_y


############################################@
