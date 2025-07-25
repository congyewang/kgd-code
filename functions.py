import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from jax import random
import time




class GradientKernel:
    def __init__(self, S_PQ, k):
        self.S_PQ = S_PQ
        self.k = k
        self.dkx = jacrev(k, argnums=0)
        self.dky = jit(jacrev(k, argnums=1))
        self.d2k = jit(jacfwd(self.dky, argnums=0))

        self.K = lambda X : vmap(lambda x: vmap(lambda y: self.k(x, y))(X))(X)
        self.dKx = lambda X : vmap(lambda x: vmap(lambda y: self.dkx(x, y))(X))(X)
        self.d2K = lambda X : vmap(lambda x: vmap(lambda y: jnp.trace(self.d2k(x, y)))(X))(X)


    def evaluate(self,X):
        K = self.K(X)
        dK = self.dKx(X)
        d2K = self.d2K(X)
        S_PQ = self.S_PQ(X)
        S_dK = jnp.einsum('ijk, ijk -> ij', dK, (S_PQ[None, :, :] - S_PQ[:, None, :]))
        k_pq = d2K + S_dK + K * jnp.dot(S_PQ, S_PQ.T)
        return k_pq




class KernelGradientDiscrepancy:
    def __init__(self, k_pq):
        self.k_pq = jit(k_pq.evaluate)
        #self.vfk0 = jit(vmap(k0.evaluate, in_axes=(None, 0, 0)))

    def evaluate(self, X):
        n = len(X)
        K_pq = self.k_pq(X)
        sum = 1/n * jnp.sqrt(jnp.sum(K_pq))
        return sum



          

class F_P:
    def __init__(self, q0, L):
        self.q0 = q0
        self.Q_0 = jit(vmap(self.q0))
        self.L = L
    
    def kl_divergence_kde(self,X, num_samples=500):
        n, d = X.shape
        
        # KDE approximation
        def kde_density(data, points, bandwidth=1.0):
            diff = jnp.expand_dims(points, axis=1) - jnp.expand_dims(data, axis=0)  # (num_samples, n, d)
            dist = jnp.linalg.norm(diff, axis=2)  # (num_samples, n)
            weights = jnp.exp(-dist**2 / (2 * bandwidth**2))  # (num_samples, n)
            density = jnp.sum(weights, axis=1) / (n * (bandwidth * jnp.sqrt(2 * jnp.pi))**d)
            return jnp.log(density)

        # Resample points from KDE
        key = random.PRNGKey(0)  # Clé pseudo-aléatoire
        indices = random.choice(key, n, shape=(num_samples,), replace=True)  # Échantillonnage
        samples = X[indices]
        log_qn = kde_density(X, samples)  # Log density of KDE at sampled points

        # Log density of standard Gaussian
        log_q0 = jnp.log(jnp.maximum(self.Q_0(samples),1e-10)) #-d/2 *jnp.log(2* jnp.pi) #multivariate_normal.logpdf(samples, mean=jnp.zeros(d), cov=jnp.eye(d))

        # KL divergence estimate
        kl_estimate = jnp.mean(log_qn - log_q0)
        return kl_estimate

    def evaluate(self, X):
        KL = self.kl_divergence_kde(X)
        return self.L(X) + KL
        




