import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from jax.scipy.stats import gaussian_kde
from jax import random
import jax
import time




class GradientKernel:
    def __init__(self, S_PQ, k):
        self.S_PQ = S_PQ
        self.k = jit(k)
        self.dkx = jit(jacrev(self.k, argnums=0))
        self.dky = jit(jacrev(self.k, argnums=1))
        self.d2k = jit(jacfwd(self.dky, argnums=0))

        self.K = lambda X : vmap(lambda x: vmap(lambda y: self.k(x, y))(X))(X)
        self.dK1 = lambda X : vmap(lambda x: vmap(lambda y: self.dkx(x, y))(X))(X)
        self.d2K = lambda X : vmap(lambda x: vmap(lambda y: jnp.trace(self.d2k(x, y)))(X))(X)

        #for extensible sampling
        self.Kx = lambda X,x : vmap(lambda y: self.k(x, y))(X)
        self.dK1x = lambda X,x : vmap(lambda y: self.dkx(x, y))(X)
        self.dK2x = lambda X,x : vmap(lambda y : self.dky(x, y))(X)
        self.d2Kx = lambda X,x : vmap(lambda y: jnp.trace(self.d2k(x, y)))(X)


    def gram_matrix(self,X): #Gram_matrix
        K = self.K(X)
        dK = self.dK1(X)
        d2K = self.d2K(X)
        S_PQ = self.S_PQ(X)
        S_dK = jnp.einsum('ijk, ijk -> ij', dK, (S_PQ[None, :, :]))
        #S_dK = jnp.einsum('ijk, ijk -> ij', dK, (S_PQ[None, :, :] - S_PQ[:, None, :]))
        k_pq = d2K + S_dK + S_dK.T + K * jnp.dot(S_PQ, S_PQ.T)
        return k_pq
    
    #for extansible sampling
    def kernel_array(self,X):
        n = len(X)
        S_PQ = self.S_PQ(X)
        Y = X[:n-1]
        x = X[n-1]
        return self.Kx(Y,x).T @ (S_PQ[:n-1] @ S_PQ[n-1]) + self.d2Kx(Y,x) + jnp.sum(self.dK1x(Y,x)* S_PQ[:n-1],axis = 1) + jnp.sum(self.dK2x(Y,x)* S_PQ[n-1],axis = 1) 
    def kernel_function(self,X): # k_PQ(x,x))
        n = len(X)
        S_PQ = self.S_PQ(X)
        x = X[n-1]
        return self.k(x,x) * jnp.dot(S_PQ[n-1],S_PQ[n-1]) + jnp.trace(self.d2k(x,x)) + jnp.dot(self.dkx(x,x), S_PQ[n-1]) + jnp.dot(self.dky(x,x), S_PQ[n-1]) 



class KernelGradientDiscrepancy:
    def __init__(self, k_pq):
        self.K_pq = jit(k_pq.gram_matrix)
        self.k_pq = k_pq
        #self.vfk0 = jit(vmap(k0.evaluate, in_axes=(None, 0, 0)))

    def evaluate(self, X):
        n = len(X)
        K_pq = self.K_pq(X)
        sum = 1/n * jnp.sqrt(jnp.sum(K_pq))
        return sum
    
    def kde_KGD(self,X,num_samples=100):
        n,d = X.shape
        bandwidth = 1/jnp.sqrt(n)  
        key = random.PRNGKey(0)
        key, key_idx, key_noise = random.split(key, 3)
        indices = random.choice(key_idx, n, shape=(num_samples,), replace=True)
        samples = random.normal(key_noise, (num_samples, d)) * bandwidth + X[indices]
        samples = jax.device_put(samples, X.device)
        return self.evaluate(samples)

    #for extansible sampling 
    def minimized_function(self,X):
        n = len(X)
        #print(self.k_pq.kernel_function(X)/2 )+ jnp.sum(self.k_pq.kernel_array(X)))
        return self.k_pq.kernel_function(X)/2 + jnp.sum(self.k_pq.kernel_array(X)) 


        
     
class F_P:
    def __init__(self, q0, L):
        self.q0 = q0
        self.Q_0 = jit(vmap(self.q0))
        self.L = L
    
    def kl_divergence_kde(self, X, num_samples):
        n, d = X.shape

        def kde_density(data, points, bandwidth):
            diff = jnp.expand_dims(points, axis=1) - jnp.expand_dims(data, axis=0)
            dist = jnp.linalg.norm(diff, axis=2)
            weights = jnp.exp(-dist**2 / (2 * bandwidth**2))
            density = jnp.sum(weights, axis=1) / (n * (bandwidth * jnp.sqrt(2 * jnp.pi))**d)
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

    def evaluate(self, X, num_samples = 50):
        KL = self.kl_divergence_kde(X,num_samples)
        return self.L(X) + KL
    
    
        






