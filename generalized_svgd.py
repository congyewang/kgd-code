import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap, grad
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from functions import F_P, GradientKernel,KernelGradientDiscrepancy
import jax_kernels as jk
import jax


class GeneralizedSVGD:
    def __init__(self,q0,L,k,l,dl):
        self.q0 = q0
        self.log_q0 = jit(lambda x: jnp.log(self.q0(x)))
        self.S_q0 = jit(vmap(grad(self.log_q0)))
        self.L = L
        self.gradL = jit(grad(lambda X : jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X)) # 
        self.k = k
        self.l = l
        self.dl = dl


        self.F_P = F_P(self.q0, self.L)
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)
    
    def run_particles(self, eta, T, X0):
        n, d = X0.shape
            
        KGD_values = jnp.zeros(T)
        F_P_values = jnp.zeros(T)
        all_particles = jnp.zeros((T, n, d))

        X = X0.copy()
        for it in range(T):
            phi = 1/n * (self.l(X,X) @ self.S_PQ(X) + jnp.sum(self.dl(X,X),axis = 0))
            X = X + eta * phi
            
            KGD_values = KGD_values.at[it].set(self.KGD.evaluate(X))
            all_particles = all_particles.at[it].set(X)
            F_P_values = F_P_values.at[it].set(self.F_P.evaluate(X))
        return KGD_values, F_P_values, all_particles
