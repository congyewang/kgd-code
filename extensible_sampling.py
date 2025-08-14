import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import jit, vmap, grad
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from functions import F_P, GradientKernel,KernelGradientDiscrepancy
from jax import random
import time
import os

class ExtensibleSampling:
    def __init__(self, q0, L, k):
        self.q0 = q0
        self.log_q0 = jit(lambda x: jnp.log(self.q0(x)))
        self.S_q0 = jit(vmap(grad(self.log_q0)))
        self.L = L
        self.k = k
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)

    def grid_search(self,X,l,u,n0,it):
        n, d = X.shape
        n_grid = n0 + int(jnp.sqrt(it))
        axis = jnp.linspace(l, u, n_grid)
        #print(axis)
        meshes = jnp.meshgrid(*[axis]*d, indexing="ij")
        grid_points = jnp.stack(meshes, axis=-1).reshape(-1, d) + 0.01 * random.normal(jax.random.PRNGKey(it), shape=(n_grid**d, d))  # Adding small noise to avoid exact duplicates
        KGD_grid = jnp.zeros(n_grid**d)
        for i in range(n_grid**d):
            #print(self.KGD.evaluate(jnp.concatenate((X, grid_points[i][None,:]), axis=0)))
            KGD_grid = KGD_grid.at[i].set(self.KGD.evaluate(jnp.concatenate((X, grid_points[i][None,:]), axis=0)))
        min_idx = jnp.argmin(KGD_grid)
        return grid_points[min_idx]
    
    #def MonteCarlo_search(self):

    #def GD_search(self,X):
         
        

    def run_particles(self,x0,T,l = -3,u = 3,n0 = 100):
        d = len(x0)
        X = jnp.zeros((T, d))
        X = X.at[0].set(x0)
        for it in range(1,T):
            if it % 10 == 0:
                print("Iteration:", it)
            X = X.at[it].set(self.grid_search(X[:it], l, u, n0, it))
        return X                        