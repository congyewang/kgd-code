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
        n_grid = n0 #+ int(jnp.sqrt(it))
        if it <= 30:
            axis = jnp.linspace(l, u, n_grid)
            #print(axis)
            meshes = jnp.meshgrid(*[axis]*d, indexing="ij")
            grid_points = jnp.stack(meshes, axis=-1).reshape(-1, d) + 1 * random.normal(jax.random.PRNGKey(it), shape=(n_grid**d, d))  # Adding small noise to avoid exact duplicates
            KGD_grid = jnp.zeros(n_grid**d)
            for i in range(n_grid**d):
                KGD_grid = KGD_grid.at[i].set(self.KGD.evaluate(jnp.concatenate((X, grid_points[i][None,:]), axis=0)))
            min_idx = jnp.argmin(KGD_grid)
        else:
            n_samples =  10000
            delta = 1.0
            key = jax.random.PRNGKey(it)

            # choisir aléatoirement quelles composantes de la mixture (i.e. quel point de X)
            comp_idx = random.randint(key, (n_samples,), minval=0, maxval=n)
            means = X[comp_idx]  # (n_samples, d)

            # bruit gaussien isotrope
            noise = jnp.sqrt(delta) * random.normal(key, (n_samples, d))

            grid_points = means + noise  # (n_samples, d)

            # évaluer le critère sur chaque point
            KGD_grid = jnp.zeros(n_samples)
            for i in range(n_samples):
                KGD_grid = KGD_grid.at[i].set(
                    self.KGD.evaluate(jnp.concatenate((X, grid_points[i][None, :]), axis=0))
                )
            min_idx = jnp.argmin(KGD_grid)
        return grid_points[min_idx]
    
    

    def GD_search(self,X,N_it=5000,step_size=0.01):
        Loss = lambda x : self.KGD.evaluate(jnp.concatenate((X, x[None,:]), axis=0))
        
        DLoss = jit(grad(Loss))
        x0 = jnp.mean(X, axis=0)
        x = x0
        for it in range(N_it):
            x = x - step_size * DLoss(x)
        return x


    def run_particles(self,x0,T,min_research = "grid_search",l = -5,u = 5,n0 = 10):
        d = len(x0)
        X = jnp.zeros((T, d))
        X = X.at[0].set(x0)
        for it in range(1,T):
            if it % 10 == 0:
                print("Iteration:", it)
            if min_research == "grid_search":
                X = X.at[it].set(self.grid_search(X[:it], l, u, n0, it))
            if min_research == "GD search":
                X = X.at[it].set(self.GD_search(X[:it]))
        return X                        