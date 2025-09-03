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




class MeanFieldLangevinDynamics:
    def __init__(self,q0,L,k,gradL = None):
        self.q0 = q0
        self.log_q0 = jit(lambda x: jnp.log(self.q0(x)))
        self.S_q0 = jit(vmap(grad(self.log_q0)))
        self.L = L
        if gradL == None:
            self.gradL = jit(grad(lambda X : jax.lax.stop_gradient(len(X)) * self.L(X)))
        else:
            self.gradL = gradL
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X)) # 
        self.k = k
    #     self.k_pq = GradientKernel(self.S_PQ, self.k)
    #     self.KGD = KernelGradientDiscrepancy(self.k_pq).evaluate

    # def optimize_eta_grid(self, X, Z, etas=jnp.logspace(-6, -3, 10)):
    #     losses = []
    #     for eta in etas:
    #         X_next = X + eta * self.S_PQ(X) + jnp.sqrt(2 * eta) * Z
    #         losses.append(self.KGD(X_next))
    #     losses = jnp.array(losses)
    #     safe_losses = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    #     return etas[jnp.argmin(safe_losses)]

    def run_particles(self, eta, T, X0, key, decrease_step_size = False, noise=0):
    
        n, d = X0.shape
        
        device = X0.device  
        all_particles = jax.device_put(jnp.zeros((T, n, d)), device)
        X = X0.copy()

        keys_tab = random.split(key, T)
        for it in range(T):
            key, subkey = random.split(keys_tab[it])
            Z = random.normal(subkey, shape=(n, d))

            if decrease_step_size:
                if (it+1) % 100 == 0:
                    eta = eta/4.0
            X = X + eta * self.S_PQ(X) + jnp.sqrt(2 * eta) * Z
            all_particles = all_particles.at[it].set(X)

        return all_particles




    

        
