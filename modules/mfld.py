import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import jit, vmap, grad
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from modules.kgd_functions import F_P, GradientKernel,KernelGradientDiscrepancy
from jax import random
import time
import os




class MeanFieldLangevinDynamics:
    def __init__(self,grad_log_q0,L,gradL = None,):
        self.S_q0 = jit(grad_log_q0)
        self.L = L
        if gradL == None:
            self.gradL = jit(grad(lambda X : jax.lax.stop_gradient(len(X)) * self.L(X)))
        else:
            self.gradL = gradL
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X)) 

    def run_particles(self, eta, T, X0, key, noise=0):
    
        n, d = X0.shape
        
        device = X0.device  
        all_particles = jax.device_put(jnp.zeros((T, n, d)), device)
        X = X0.copy()

        keys_tab = random.split(key, T)
        for it in range(T):
            key, subkey = random.split(keys_tab[it])
            Z = random.normal(subkey, shape=(n, d))
            X = X + eta * self.S_PQ(X) + jnp.sqrt(2 * eta) * Z
            all_particles = all_particles.at[it].set(X)

        return all_particles




    

        
