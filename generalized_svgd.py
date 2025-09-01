import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap, grad
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from functions import F_P, GradientKernel,KernelGradientDiscrepancy
import jax_kernels as jk
import jax
from jax import random


class GeneralizedSVGD:
    def __init__(self,q0,L,k,m,dm,gradL=None):
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
        self.m = m
        self.dm = dm
    
    def run_particles(self, eta, T, X0,decrease_step_size = False):
        n, d = X0.shape
        
        device = X0.device  
        all_particles = jax.device_put(jnp.zeros((T, n, d)), device)
        X = X0.copy()
        

        #keys_tab = random.split(key, T)
        for it in range(T):
            #key, subkey = random.split(keys_tab[it])
            phi = 1/n * (self.m(X,X) @ self.S_PQ(X) + jnp.sum(self.dm(X,X),axis = 0))
            if decrease_step_size:
                if (it+1) % 200 == 0:
                    eta = eta/1.5
            X = X + eta * phi
            all_particles = all_particles.at[it].set(X)

        return all_particles
