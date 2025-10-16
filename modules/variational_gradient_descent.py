import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap, grad
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from modules.kgd_functions import F_P, GradientKernel,KernelGradientDiscrepancy
import jax
from jax import random
import optax 


class VariationalGradientDescent:
    def __init__(self,grad_log_q0,L,m,dm,gradL=None):
        self.S_q0 = jit(grad_log_q0)
        self.L = L
        if gradL == None:
            self.gradL = jit(grad(lambda X : jax.lax.stop_gradient(len(X)) * self.L(X)))
        else:
            self.gradL = gradL
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X)) # 
        self.m = m
        self.dm = dm
    
    def run_particles(self, eta, T, X0,optimise_stepsize= False):
        n, d = X0.shape
        
        device = X0.device  
        all_particles = jax.device_put(jnp.zeros((T, n, d)), device)
        X = X0.copy()
        if optimise_stepsize:
            optimizer = optax.adam(eta)
            opt_state = optimizer.init(X)

        for it in range(T):
            phi = 1/n * (self.m(X,X) @ self.S_PQ(X) + jnp.sum(self.dm(X,X),axis = 0))
            if optimise_stepsize:
                updates, opt_state = optimizer.update(-phi, opt_state)
                X = optax.apply_updates(X, updates)
            else:
                X = X + eta * phi
            all_particles = all_particles.at[it].set(X)

        return all_particles
