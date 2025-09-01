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
                if (it+1) % 200 == 0:
                    eta = eta/1.5
            X = X + eta * self.S_PQ(X) + jnp.sqrt(2 * eta) * Z
            all_particles = all_particles.at[it].set(X)

        # if save_data_name != None:
        #     save_dir = "saved_data/mfld"
        #     os.makedirs(save_dir, exist_ok=True)    
        #     save_path = os.path.join(save_dir, save_data_name)
        #     jnp.save(save_path, all_particles[jnp.arange(0,T,100)])

        return all_particles




    

        
