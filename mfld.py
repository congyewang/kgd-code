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





class MeanFieldLangevinDynamics:
    def __init__(self,q0,L,k):
        self.q0 = q0
        self.log_q0 = jit(lambda x: jnp.log(self.q0(x)))
        self.S_q0 = jit(vmap(grad(self.log_q0)))
        self.L = L
        self.gradL = jit(grad(lambda X : jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X)) # 
        self.k = k

        self.F_P = F_P(self.q0, self.L)
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)


    def run_particles(self,eta,T,X0,noise = 0):###@
        n,d = X0.shape
        KGD_values = jnp.zeros(T)
        F_P_values = jnp.zeros(T)
        all_particles = jnp.zeros((T, n, d))

        X = X0.copy()
        for it in range(T):
            Z = random.normal(random.PRNGKey(it), shape=(n,d))  
            eps = random.normal(random.PRNGKey(it+ 1000), shape=(n,d))
            X = X + eta * self.S_PQ(X) + jnp.sqrt(2*eta)* Z # + 1/2**(it//100) * noise * eps)
            
            
            KGD_values = KGD_values.at[it].set(self.KGD.evaluate(X))
            all_particles = all_particles.at[it].set(X)
            F_P_values = F_P_values.at[it].set(jit(self.F_P.evaluate)(X))


            # if it % 10 ==0:
            #     plt.figure()
            #     plt.scatter(all_particles[it,:,0],all_particles[it,:,0])
        return KGD_values, F_P_values, all_particles




    

        
