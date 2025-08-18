import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, vmap, grad
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
import functions as f
from functions import F_P, GradientKernel,KernelGradientDiscrepancy
from jax import random
import time
import os
from jax import grad, random,vmap,tree_util
from jax.flatten_util import ravel_pytree
from flax import linen as nn

class VariationalInference:
    def __init__(self, q0, L, k,d,layers,activation_function):
        self.d = d
        self.q0 = q0
        self.log_q0 = jit(lambda x: jnp.log(self.q0(x)))
        self.S_q0 = jit(vmap(grad(self.log_q0)))
        self.L = L
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k_pq = GradientKernel(self.S_PQ, k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)
        self.model_T = f.MLP(layers,self.d,activation_function)

        key = random.PRNGKey(0)
        params = self.model_T.init(key, jnp.zeros((self.d,)))  
        self.unravel_fn = ravel_pytree(params)[1]

    
    def push_T_theta(self,theta,W):
         param = self.unravel_fn(theta)
         return vmap(lambda w : self.model_T.apply(param, w))(W)
        
    def F(self,theta,key,n_sim,d):
            W = random.normal(key, shape=(n_sim, d))
            X = self.push_T_theta(theta,W)
            return self.KGD.evaluate(X)

    def run(self,key,eta,T,n_sim=100):
            Keys = random.split(key, T+1)
            param0 = self.model_T.init(Keys[-1], jnp.zeros((self.d,)))
            theta,_ = ravel_pytree(param0)
            Theta = jnp.zeros((T, len(theta)))
            Theta = Theta.at[0].set(theta)
            for it in range(1,T):
                key = Keys[0]#it]
                DF = grad(lambda theta : self.F(theta, key, n_sim, self.d))
                noise = 0 #1e-7 * random.normal(key+10000, shape=theta.shape)
                theta = theta - eta * DF(theta + noise)
                Theta = Theta.at[it].set(theta)
            return Theta





    

