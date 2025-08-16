import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, vmap, grad
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from functions import F_P, GradientKernel,KernelGradientDiscrepancy
from jax import random
import time
import os
from jax import grad, random,vmap,tree_util
from jax.flatten_util import ravel_pytree

class VariationalInference:
    def __init__(self, q0, L, k,d):
        self.d = d
        self.q0 = q0
        self.log_q0 = jit(lambda x: jnp.log(self.q0(x)))
        self.S_q0 = jit(vmap(grad(self.log_q0)))
        self.L = L
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k_pq = GradientKernel(self.S_PQ, k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)


    def init_params(self,key, hidden_dim1=10,hidden_dim2 = 5):
        rng1, rng2,rng3,rng4,rng5,rng6 = random.split(key,6)
        params = {
            'W1': (random.uniform(rng1, (hidden_dim1, self.d))) * 10.0,
            'b1': random.normal(rng4, (hidden_dim1,)),
            'W2': (random.uniform(rng2, (hidden_dim2, hidden_dim1))) * 10.0,
            'b2': random.normal(rng3, (hidden_dim2,)),
            'W3' : (random.uniform(rng5, (self.d, hidden_dim2))) * 10.0,
            'b3' : random.normal(rng6, (self.d,))
        }
        return params

    
    def forward(self,params, w):
        h1 = jnp.tanh(jnp.dot(params['W1'], w) + params['b1'])  # hidden layer
        h2 = jnp.tanh(jnp.dot(params['W2'], h1) + params['b2'])  # hidden layer
        y = jnp.dot(params['W3'], h2) + params['b3']             
        return y.squeeze()

    def model(self,theta,W):
        key = random.PRNGKey(0)
        params = self.init_params(key)  
        flat_params, unravel_fn = ravel_pytree(params)
        param = unravel_fn(theta)
        return vmap(lambda w : self.forward(param, w))(W)


    def F(self,theta,key,n_sim,d):
            W = random.normal(key, shape=(n_sim, d))
            X = self.model(theta,W)
            return self.KGD.evaluate(X)

    def run(self,key,eta,T,n_sim=100):
            Keys = random.split(key, T+1)
            param0 = self.init_params(Keys[-1])
            theta,_ = ravel_pytree(param0)
            Theta = jnp.zeros((T, len(theta)))
            Theta = Theta.at[0].set(theta)
            for it in range(1,T):
                key = Keys[it]
                DF = jit(grad(lambda theta : self.F(theta, key, n_sim, self.d)))
                noise = 0#1e-7 * random.normal(key+10000, shape=theta.shape)
                theta = theta - eta * DF(theta + noise)
                Theta = Theta.at[it].set(theta)
            return Theta





    

