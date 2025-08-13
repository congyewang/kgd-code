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

class NormalizingFlow:
    def __init__(self, q0, L, k, model):
        self.q0 = q0
        self.log_q0 = jit(lambda x: jnp.log(self.q0(x)))
        self.S_q0 = jit(vmap(grad(self.log_q0)))
        self.L = L
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k_pq = GradientKernel(self.S_PQ, k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)

        self.model = model

    def F(self,theta,key,n_sim,d):
        W = random.normal(key, shape=(n_sim, d))
        X = self.model(theta,W)
        return self.KGD.evaluate(X)

    def run(self,theta0,eta,T,d,n_sim=100):
        Theta = jnp.zeros((T, len(theta0)))
        theta = theta0.copy()
        Theta = Theta.at[0].set(theta)
        for it in range(1,T):
            key = random.PRNGKey(it)
            DF = jit(grad(lambda theta : self.F(theta, key, n_sim, d)))
            theta = theta - eta * DF(theta)
            print(theta[0])
            Theta = Theta.at[it].set(theta)
        return Theta





    

