import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import jit, vmap, grad
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
import modules.kgd_functions as f
from modules.kgd_functions import F_P, GradientKernel,KernelGradientDiscrepancy
from jax import random
import os
from jax import grad, random,vmap,tree_util
from jax.flatten_util import ravel_pytree
from flax import traverse_util

import optax

import flax
from flax import linen as nn

class KGD_Descent:
    def __init__(self, grad_log_q0, L, k):
        self.S_q0 = grad_log_q0
        self.L = L
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k = k
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)   

    def run_particles(self,X0,T, learning_rate=1e-2,gamma = 300):
        tx = optax.adam(learning_rate)
        opt_state = tx.init(X0)
        X = X0.copy()

        loss_and_grad = jax.jit(jax.value_and_grad(self.KGD.evaluate))
        all_particles = [X]

        losses = []
        for step in range(T):
            loss_val, g = loss_and_grad(X)
            updates, opt_state = tx.update(g, opt_state)
            X = optax.apply_updates(X, updates)
            all_particles.append(X)

            if step % 10 == 0:
                mse = self.L(X) / gamma   
                print(f"Step {step}, loss = {loss_val}, mse = {mse}")
                losses.append(loss_val)     
        return all_particles    