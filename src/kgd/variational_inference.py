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

# Pushforward map
class Model(nn.Module):
    hidden_dim: int
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.hidden_dim)(x)
        y = nn.elu(y)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.elu(y)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.elu(y)
        y = nn.Dense(self.output_dim)(y)
        return  y.squeeze() 




class VariationalInference:
    def __init__(self, grad_log_q0, L, k, d, layers_dim, learning_rate = 1e-3):
        self.d = d
        self.S_q0 = grad_log_q0
        self.L = L
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k = k
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)

        self.learning_rate = learning_rate
        self.model_T = Model(layers_dim, self.d)
        self.tx = optax.adam(learning_rate=learning_rate)
        
    #mu_0
    def noise_fn(self,key, shape):
        return random.uniform(key, shape, minval=-3, maxval=3)
    
    def pretraining(self,key,T_pretrain = 1000):
        def pretraining_loss(key, params, n_samples=100, init_std=3):
            key1, key2 = random.split(key)
            noise = self.noise_fn(key1, (n_samples, self.d))
            X = self.model_T.apply(params, noise)
            target_samples = init_std*random.normal(key2, X.shape)
            assert target_samples.shape == X.shape
            k = lambda x, y: f.imq_scale_mixtures(x, y, jnp.logspace(-2, 2, 5))
            Kxx = vmap(lambda x: vmap(lambda y: k(x, y))(X))(X)
            Kxy = vmap(lambda x: vmap(lambda y: k(x, y))(target_samples))(X)
            Kyy = vmap(lambda x: vmap(lambda y: k(x, y))(target_samples))(target_samples)
            return (Kxx - 2*Kxy + Kyy).mean()
        
        key1, key2, *(keys) = random.split(key, T_pretrain+2)
        x = random.normal(key1, (self.d,)) 
        params = self.model_T.init(key2, x)
        jax.tree_util.tree_map(lambda x: x.shape, params) # Checking output shapes
        self.model_T.apply(params, x)
        opt_state = self.tx.init(params)

        for step in tqdm(range(T_pretrain)):
            loss_grad_fn = jax.value_and_grad(lambda params: pretraining_loss(keys[step], params))
            loss_value, grads = loss_grad_fn(params)
            updates, opt_state = self.tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        return params

    def l2_norm(self,params):
        return sum([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)])
    def loss(self,key, params, num_samples=100, d_noise=4, reg=0.0
        ):
        noise = self.noise_fn(key, (num_samples, d_noise))
        X = self.model_T.apply(params, noise)
        return self.KGD.square_kgd(X) + reg*self.l2_norm(params) 
    
    def run_particles(self,key,T,n,N = 2000,gamma = 300):
        keys = random.split(key, T+2)

        params = self.pretraining(keys[-2])

        losses = []
        all_particles = []

        noise = self.noise_fn(keys[-1], (n, self.d))
        X_kgd = self.model_T.apply(params, noise)
        
        schedule = optax.exponential_decay(
        init_value=self.learning_rate,  
        transition_steps=100,
        decay_rate=0.99
        )
        self.tx = optax.chain(
            optax.adam(learning_rate=self.learning_rate),
        )
        opt_state = self.tx.init(params)
        

        loss_ = jit(lambda key, params: self.loss(key, params, num_samples=n, d_noise=self.d, reg=2.0))

        for step in tqdm(range(T)):
            key1,key2 = random.split(keys[step])

            all_particles.append(X_kgd)

            loss_grad_fn = jax.value_and_grad(lambda params: loss_(key1, params))
            loss_value, grads = loss_grad_fn(params)
            losses.append(loss_value)

            updates, opt_state = self.tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            noise_kgd = self.noise_fn(key2, (n, self.d))
            X_kgd = self.model_T.apply(params, noise_kgd)

            if step % 1000 == 0:
                X = self.model_T.apply(params, noise)
                mse = self.L(X) / gamma
                gradsize = (self.gradL(X)**2).sum() / gamma**2
                print(f"Step {step}, Loss: {loss_value}, MSE {mse}, Grad {gradsize}")
        return all_particles
