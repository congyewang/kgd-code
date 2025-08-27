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
from flax import traverse_util
from flax import linen as nn
from flax import nnx
import optax


import flax.nnx as nnx

class MLP(nnx.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="tanh", *, rngs: nnx.Rngs):
        self.layers = []
        in_dim = input_dim

        for h in hidden_dims:
            self.layers.append(nnx.Linear(in_dim, h, rngs=rngs))
            in_dim = h

        self.out = nnx.Linear(in_dim, output_dim, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers:
            if self.activation == "relu":
                x = nnx.relu(x)
            elif self.activation == "tanh":
                x = jnp.tanh(x)
            elif self.activation == "sigmoid":
                x = nnx.sigmoid(x)
            elif self.activation == "elu":
                x = nnx.elu(x)
            
            x = layer(x)


        return self.out(x)



class VariationalInference:
    def __init__(self, q0, L, k,d,layers,activation_function,learning_rate = 1e-3):
        self.d = d
        self.q0 = q0
        self.log_q0 = jit(lambda x: jnp.log(self.q0(x)))
        self.S_q0 = jit(vmap(grad(self.log_q0)))
        self.L = L
        self.gradL = jit(grad(lambda X: jax.lax.stop_gradient(len(X)) * self.L(X)))
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k_pq = GradientKernel(self.S_PQ, k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)
        #self.model_T = MLP(self.d,layers,self.d,activation_function, rngs=nnx.Rngs(0)) 
        self.model_T = MLP(self.d, layers, self.d, activation_function, rngs=nnx.Rngs(0))

        # # Multiplier tous les paramètres par 10
        # params = nnx.state(self.model_T, nnx.Param)
        # scaled_params = jax.tree_map(lambda p: p * 5, params)
        # nnx.update(self.model_T, scaled_params)
        self.optimizer = nnx.Optimizer(self.model_T, optax.adam(1e-3), wrt=nnx.Param)


    def flatten_params(self,params):
        leaves, _ = jax.tree_util.tree_flatten(params)
        flat = jnp.concatenate([jnp.ravel(p) for p in leaves])
        return flat

    #@nnx.jit 
    def train_step(self,model_T, optimizer, W):
        def loss_fn(model_T):
            X = vmap(lambda w : model_T(w))(W)
            return self.KGD.evaluate(X) #+ 1/1000 * self.L(X)

        loss, grads = nnx.value_and_grad(loss_fn)(model_T)
        self.optimizer.update(grads)  # in-place updates
        params = nnx.state(model_T, nnx.Param)
        return  self.flatten_params(params),loss #
    
    def run(self,T,key,n_sim = 100):
        Keys = random.split(key, T)
        params = nnx.state(self.model_T, nnx.Param)
        flat_param0 = self.flatten_params(params)
        dim_theta = flat_param0.shape[0]
        Theta = jnp.zeros((T,dim_theta ))
        Loss = jnp.zeros((T,))
        for it in range(T):
            if it % 100 == 0:
                print(it)
            key = Keys[it]
            W = random.normal(key, shape=(n_sim, self.d))
            theta,loss = self.train_step(self.model_T, self.optimizer, W)
            Theta = Theta.at[it].set(theta)
            Loss = Loss.at[it].set(loss)
        return Theta, Loss
            




