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
        x_prop = x.copy()
        for i,layer in enumerate(self.layers[1:]):
            if i == 0:
                x_prop = self.layers[0](x_prop)
            if self.activation == "relu":
                x_prop = nnx.relu(x_prop)
            elif self.activation == "tanh":
                x_prop = jnp.tanh(x_prop)
            elif self.activation == "sigmoid":
                x_prop = nnx.sigmoid(x_prop)
            elif self.activation == "elu":
                x_prop = nnx.elu(x_prop)
            
            x_prop = layer(x_prop)
            # print(x.shape)

        return  self.out(x_prop) + 0. * x

class Model(nn.Module):
    hidden_dim: int
    d : int

    @nn.compact
    def __call__(self, x):
        x_ = x.copy()
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.elu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.elu(x)
        # z = nn.Dense(self.hidden_dim)(x)
        # x = nn.elu(x)
        x = nn.Dense(self.d)(x)
        return (x + x_).squeeze()



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
        self.model_T = MLP(self.d,layers,self.d,activation_function, rngs=nnx.Rngs(0)) 
        #self.model_T = Model(20,self.d) #MLP(self.d, layers, self.d, activation_function, rngs=nnx.Rngs(0))
        self.learning_rate = learning_rate
        #self.model_T = jax.tree.map(lambda p: jnp.zeros_like(p), self.model_T)

        # Multiplier tous les paramètres par 10
        self.optimizer = nnx.Optimizer(self.model_T, optax.adam(self.learning_rate), wrt=nnx.Param)


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
    
    def run(self,T,key,n_sim = 50):
        Keys = random.split(key, T)
        positions = []
        params = nnx.state(self.model_T, nnx.Param)
        flat_param0 = self.flatten_params(params)
        dim_theta = flat_param0.shape[0]
        Theta = jnp.zeros((T,dim_theta ))
        Loss = jnp.zeros((T,))
        for it in range(T):
            if it % 1000 == 0:
                print(it)
            key = Keys[0]
            W = random.uniform(key, shape=(n_sim, self.d),minval = -6.0,maxval=6.0) #2.5*random.normal(key, shape=(n_sim, self.d))
            theta,loss = self.train_step(self.model_T, self.optimizer, W)
            Theta = Theta.at[it].set(theta)
            Loss = Loss.at[it].set(loss)
            positions.append(vmap(lambda w : self.model_T(w))(W))
        return Theta, Loss, positions
            




