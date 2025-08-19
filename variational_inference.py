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
            x = layer(x)
            if self.activation == "relu":
                x = nnx.relu(x)
            elif self.activation == "tanh":
                x = jnp.tanh(x)
            elif self.activation == "sigmoid":
                x = nnx.sigmoid(x)

        return self.out(x)

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
        self.model_T = MLP(self.d,layers,self.d,activation_function, rngs=nnx.Rngs(0)) 
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
            print(it)
            key = Keys[it]
            W = random.normal(key, shape=(n_sim, self.d))
            theta,loss = self.train_step(self.model_T, self.optimizer, W)
            Theta = Theta.at[it].set(theta)
            Loss = Loss.at[it].set(loss)
        return Theta, Loss
            

# key = random.PRNGKey(0)
        # params = self.model_T.init(key, jnp.zeros((self.d,)))  
        # self.unravel_fn = ravel_pytree(params)[1]

    
    # def flatten_params_to_array(self,model):
    #     params = nnx.state(model, nnx.Param)
        
    #     flat_params = traverse_util.flatten_dict(params)
        
    #     flat_list = []
    #     for _, v in flat_params.items():
    #         flat_list.extend(jnp.ravel(v).tolist())  
        
    #     return jnp.array(flat_list)
    
    # def push_T_theta(self,theta,W):
    #      param = self.unravel_fn(theta)
    #      return vmap(lambda w : self.model_T.apply(param, w))(W)
        
    # def F(self,theta,key,n_sim,d):
    #         W = random.normal(key, shape=(n_sim, d))
    #         X = self.push_T_theta(theta,W)
    #         return self.KGD.evaluate(X)

    # def run(self,key,eta,T,n_sim=100):
    #         Keys = random.split(key, T+1)
    #         param0 = self.model_T.init(Keys[-1], jnp.zeros((self.d,)))
    #         theta,_ = ravel_pytree(param0)
    #         Theta = jnp.zeros((T, len(theta)))
    #         Theta = Theta.at[0].set(theta)
    #         for it in range(1,T):
    #             key = Keys[0]#it]
    #             DF = grad(lambda theta : self.F(theta, key, n_sim, self.d))
    #             noise = 0 #1e-7 * random.normal(key+10000, shape=theta.shape)
    #             theta = theta - eta * DF(theta + noise)
    #             Theta = Theta.at[it].set(theta)
    #         return Theta





    

# model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
# optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


# @nnx.jit  # automatic state management for JAX transforms
# def train_step(model, optimizer, x, y):
#     def loss_fn(model):
#         y_pred = model(x)  # call methods directly
#         return ((y_pred - y) ** 2).mean()

#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     optimizer.update(model, grads)  # in-place updates

#     return loss

# @nnx.jit  # automatic state management for JAX transforms
# def train_step(model_T, optimizer, W):
#     def loss_fn(model_T):
#         X = vmap(lambda w : model_T(w))(W)
#         return self.KGD.evaluate(X)

#     loss, grads = nnx.value_and_grad(loss_fn)(model_T)
#     optimizer.update(model_T, grads)  # in-place updates

#     return loss