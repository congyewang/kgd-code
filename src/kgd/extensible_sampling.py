import jax.numpy as jnp
import jax
from jax import jit, vmap, grad
from modules.kgd_functions import GradientKernel,KernelGradientDiscrepancy
from jax import random

class ExtensibleSampling:
    def __init__(self, grad_log_q0, L, k, gradL = None):
        self.S_q0 = jit(grad_log_q0)
        self.L = L
        self.k = k
        if gradL == None:
            self.gradL = jit(grad(lambda X : jax.lax.stop_gradient(len(X)) * self.L(X)))
        else:
            self.gradL = gradL
        self.S_PQ = jit(lambda X: self.S_q0(X) - self.gradL(X))
        self.k_pq = GradientKernel(self.S_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_pq)

    def grid_search(self,key,X,l,u,noise_std,delta,n_grid,it):
        n, d = X.shape
        if it <= 30:
            axis = jnp.linspace(l, u, n_grid)
            meshes = jnp.meshgrid(*[axis]*d, indexing="ij")
            grid_points = jnp.stack(meshes, axis=-1).reshape(-1, d) + noise_std * random.normal(key, shape=(n_grid**d, d))  # Adding small noise to avoid exact duplicates
            KGD_grid = jnp.zeros(n_grid**d)
            for i in range(n_grid**d):
                KGD_grid = KGD_grid.at[i].set(self.KGD.evaluate(jnp.concatenate((X, grid_points[i][None,:]), axis=0)))
            min_idx = jnp.argmin(KGD_grid)
        else:
            # From number of particles > 30 we sample from a mixture of gaussian whose means are particles already sampled

            n_samples =  10**d
            key,subkey = random.split(key)

            comp_idx = random.randint(key, (n_samples,), minval=0, maxval=n)
            means = X[comp_idx]  

            
            noise = jnp.sqrt(delta) * random.normal(subkey, (n_samples, d))

            grid_points = means + noise  # (n_samples, d)

            KGD_grid = jnp.zeros(n_samples)
            for i in range(n_samples):
                KGD_grid = KGD_grid.at[i].set(
                    self.KGD.evaluate(jnp.concatenate((X, grid_points[i][None, :]), axis=0))
                )
            min_idx = jnp.argmin(KGD_grid)
        return grid_points[min_idx]

    def prior_search(self,key,X,mean,cov,n_samples,it):
        d = X.shape[1]
        points = cov * random.normal(key, (n_samples, d)) + mean
        KGD_points = jnp.zeros(n_samples)
        for i in range(n_samples):
            KGD_points = KGD_points.at[i].set(self.KGD.evaluate(jnp.concatenate((X, points[i][None,:]), axis=0)))
        min_idx = jnp.argmin(KGD_points)
        return points[min_idx]
    

    def GD_search(self,X,N_it=5000,step_size=0.01):
        Loss = lambda x : self.KGD.evaluate(jnp.concatenate((X, x[None,:]), axis=0))
        
        DLoss = jit(grad(Loss))
        x0 = jnp.mean(X, axis=0)
        x = x0
        for it in range(N_it):
            x = x - step_size * DLoss(x)
        return x


    def run_particles(self,key,x0,T,min_search_method = "grid_search",settings = [-5,5,0.1,0.5,20]) :
        d = len(x0)
        X = jnp.zeros((T, d))
        X = X.at[0].set(x0)

        keys = random.split(key, T) 
        for it in range(1,T):
            if it % 10 == 0:
                print("Iteration:", it)
            if min_search_method == "grid_search":
                X = X.at[it].set(self.grid_search(keys[it],X[:it], settings[0], settings[1], settings[2],settings[3],settings[4], it))
            if min_search_method == "GD search":
                X = X.at[it].set(self.GD_search(X[:it]))
            if min_search_method == "prior_search":
                X = X.at[it].set(self.prior_search(keys[it],X[:it], settings[0], settings[1], settings[2], it))
        return X                        