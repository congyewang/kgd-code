import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import jacfwd, jacrev





#######GAUSSIAN KERNEL#######
def k_gauss(x, y, sigma):  # Gram matrix for Gaussian kernel
    X_norm = jnp.sum(x**2, axis=1)
    Y_norm = jnp.sum(y**2, axis=1)
    xy = jnp.dot(x, y.T)
    dist = X_norm[:, None] + Y_norm[None, :] - 2 * xy
    return jnp.exp(-dist / (2 * sigma**2))

def dk_gauss(x, y, sigma):  # Gram matrix of nabla_1 k(x,y), dim = (n,m,d)
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    return diff / (sigma**2) * k_gauss(x, y, sigma)[:, :, None]

def ddk_gauss(x, y, sigma):  # Gram matrix of nabla_2 . nabla_1 k(x,y), dim = (n,m)
    d = len(x[0])
    X_norm = jnp.sum(x**2, axis=1)
    Y_norm = jnp.sum(y**2, axis=1)
    xy = jnp.dot(x, y.T)
    dist = X_norm[:, None] + Y_norm[None, :] - 2 * xy
    K = jnp.exp(-dist / (2 * sigma**2))
    return 1 / sigma**2 * K * (d - 1 / sigma**2 * dist)


######## INVERSE MULTIQUADRIC KERNEL ########


def k_IMQ(x,y,c,b):
    X_norm = jnp.sum(x**2, axis=1)
    Y_norm = jnp.sum(y**2, axis=1)
    xy = jnp.dot(x, y.T)
    dist = X_norm[:, None] + Y_norm[None, :] - 2 * xy
    return 1/jnp.sqrt(c + dist)**(-b)

# Gram matrix of nabla_1 k(x,y), dim = (n,m,d)
def dk_IMQ_dx(x, y, c, b):
    # x: (n, d), y: (n, d)
    # output: (n, n, d)

    x_exp = x[:, None, :]       # (n, 1, d)
    y_exp = y[None, :, :]       # (1, n, d)

    diff = x_exp - y_exp        # (n, n, d)
    dist_sq = jnp.sum(diff**2, axis=2)  # (n, n)

    denom = (c + dist_sq) ** (-b / 2 - 1)  # (n, n)
    grad = -b * diff * denom[:, :, None]  # (n, n, d)

    return grad





######## LAPLACE KERNEL ########
def k_laplace(x, y, sigma):  # Gram matrix for Laplace kernel
    X_norm = jnp.sum(x**2, axis=1)
    Y_norm = jnp.sum(y**2, axis=1)
    xy = jnp.dot(x, y.T)
    dist = X_norm[:, None] + Y_norm[None, :] - 2 * xy
    return jnp.exp(-jnp.sqrt(dist) / sigma)

def dk_laplace(x, y, sigma):  # Gram matrix of nabla_1 k(x,y), dim = (n,m,d)
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    dist = jnp.linalg.norm(diff, axis=2)  # Compute the Euclidean distance
    grad = jnp.zeros_like(diff)  # Initialize gradient to zero
    non_zero_mask = dist > 0  # Mask for non-zero distances
    expanded_mask = non_zero_mask[:, :, None]
    grad = grad.at[expanded_mask].set(
        -diff[expanded_mask] / (sigma * dist[non_zero_mask][:, :, None]) * jnp.exp(-dist[non_zero_mask] / sigma)[:, :, None]
    )
    return grad

def ddk_laplace(x, y, sigma):  # Gram matrix of nabla_2 . nabla_1 k(x,y), dim = (n,m)
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    dist = jnp.linalg.norm(diff, axis=2)  # Compute the Euclidean distance
    K = jnp.exp(-dist / sigma)  # Laplace kernel values
    d = x.shape[1]  # Dimensionality of the input data
    second_derivative = jnp.zeros((x.shape[0], y.shape[0]))  # Initialize second derivative matrix to zero
    non_zero_mask = dist > 0  # Mask for non-zero distances
    second_derivative = second_derivative.at[non_zero_mask].set(
        K[non_zero_mask] * ((d - 1) / (sigma**2 * dist[non_zero_mask]) - 1 / sigma**3)
    )
    return second_derivative




###### NORME KERNEL ######

def k_norme(x, y):  
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat  # Transpose to match dimensions
    dist = jnp.linalg.norm(diff, axis=2)
    return dist


def dk_norme(x, y):  # Gram matrix of nabla_1 k(x,y), dim = (n,m,d)
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat  # Transpose to match dimensions
    dist = jnp.linalg.norm(diff, axis=2)  + 1e-7 # Adding a small constant to avoid division by zero
    grad = -diff / dist[:, :, None]  # Gradient calculation
    return grad

def ddk_norme(x, y):  # Gram matrix of nabla_2 . nabla_1 k(x,y), dim = (n,m)
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    dist = jnp.linalg.norm(diff, axis=2) + 1e-7  # Adding a small constant to avoid division by zero
    K = jnp.sqrt(dist)  # Norme kernel values
    d = x.shape[1]  # Dimensionality of the input data

    second_derivative = jnp.zeros((x.shape[0], y.shape[0]))  # Initialize second derivative matrix to zero
    non_zero_mask = dist > 0  # Mask for non-zero distances

    second_derivative = second_derivative.at[non_zero_mask].set(
        K[non_zero_mask] * ((d - 1) / (dist[non_zero_mask]**(3/2)) - 1 / (dist[non_zero_mask]**(5/2)))
    )

    return second_derivative