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

