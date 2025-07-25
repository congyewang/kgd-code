

#######################


# class GradientKernel:
#     def __init__(self, s_PQ, k):
#         self.s_PQ = s_PQ
#         self.k = k
#         self.dkx = jit(jacrev(k, argnums=0))
#         self.dky = jit(jacrev(k, argnums=1))
#         self.d2k = jit(jacfwd(self.dky, argnums=0))

#     def evaluate(self,X, x, y):
#         dkx = self.dkx(x, y)
#         dky = self.dky(x, y)
#         d2k = jnp.trace(self.d2k(x, y))
#         k = self.k(x, y)
#         s_PQx = self.s_PQ(X,x)
#         s_PQy = self.s_PQ(X,y)
#         k0 = d2k + \
#             jnp.dot(dkx, s_PQy) + \
#             jnp.dot(dky, s_PQx) + \
#             k * jnp.dot(s_PQx, s_PQy)
#         return k0

    


# class KernelGradientDiscrepancy:
#     def __init__(self, k0):
#         self.k0 = k0
#         self.vfk0 = jit(vmap(k0.evaluate, in_axes=(None, 0, 0)))


#     def k0mat(self, x):
#         n = len(x)
#         ir, ic = np.tril_indices(n, k=-1)
#         k0_tril = self.vfk0(x,x[ir], x[ic])
#         k0_diag = self.vfk0(x,x, x)
#         return (k0_tril, k0_diag, ir, ic)

#     def evaluate(self, x):
#         k0_tril, k0_diag, _, _ = self.k0mat(x)
#         k0_sum = np.sum(k0_tril) * 2 + np.sum(k0_diag)
#         return np.sqrt(k0_sum) / x.shape[0]

#     def cumeval(self,x):
#         n = x.shape[0]
#         kgd = np.empty(n)
#         k0_tril, k0_diag, ir, _ = self.k0mat(x)
#         k0_sum = 0.
#         for i in range(n):
#             k0_sum += np.sum(k0_tril[ir == i]) * 2 + k0_diag[i]
#             kgd[i] = np.sqrt(k0_sum) / (i + 1)
#         return kgd



########################################################
########################################################

# def s_q0(x): 
#     return - 8 * x
# def _grad_v(x):
#     return -4 * x

# def grad_L(X,x):
#     return _grad_v(x) 