# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import numpy as onp
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from jax import vmap

# %%
class WishartProcess:
    """PROTOTYPE FOR WISHART PROCESS INFERENCE
    - Wilson & Ghahramani (2010). Generalised Wishart Processes.
    https://arxiv.org/abs/1101.0240
    """

    def __init__(self, kernel, nu, V):
        """
        Parameters
        ----------
        kernel : function
            Positive definite kernel function.
        nu : int
            Degrees of freedom parameter.
        V : positive definite matrix
            Scale parameter.
        """
        self.kernel = kernel
        self.nu = nu
        self.L = jnp.linalg.cholesky(V)
        self.num_dims = self.L.shape[0]

    def evaluate_kernel(self, xs):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(xs)

    def sample(self, key, xs):
        """
        Draws a sample from the prior.
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key used as the random key for jax.
        xs : array
            Array of input inputs where `xs.shape[0]`
            corresponds to the number of observations
            and `xs.shape[1:]` corresponds to the
            dimensions accepted by the kernel function.
        """
        num_obs = xs.shape[0]
        
        # TODO: make this more efficient
        K = self.evaluate_kernel(xs)


        # Sample Gaussian process
        U = jax.random.multivariate_normal(
            key, jnp.zeros(num_obs), K,
            shape=(self.num_dims, self.nu)
        )

        # Compute covariance
        return jnp.einsum(
            "ai,ijk,ljk,bl->abk", self.L, U, U, self.L
        )

# %%
class GaussianProcess:
    def __init__(self, kernel, num_dims):
        """
        Parameters
        ----------
        kernel : function
            Positive definite kernel function.
        nu : int
            Degrees of freedom parameter.
        V : positive definite matrix
            Scale parameter.
        """
        self.kernel = kernel
        self.num_dims = num_dims


    def evaluate_kernel(self, xs):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(xs)


    def sample(self, key, xs):
        """
        Draws a sample from the prior.
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key used as the random key for jax.
        xs : array
            Array of input inputs where `xs.shape[0]`
            corresponds to the number of observations
            and `xs.shape[1:]` corresponds to the
            dimensions accepted by the kernel function.
        """
        num_obs = xs.shape[0]
        
        # TODO: make this more efficient
        K = self.evaluate_kernel(xs)

        # Sample Gaussian process
        f = jax.random.multivariate_normal(
            key, jnp.zeros(num_obs), K,
            shape=(self.num_dims, 1)
        )

        # Compute covariance
        return f
    
# %%
class ConditionalLikelihood:
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self,keys,num_samples=1):
        Y = jnp.stack([jax.random.multivariate_normal(
            keys[i], self.mu[:,0,i], self.sigma[:,:,i],
            shape=(num_samples,1)
        ) for i in range(self.mu.shape[-1])],0).astype(float)[:,:,0].transpose(1,0,2)
        return Y
    

# %%
class NormalGaussianWishartProcess:
    def __init__(self, gp, wp):
        self.gp = gp
        self.wp = wp

    def model(self, x, y):
        B,N,D = y.shape

        L = numpyro.param('L', self.wp.L) # self.wp.L 

        c_f = self.wp.evaluate_kernel(x)
        c_g = self.gp.evaluate_kernel(x)

        F = numpyro.sample('F',dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_f),sample_shape=(D,self.wp.nu))
        G = numpyro.sample('G',dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_g),sample_shape=(D,1)).squeeze().T

        fft = jnp.einsum('abn,cbn->acn',F,F)
        afft = jnp.einsum('ab,bcn->acn',L,fft) # self.wp.L
        sigma = jnp.einsum('abn,bc->nac',afft,L.T) # self.wp.L.T

        with numpyro.plate('obs', N) as ind:
            numpyro.sample('y',dist.MultivariateNormal(G[ind,...],sigma[ind,...]),obs=y)
        return
    
