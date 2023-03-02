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
    def __init__(self, kernel, nu, V):
        self.kernel = kernel
        self.nu = nu
        self.num_dims = V.shape[0]
        self.L = jnp.linalg.cholesky(V/self.num_dims)

    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(ys)

    def sample(self, x):
        N = x.shape[0]

        L = numpyro.param('L', self.L)

        c_f = self.evaluate_kernel(x,x)

        F = numpyro.sample(
            'F',dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_f),
            sample_shape=(self.num_dims,self.nu)
        )

        fft = jnp.einsum('abn,cbn->acn',F,F)
        afft = jnp.einsum('ab,bcn->acn',L,fft) 
        sigma = jnp.einsum('abn,bc->nac',afft,L.T) 

        return sigma

    
    def posterior(self, X, Y, x):
        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        f = jnp.einsum('ij,jmn->mni',(K_X_x.T@Ki),Y)
        K = K_x_x - K_X_x.T@Ki@K_X_x

        F = numpyro.sample(
            'F_test',dist.MultivariateNormal(f,covariance_matrix=jnp.eye(len(K))),
            sample_shape=(1,1)
        ).squeeze()
        print(F.shape)

        return jnp.einsum(
            "ai,ijk,ljk,bl->kab", self.L, F, F, self.L
        )


# %%
class GaussianProcess:
    def __init__(self, kernel, num_dims):
        self.kernel = kernel
        self.num_dims = num_dims


    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(ys)


    def sample(self, x):
        N = x.shape[0]
        c_g = self.evaluate_kernel(x,x)
        G = numpyro.sample(
            'G',dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_g),
            sample_shape=(self.num_dims,1)
        ).squeeze().T
        return G
    
    def posterior(self, X, Y, x):
        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)
        
        Ki   = jnp.linalg.inv(K_X_X)
        f = jnp.einsum('ij,jm->mi',(K_X_x.T@Ki),Y)
        
        K = K_x_x - K_X_x.T@Ki@K_X_x

        G_new = numpyro.sample(
            'G_test',dist.MultivariateNormal(f,covariance_matrix=jnp.eye(len(K))),
            sample_shape=(1,1)
        ).squeeze().T
        
        return G_new
    
# %%
class NormalConditionalLikelihood:
    def sample(self,mu,sigma,ind=None,y=None):
        Y = numpyro.sample(
            'y',dist.MultivariateNormal(mu[ind,...],sigma[ind,...]),
            obs=y
        )
        return Y
    
# %%
class PoissonConditionalLikelihood:
    def __init__(self,rate):
        self.rate = rate

    def sample(self,mu,sigma,ind=None,y=None):
        rate = numpyro.param('rate', self.rate)

        G = numpyro.sample('g',dist.MultivariateNormal(mu[ind,...],sigma[ind,...]))
        
        Y = numpyro.sample('y',dist.Poisson(jnp.exp(G[ind,...])*rate[None]).to_event(1),obs=y)
        
        return Y

# %%
class JointGaussianWishartProcess:
    def __init__(self, gp, wp, likelihood):
        self.gp = gp
        self.wp = wp
        self.likelihood = likelihood

    def model(self, x, y):
        B,N,D = y.shape

        sigma = self.wp.sample(x)
        G = self.gp.sample(x)

        with numpyro.plate('obs', N) as ind:
            self.likelihood.sample(G,sigma,ind,y=y)
        return

# %%
class NormalGaussianWishartPosterior:
    def __init__(self, joint, posterior, x):
        self.joint = joint
        self.posterior = posterior
        self.x = x

    def sample(self, x):
        F,G,_ = self.posterior.sample()
        mu = self.joint.gp.posterior(self.x, G, x)
        sigma = self.joint.wp.posterior(self.x, F, x) 

        return mu, sigma