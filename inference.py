# -*- coding: utf-8 -*-
"""
@author: Amin
"""

# %% Inference
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal

import jax.numpy as jnp
import numpyro.distributions as dist

# %%
class Variational:
    def infer(self,optim,x,y,n_iter=10000,key=jax.random.PRNGKey(0)):
        svi = SVI(
            self.model, self.guide, optim, Trace_ELBO()
        )
        svi_result = svi.run(key, n_iter, x, y)

        self.losses = svi_result.losses
        self.posterior = svi_result.params


class VariationalDetla(Variational):
    def __init__(self,model):
        self.guide = AutoDelta(model)
        self.model = model
        
    def sample(self,key=None):
        L = self.posterior['L']
        F, G = self.posterior['F_auto_loc'], self.posterior['G_auto_loc']
        fft = jnp.einsum('abn,cbn->acn',F,F)
        afft = jnp.einsum('ab,bcn->acn',L,fft)
        sigma = jnp.einsum('abn,bc->acn',afft,L.T)
        return G,sigma
    
class VariationalNormal(Variational):
    def __init__(self,model):
        self.guide = AutoNormal(model)
        self.model = model

    def sample(self,key=jax.random.PRNGKey(0)):
        F_mean, G_mean = self.posterior['F_auto_loc'], self.posterior['G_auto_loc']
        F_scale, G_scale = self.posterior['F_auto_scale'], self.posterior['G_auto_scale']

        L = self.posterior['L']
        F = dist.Normal(F_mean,F_scale).sample(key)
        G = dist.Normal(G_mean,G_scale).sample(key)
        
        fft = jnp.einsum('abn,cbn->acn',F,F)
        afft = jnp.einsum('ab,bcn->acn',L,fft)
        sigma = jnp.einsum('abn,bc->acn',afft,L.T)

        return G,sigma