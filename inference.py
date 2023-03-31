# -*- coding: utf-8 -*-
"""
@author: Amin
"""

# %% Inference
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta, AutoNormal

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro
# %%
class Variational:
    def infer(self,optim,x,y,n_iter=10000,key=jax.random.PRNGKey(0),num_particles=1):
        svi = SVI(
            self.model, self.guide, optim, Trace_ELBO(num_particles=num_particles)
        )
        svi_result = svi.run(key, n_iter, x, y)

        self.losses = svi_result.losses
        self.posterior = svi_result.params


class VariationalDelta(Variational):
    def __init__(self,model,init=None):
        self.guide = AutoDelta(model) if init is None else AutoDelta(model,init_loc_fn=init_to_value(values=init))
        self.model = model


        
    def sample(self):
        L = self.posterior['L']
        F, mu = self.posterior['F_auto_loc'], self.posterior['G_auto_loc']
        fft = jnp.einsum('abn,cbn->acn',F,F)
        afft = jnp.einsum('ab,bcn->acn',L,fft)
        sigma = jnp.einsum('abn,bc->acn',afft,L.T)
        return F.transpose(2,0,1),mu.squeeze().T,sigma.transpose(2,0,1)
    
class VariationalNormal(Variational):
    def __init__(self,model,init=None):
        self.guide = AutoNormal(model) if init is None else AutoNormal(model,init_loc_fn=init_to_value(values=init))
        self.model = model

    def sample(self):
        F_mean, G_mean = self.posterior['F_auto_loc'], self.posterior['G_auto_loc']
        F_scale, G_scale = self.posterior['F_auto_scale'], self.posterior['G_auto_scale']

        L = self.posterior['L']
        
        F = numpyro.sample('F_post',dist.Normal(F_mean,F_scale))
        mu = numpyro.sample('mu_post',dist.Normal(G_mean,G_scale))
        
        fft = jnp.einsum('abn,cbn->acn',F,F)
        afft = jnp.einsum('ab,bcn->acn',L,fft)
        sigma = jnp.einsum('abn,bc->acn',afft,L.T)

        return F.transpose(2,0,1),mu.squeeze().T,sigma.transpose(2,0,1)