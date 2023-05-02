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
    
    def load(self, file):
        result = jnp.load(file,allow_pickle=True).item()
        self.losses = result['losses']
        self.posterior = result['posterior']

    def infer(self,optim,x,y,n_iter=10000,key=jax.random.PRNGKey(0),num_particles=1):
        svi = SVI(
            self.model, self.guide, optim, Trace_ELBO(num_particles=num_particles)
        )
        svi_result = svi.run(key, n_iter, x, y, stable_update=True)

        self.losses = svi_result.losses
        self.posterior = svi_result.params

    def save(self,file):
        jnp.save(file,{'losses':self.losses, 'posterior':self.posterior})


class VariationalDelta(Variational):
    def __init__(self,model,init=None):
        self.guide = AutoDelta(model) if init is None else AutoDelta(model,init_loc_fn=init_to_value(values=init))
        self.model = model
        
    def sample(self):
        L = self.posterior['L']
        F, G = self.posterior['F_auto_loc'], self.posterior['G_auto_loc']
        return F, G
    
class VariationalNormal(Variational):
    def __init__(self,model,init=None):
        self.guide = AutoNormal(model) if init is None else AutoNormal(
            model,init_loc_fn=init_to_value(values=init),init_scale=1e-2
        )
        self.model = model

    def sample(self):
        F_mean, G_mean = self.posterior['F_auto_loc'], self.posterior['G_auto_loc']
        F_scale, G_scale = self.posterior['F_auto_scale'], self.posterior['G_auto_scale']
        
        F = numpyro.sample('F_post',dist.Normal(F_mean,F_scale))
        G = numpyro.sample('mu_post',dist.Normal(G_mean,G_scale))
        return F, G
