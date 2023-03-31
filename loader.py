# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax
import jax.numpy as jnp
import numpyro


from labrepo.datasets.allen_brain_observatory import AllenBrainData

import models

def get_kernel(kernel,params):
    if kernel == 'periodic': return lambda x, y: params['scale']*(params['diag']*(x==y)+jnp.exp(-2*jnp.sin(jnp.pi*jnp.abs(x-y)/360.)**2/(params['sigma']**2))),
    if kernel == 'RBF': return lambda x, y: params['scale']*(params['diag']*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*params['sigma']**2)))


# %%
class AllenStaticGratingsLoader:
    def __init__(self,params):
        '''
        params is a dictionary that includes snr_threshold, bin_edges, selector as keys
        selector is a statement that selects a subset of conditions
            selector = 'conditions[i][\'angle\'] == 0 & conditions[i][\'phase\'] == 0'
            selector = 'True'
        '''
        dataloader = AllenBrainData()
        counts, conditions, unit_locs = dataloader.load_mouse_vis_static_gratings(
            snr_threshold=params['snr_threshold'],
            bin_edges=params['bin_edges']
        )

        n_trials = min([c.shape[1] for c in counts])

        self.y = jnp.array([counts[i][:,:n_trials] for i in range(len(counts)) 
            if eval(params['selector'])
        ]).transpose(2,0,1)

        self.x = jnp.array([list(conditions[i].values()) for i in range(len(counts)) 
            if eval(params['selector'])
        ])

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x, self.y
# %%
class NeuralTuningProcessLoader:
    def __init__(self,params):
        x = jnp.linspace(0,360,params['M'],endpoint=False)
        x_test = x[[0,len(x)//2]+1]
        x = jnp.setdiff1d(x,x_test)

        # %% Prior
        kernel_wp = get_kernel(params['kernel_gp'])
        V = params['epsilon']*jnp.eye(params['D'])

        nt = models.NeuralTuningProcess(num_dims=params['D'],spread=params['spread'],amp=params['amp'])
        wp = models.WishartProcess(kernel=kernel_wp,nu=2*params['D'],V=V)

        # %% Likelihood
        likelihood = models.NormalConditionalLikelihood()

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu = nt.sample(jnp.hstack((x,x_test)))
            sigma = wp.sample(jnp.hstack((x,x_test)))
            y = jnp.stack([likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(params['N'])])

        mu_test = mu[len(x):]
        mu = mu[:len(x)]

        sigma_test = sigma[len(x):]
        sigma = sigma[:len(x)]

        self.mu_test = mu_test
        self.mu = mu

        self.sigma_test = sigma_test
        self.sigma = sigma

        y_test = {}
        y_test['x_new'] = y[:,len(x):]
        y = y[:,:len(x)]
        num_train = int(params['N']*params['train_prop'])
        y_test['x'] = y[num_train:]
        y = y[:num_train]

        self.x = x
        self.y = y

        self.x_test = x_test
        self.y_test = y_test

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x_test, self.y_test



