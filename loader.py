# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax.numpy as jnp
import numpyro

import models
import inference
import utils

# %%
class NeuralTuningProcessLoader:
    def __init__(self,params):
        x = jnp.linspace(0,360,params['C'],endpoint=False)[:,None]
        # Prior
        wp_kernel = utils.get_kernel(params['wp_kernel'],params['wp_kernel_diag'])

        V = utils.get_scale_matrix(params)

        nt = models.NeuralTuningProcess(N=params['N'],spread=params['spread'],amp=params['amp'])
        wp = eval('models.'+params['prior'])(
            kernel=wp_kernel,nu=params['P'],V=V,
            diag_scale=params['wp_sample_diag']
        )

        # Likelihood
        likelihood = eval('models.'+params['likelihood']+'()')

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu = nt.sample(jnp.hstack((x)))
            sigma = wp.sample(jnp.hstack((x)))
            y = jnp.stack([likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(params['K'])])

        self.x,self.y,self.mu,self.sigma,self.x_test,self.y_test,self.mu_test,self.sigma_test,self.F,self.F_test,_,_,_,_ = utils.split_data(
            x,y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=mu,sigma=sigma,F=wp.F
        )

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x_test, self.y_test

# %%
class PoissonGPWPLoader():
    def __init__(self,params):
        x = jnp.linspace(0,360,params['C'],endpoint=False)
        # Prior
        gp_kernel = utils.get_kernel(params['gp_kernel'],params['gp_kernel_diag'])
        wp_kernel = utils.get_kernel(params['wp_kernel'],params['wp_kernel_diag'])

        V = utils.get_scale_matrix(params)
        self.V = V

        diag_scale = params['wp_sample_diag'] if 'wp_sample_diag' in params else 1e-1

        gp = models.GaussianProcess(kernel=gp_kernel,N=params['N'])
        wp = models.WishartProcess(kernel=wp_kernel,P=params['P'],V=V,diag_scale=diag_scale)

        # Likelihood
        likelihood = eval('models.'+params['likelihood'])(params['N'])

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu_g = gp.sample(x)
            sigma_g = wp.sample(x)
            y = jnp.stack([likelihood.sample(mu_g,sigma_g,ind=jnp.arange(len(mu_g))) for i in range(params['K'])])


        joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 
        true_post = inference.VariationalDelta(joint.model)
        true_post.posterior = {'G_auto_loc': mu_g.T[:,None], 'F_auto_loc':wp.F}
        true_posterior = models.NormalGaussianWishartPosterior(joint,true_post,x)
        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu = true_posterior.mean_stat(lambda x: x, x)
            sigma = true_posterior.mean_stat(lambda x: jnp.einsum('cd,ck->cdk',x-mu,x-mu), x)


        self.x,self.y,self.mu,self.sigma,\
        self.x_test,self.y_test,self.mu_test,self.sigma_test,\
        self.F,self.F_test,self.mu_g,self.mu_g_test,self.sigma_g,self.sigma_g_test = utils.split_data(
            x[:,None],y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=mu,sigma=sigma,F=wp.F,mu_g=mu_g,sigma_g=sigma_g
        )
        self.x = self.x.squeeze()
        self.x_test = self.x_test.squeeze()

        self.likelihood = likelihood

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x_test, self.y_test
    
# %%
class GPWPLoader():
    def __init__(self,params):
        x = jnp.linspace(0,360,params['C'],endpoint=False)

        # Prior
        gp_kernel = utils.get_kernel(params['gp_kernel'],params['gp_kernel_diag'])
        wp_kernel = utils.get_kernel(params['wp_kernel'],params['wp_kernel_diag'])

        V = utils.get_scale_matrix(params)
        self.V = V

        diag_scale = params['wp_sample_diag'] if 'wp_sample_diag' in params else 1e-1

        gp = models.GaussianProcess(kernel=gp_kernel,N=params['N'])
        wp = models.WishartProcess(kernel=wp_kernel,P=params['P'],V=V,diag_scale=diag_scale)

        # Likelihood
        likelihood = eval('models.'+params['likelihood'])(params['N'])

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu = gp.sample(x)
            sigma = wp.sample(x)
            y = jnp.stack([likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(params['K'])])
        

        self.x,self.y,self.mu,self.sigma,self.x_test,self.y_test,self.mu_test,self.sigma_test,self.F,self.F_test,_,_,_,_ = utils.split_data(
            x[:,None],y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=mu,sigma=sigma,F=wp.F
        )
        self.x = self.x.squeeze()
        self.x_test = self.x_test.squeeze()

        self.likelihood = likelihood

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x_test, self.y_test
