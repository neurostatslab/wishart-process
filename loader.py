# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax.numpy as jnp
import numpyro

from labrepo.datasets.allen_brain_observatory import AllenBrainData

import models
import inference
import numpy as np
import utils

from scipy.io import loadmat

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
        if params['gratings'] == 'static':
            counts, conditions, unit_locs = dataloader.load_mouse_vis_static_gratings(
                snr_threshold=params['snr_threshold'],
                bin_edges=params['bin_edges']
            )

        if params['gratings'] == 'drifting':
            counts, conditions, unit_locs = dataloader.load_mouse_vis_drifting_gratings(
                snr_threshold=params['snr_threshold'],
                bin_edges=params['bin_edges']
            )

            if 'representation' in params.keys() and params['representation'] == 'direction':
                conditions = [{
                    'orientation':np.mod(c['angle'],180),
                    'temporal_freq':c['temporal_freq'],
                    'direction':c['angle'] < 180
                    } for c in conditions]
        

        n_trials = min([c.shape[1] for c in counts])
        
        y = jnp.array([counts[i][:,:n_trials] for i in range(len(counts)) 
            if eval(params['selector'],{'conditions':conditions, 'i':i})
        ]).transpose(2,0,1)

        y = jnp.sqrt(y)
        # y = y - y.mean(0).mean(0)[None,None]

        x = jnp.array([list(conditions[i].values()) for i in range(len(counts)) 
            if eval(params['selector'],{'conditions':conditions, 'i':i})
        ])

        self.x,self.y,self.mu,self.sigma,self.x_test,self.y_test,self.mu_test,self.sigma_test,self.F,self.F_test,_,_,_,_ = utils.split_data(
            x,y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=None,sigma=None,F=None
        )

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x_test, self.y_test
    
# %% 
class MonkeyReachLoader:
    def __init__(self,params):
        # TODO: Fix the error, this returns outliers (maybe for return trials?)
        '''
            params['session']: session file address
            params['window']: time in milliseconds for calculating spike counts
            params['train_prop']: training data proportion, in [0,1]
        '''

        data = loadmat(params['session'],simplify_cells=True)['r'] # For original files, use 'R'
        targetOn = np.array([row['timeTargetOn'] for row in data])
        targetPos = np.array([row['startTrialParams']['posTarget'] for row in data])
        spikes = [row['spikeRaster2'] for row in data]

        uncued_time = 20

        valid = np.where((
            ~np.isnan(targetOn) & 
            (np.around(targetOn,0) != uncued_time)
        ))[0]

        targetOn = targetOn[valid]

        car2polar = lambda z: np.array((
            np.around(np.rad2deg(np.arctan2(z[:,0], z[:,1])),-1),
            np.around(np.sqrt(z[:,0]**2+z[:,1]**2),0)
        )).T
        
        if params['representation'] == 'cartesian': targetPos = targetPos[valid,:2]
        if params['representation'] == 'polar': targetPos = car2polar(targetPos[valid,:2])

        spikes = [spikes[i] for i in valid]

        x, indices, counts = np.unique(
            targetPos,axis=0,
            return_inverse=True,
            return_counts=True
        )
        
        n_trials = min(counts)

        y = np.array([
            [spikes[j].toarray()[
                :,int(targetOn[j]-params['window']):int(targetOn[j])].sum(1) 
            for j in np.where(indices==i)[0][:n_trials].tolist()] 
            for i in range(x.shape[0])]
        ).transpose(1,0,2)

        # diff = (np.diff(y.var(0),axis=0)**2).sum(1)
        # y = y[:,:,np.argsort(diff)[:10]]
        if 'sqrt_transform' in params.keys():
            y = np.sqrt(y)
        # y = y - y.mean(0).mean(0)[None,None]


        if params['representation'] == 'cartesian':  conditions = [{'x':x[i,0],'y':x[i,1]} for i in range(len(x))]
        if params['representation'] == 'polar': conditions = [{'theta':x[i,0],'r':x[i,1]} for i in range(len(x))]

        selected =[
            i for i in range(len(counts)) 
            if eval(params['selector'],{'conditions':conditions, 'i':i})
        ]
        
        x = x[selected]
        y = y[:,selected,:]

        self.x,self.y,self.mu,self.sigma,self.x_test,self.y_test,self.mu_test,self.sigma_test,self.F,self.F_test,_,_,_,_ = utils.split_data(
            x,y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=None,sigma=None,F=None
        )
        
    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x_test, self.y_test

# %%
class NeuralTuningProcessLoader:
    def __init__(self,params):
        x = jnp.linspace(0,360,params['M'],endpoint=False)[:,None]
        # %% Prior
        wp_kernel = utils.get_kernel(params['wp_kernel'],params['wp_kernel_diag'])

        V = utils.get_scale_matrix(params)

        nt = models.NeuralTuningProcess(num_dims=params['D'],spread=params['spread'],amp=params['amp'])
        wp = eval('models.'+params['prior'])(
            kernel=wp_kernel,nu=params['nu'],V=V,
            diag_scale=params['wp_sample_diag']
        )
        # wp = models.WishartProcess(kernel=wp_kernel,nu=params['nu'],V=V)

        # %% Likelihood
        likelihood = eval('models.'+params['likelihood']+'()')

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu = nt.sample(jnp.hstack((x)))
            sigma = wp.sample(jnp.hstack((x)))
            y = jnp.stack([likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(params['N'])])

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
        x = jnp.linspace(0,360,params['M'],endpoint=False)
        # %% Prior
        gp_kernel = utils.get_kernel(params['gp_kernel'],params['gp_kernel_diag'])
        wp_kernel = utils.get_kernel(params['wp_kernel'],params['wp_kernel_diag'])

        V = utils.get_scale_matrix(params)
        self.V = V

        diag_scale = params['wp_sample_diag'] if 'wp_sample_diag' in params else 1e-1

        gp = models.GaussianProcess(kernel=gp_kernel,num_dims=params['D'])
        wp = models.WishartProcess(kernel=wp_kernel,nu=params['nu'],V=V,diag_scale=diag_scale)

        # %% Likelihood
        likelihood = eval('models.'+params['likelihood'])(params['D'])

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu_g = gp.sample(x)
            sigma_g = wp.sample(x)
            y = jnp.stack([likelihood.sample(mu_g,sigma_g,ind=jnp.arange(len(mu_g))) for i in range(params['N'])])


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
        x = jnp.linspace(0,360,params['M'],endpoint=False)
        # %% Prior
        gp_kernel = utils.get_kernel(params['gp_kernel'],params['gp_kernel_diag'])
        wp_kernel = utils.get_kernel(params['wp_kernel'],params['wp_kernel_diag'])

        V = utils.get_scale_matrix(params)
        self.V = V

        diag_scale = params['wp_sample_diag'] if 'wp_sample_diag' in params else 1e-1

        gp = models.GaussianProcess(kernel=gp_kernel,num_dims=params['D'])
        wp = models.WishartProcess(kernel=wp_kernel,nu=params['nu'],V=V,diag_scale=diag_scale)

        # %% Likelihood
        likelihood = eval('models.'+params['likelihood'])(params['D'])

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu = gp.sample(x)
            sigma = wp.sample(x)
            y = jnp.stack([likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(params['N'])])
        

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
