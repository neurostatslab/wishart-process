# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax
import jax.numpy as jnp
import numpyro

from functools import reduce, partial

from labrepo.datasets.allen_brain_observatory import AllenBrainData

import models

from scipy.linalg import block_diag
import numpy as np

from scipy.io import loadmat

# %%
def get_kernel(params,diag):
    '''Returns the full kernel of multi-dimensional condition spaces
    '''
    if len(params) > 1: 
        return lambda x,y: diag*(x==y)+reduce(
            lambda a,b: a*b, [
                _get_kernel(params[i]['type'],params[i])(x[i],y[i]) for i in range(len(params))
            ])
    else: 
        return lambda x,y: diag*(x==y)+_get_kernel(params[0]['type'],params[0])(x,y)
        

def _get_kernel(kernel,params):
    '''Private function, returns the kernel corresponding to a single dimension
    '''
    if kernel == 'periodic': 
        return lambda x,y: params['scale']*jnp.exp(-2*jnp.sin(jnp.pi*jnp.abs(x-y)/params['normalizer'])**2/(params['sigma']**2))
    if kernel == 'RBF': 
        return lambda x,y: params['scale']*jnp.exp(-jnp.linalg.norm(x-y)**2/(2*params['sigma']**2))


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
        
        y = jnp.array([counts[i][:,:n_trials] for i in range(len(counts)) 
            if eval(params['selector'],{'conditions':conditions, 'i':i})
        ]).transpose(2,0,1)

        y = jnp.sqrt(y)

        x = jnp.array([list(conditions[i].values()) for i in range(len(counts)) 
            if eval(params['selector'],{'conditions':conditions, 'i':i})
        ])

        self.x,self.y,self.mu,self.sigma,self.x_test,self.y_test,self.mu_test,self.sigma_test = split_data(
            x,y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=None,sigma=None
        )

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return None, self.y_test
    
# %% 
class MonkeyReachLoader:
    def __init__(self,params):
        '''
            params['session']: session file address
            params['window']: time in milliseconds for calculating spike counts
            params['train_prop']: training data proportion, in [0,1]
        '''

        data = loadmat(params['session'],simplify_cells=True)['r'] # For original files, use 'R'
        targetOn = np.array([row['timeTargetOn'] for row in data])
        targetPos = np.array([row['startTrialParams']['posTarget'] for row in data])
        spikes = [row['spikeRaster'] for row in data]

        uncued_time = 20

        valid = np.where((
            ~np.isnan(targetOn) & 
            (targetOn != uncued_time)
        ))[0]

        targetOn = targetOn[valid]
        targetPos = targetPos[valid,:2]
        
        spikes = [spikes[i] for i in valid]

        positions, indices, counts = np.unique(
            targetPos,axis=0,
            return_inverse=True,
            return_counts=True
        )
        
        n_trials = min(counts)

        y = np.array([
            [spikes[j].toarray()[
                :,int(targetOn[j]-params['window']):int(targetOn[j])].sum(1) 
            for j in np.where(indices==i)[0][:n_trials]] 
            for i in range(positions.shape[0])]
        ).transpose(1,0,2)

        y = np.sqrt(y)

        car2polar = lambda z: np.array((
            np.rad2deg(np.arctan2(z[:,0], z[:,1])),
            np.sqrt(z[:,0]**2+z[:,1]**2)
        )).T

        if params['representation'] == 'cartesian': 
            x = positions
            conditions = [{'x':x[i,0],'y':x[i,1]} for i in range(len(x))]
        if params['representation'] == 'polar':
            x = car2polar(positions)
            conditions = [{'theta':x[i,0],'r':x[i,1]} for i in range(len(x))]

        selected =[
            i for i in range(len(counts)) 
            if eval(params['selector'],{'conditions':conditions, 'i':i})
        ]
        
        x = x[selected]
        y = y[:,selected,:]

        self.x,self.y,self.mu,self.sigma,self.x_test,self.y_test,self.mu_test,self.sigma_test = split_data(
            x,y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=None,sigma=None
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
        wp_kernel = get_kernel(params['wp_kernel'],params['wp_kernel_diag'])

        V = get_scale_matrix(params)

        nt = models.NeuralTuningProcess(num_dims=params['D'],spread=params['spread'],amp=params['amp'])
        wp = models.WishartProcess(kernel=wp_kernel,nu=params['nu'],V=V)

        # %% Likelihood
        likelihood = eval('models.'+params['likelihood']+'()')

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu = nt.sample(jnp.hstack((x)))
            sigma = wp.sample(jnp.hstack((x)))
            y = jnp.stack([likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(params['N'])])

        self.x,self.y,self.mu,self.sigma,self.x_test,self.y_test,self.mu_test,self.sigma_test = split_data(
            x,y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=mu,sigma=sigma
        )

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x_test, self.y_test

def get_scale_matrix(params):
    if params['scale_type'] == 'low_rank':
        return params['epsilon']*(CovarianceModel.low_rank(params['D'],params['rank'],seed=params['seed'],g=1e0)+\
                1e-1*params['epsilon']*jnp.eye(params['D']))
    if params['scale_type'] == 'multi_region':
        return params['epsilon']*(CovarianceModel.multi_region(
                params['D'],C=params['C'],seed=params['seed'],g=1e0
            ) + 1e0*jnp.eye(params['D']))
    if params['scale_type'] == 'diag':
        return params['epsilon']*jnp.eye(params['D'])
        
    
# %%
class GPWPLoader():
    def __init__(self,params):
        x = jnp.linspace(0,360,params['M'],endpoint=False)
        # %% Prior
        gp_kernel = get_kernel(params['gp_kernel'],params['gp_kernel_diag'])
        wp_kernel = get_kernel(params['wp_kernel'],params['wp_kernel_diag'])

        V = get_scale_matrix(params)

        gp = models.GaussianProcess(kernel=gp_kernel,num_dims=params['D'])
        wp = models.WishartProcess(kernel=wp_kernel,nu=params['nu'],V=V)

        # %% Likelihood
        likelihood = eval('models.'+params['likelihood']+'()')

        with numpyro.handlers.seed(rng_seed=params['seed']):
            mu = gp.sample(x)
            sigma = wp.sample(x)
            y = jnp.stack([likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(params['N'])])
        

        self.x,self.y,self.mu,self.sigma,self.x_test,self.y_test,self.mu_test,self.sigma_test = split_data(
            x[:,None],y,params['train_trial_prop'],params['train_condition_prop'],
            seed=params['seed'],mu=mu,sigma=sigma
        )
        self.x = self.x.squeeze()
        self.x_test = self.x_test.squeeze()

    def load_data(self):
        return self.x, self.y
    
    def load_test_data(self):
        return self.x_test, self.y_test

def split_data(
        x,y,train_trial_prop,train_condition_prop,seed,
        mu=None,sigma=None
    ):
        N,M,D = y.shape
        
        train_conditions = jax.random.choice(
            jax.random.PRNGKey(seed),
            shape=(int(train_condition_prop*M),),
            a=np.arange(M),
            replace=False
        ).sort()
        
        train_trials = jax.random.choice(
            jax.random.PRNGKey(seed),
            shape=(int(N*train_trial_prop),),
            a=np.arange(N),
            replace=False
        ).sort()

        
        test_conditions = jnp.setdiff1d(np.arange(M),train_conditions).tolist()
        test_trials = jnp.setdiff1d(np.arange(N),train_trials).tolist()
        
        y_test = {
            'x':y[test_trials,:,:][:,train_conditions],
            'x_test':y[:,test_conditions]
        }

        x_train = x[train_conditions,:]
        y_train = y[train_trials,:,:][:,train_conditions]
        x_test = x[test_conditions,:]

        if mu is not None:  mu_test,mu_train = mu[test_conditions,:],mu[train_conditions,:]
        else: mu_test,mu_train = None,None
        
        if sigma is not None:  sigma_test,sigma_train = sigma[test_conditions,:,:],sigma[train_conditions,:,:]
        else: sigma_test,sigma_train = None,None
        
        return x_train,y_train,mu_train,sigma_train,x_test,y_test,mu_test,sigma_test


        
class CovarianceModel:
    @staticmethod
    def low_rank(N,K,seed,g=1):
        '''if N==K returns dense psd matrix
        '''
        key = jax.random.PRNGKey(seed)
        U = np.sqrt(g)*jax.random.normal(key,shape=(N,K))/K
        return U@U.T

    # %%
    @staticmethod
    def clustered(
            N,C,seed,C_std=.2,
            clusters_mean=1.,clusters_stds=.1,clusters_prob=1,
            external_mean=.1,external_stds=.1,external_prob=.5
        ):
        key = jax.random.PRNGKey(seed)
        
        bdiag = lambda c,v : block_diag(
            *[jnp.ones((c[i],c[i]))*v[i] for i in range(len(c))
        ])
        csz = jnp.round((C_std*N/C)*jax.random.normal(key,shape=(C,))+N/C).astype(int)
        csz = csz.at[-1].set(N-csz[:-1].sum()) 
        
        mask = 1-bdiag(csz,np.ones(C))

        J_prob = bdiag(csz,clusters_prob+jnp.zeros((C))) + bdiag([csz.sum()],[external_prob])*mask
        J_mean = bdiag(csz,clusters_mean*csz.mean()/csz) + bdiag([csz.sum()],[external_mean])*mask
        J_stds = bdiag(csz,clusters_stds+jnp.zeros((C))) + bdiag([csz.sum()],[external_stds])*mask

        J = jax.random.bernoulli(key,shape=J_prob.shape,p=J_prob)*(jax.random.normal(key,shape=(N,N))*J_stds+J_mean)
        W = np.tril(J) + np.triu(J.T, 1)
        
        return W
    
    @staticmethod
    def multi_region(
        N,C,seed,C_std=.2,diag=1,g=1,
    ):
        key = jax.random.PRNGKey(seed)
        coarse = jax.random.normal(key,shape=(C,C)) + diag*jnp.eye(C)
        csz = jnp.round((C_std*N/C)*jax.random.normal(key,shape=(C,))+N/C).astype(int)
        csz = csz.at[-1].set(N-csz[:-1].sum()) 
        J = np.hstack(
            [np.vstack(
                [coarse[i,j]+jax.random.normal(key,shape=(csz[i],csz[j])) for i in range(C)]
            ) for j in range(C)]
        )
        W = np.tril(J) + np.triu(J.T, 1)
        return g*W
