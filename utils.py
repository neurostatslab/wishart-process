# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax
from functools import reduce
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag

# %%
def get_kernel(params,diag):
    '''Returns the full kernel of multi-dimensional condition spaces
    '''
    if len(params) > 1: 
        return lambda x,y: diag*jnp.all(x==y)+reduce(
            lambda a,b: a*b, [
                _get_kernel(params[i]['type'],params[i])(x[i],y[i]) for i in range(len(params))
            ])
    else: 
        return lambda x,y: diag*(x==y)+_get_kernel(params[0]['type'],params[0])(x,y)
        

def _get_kernel(kernel,params):
    '''Private function, returns the kernel corresponding to a single dimension
    '''
    if kernel == 'periodic': 
        return lambda x,y: params['scale']*jnp.exp(-2*(jnp.sin(jnp.pi*jnp.abs(x-y)/params['normalizer'])**2)/(params['sigma']**2))
    if kernel == 'RBF': 
        return lambda x,y: params['scale']*jnp.exp(-(jnp.linalg.norm((x-y)/params['normalizer'])**2)/(2*params['sigma']**2))


# %%
def split_data(
        x,y,train_trial_prop,train_condition_prop,seed,
        mu=None,sigma=None,F=None,mu_g=None,sigma_g=None
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

        if mu_g is not None:  mu_g_test,mu_g_train = mu_g[test_conditions,:],mu_g[train_conditions,:]
        else: mu_g_test,mu_g_train = None,None
        
        if sigma is not None:  sigma_test,sigma_train = sigma[test_conditions,:,:],sigma[train_conditions,:,:]
        else: sigma_test,sigma_train = None,None

        if sigma_g is not None:  sigma_g_test,sigma_g_train = sigma_g[test_conditions,:,:],sigma_g[train_conditions,:,:]
        else: sigma_g_test,sigma_g_train = None,None

        if F is not None:  F_test,F_train = F[:,:,test_conditions],F[:,:,train_conditions]
        else: F_test,F_train = None,None

        
        return x_train,y_train,mu_train,sigma_train,x_test,y_test,mu_test,sigma_test,F_train,F_test,mu_g_train,mu_g_test,sigma_g_train,sigma_g_test


# %%
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances

def create_adjacency(x):
    idx = rankdata(x, method='dense',axis=0)-1
    dist = pairwise_distances(idx,metric='l1')
    dist[dist != 1] = 0
    return dist
        
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

    @staticmethod
    def exp_decay_eig(N,seed):
        key = jax.random.PRNGKey(seed)
        U = jax.random.orthogonal(key,N)
        Lambda = jnp.diag(jnp.logspace(0,-5,N))
        return U@Lambda@U.T
    
# %%
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
    if params['scale_type'] == 'exp_decay_eig':
        return params['epsilon']*CovarianceModel.exp_decay_eig(
            params['D'],seed=params['seed']
        )