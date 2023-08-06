# -*- coding: utf-8 -*-
"""
@authors: Amin, Isabel
"""

# %%
import jax.numpy as jnp
from sklearn.covariance import LedoitWolf
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import FactorAnalysis

from posce import PopulationShrunkCovariance
from nilearn.connectome import vec_to_sym_matrix

# %%

def compare(y,prec=False,params={}):
    result = {}

    try:
        if prec:
            result['lw'] = jnp.stack([
                LedoitWolf().fit(y[:,i,:]).precision_ for i in range(y.shape[1])
            ], axis=-1)
        else:
            result['lw'] = jnp.stack([
                LedoitWolf().fit(y[:,i,:]).covariance_ for i in range(y.shape[1])
            ], axis=-1)
    except: pass
    try:
        if prec:
            result['lasso'] = jnp.stack([
                GraphicalLasso(alpha=params['alpha']).fit(y[:, i, :]).precision_ for i in range(y.shape[1])
            ], axis=-1)
        else:
            result['lasso'] = jnp.stack([
                GraphicalLasso(alpha=params['alpha']).fit(y[:, i, :]).covariance_ for i in range(y.shape[1])
            ], axis=-1)
    except: pass
    try:
        if prec:
            result['empirical'] = jnp.stack([
                EmpiricalCovariance().fit(y[:, i, :]).precision_ for i in range(y.shape[1])
            ], axis=-1)
        else:
            result['empirical'] = jnp.stack([
                EmpiricalCovariance().fit(y[:, i, :]).covariance_ for i in range(y.shape[1])
            ], axis=-1)
    except: pass
    try:
        if prec:
            result['fa'] = jnp.stack([
                FactorAnalysis(n_components=params['n_components']).fit(y[:, i, :]).get_precision() for i in range(y.shape[1])
            ], axis=-1)
        else:
            result['fa'] = jnp.stack([
                FactorAnalysis(n_components=params['n_components']).fit(y[:, i, :]).get_covariance() for i in range(y.shape[1])
            ], axis=-1)
    except: pass
    try:
        posce = PopulationShrunkCovariance(shrinkage=params['shrinkage'])
        posce.fit(y.transpose(1,0,2))
        shrunk_connectivities = posce.transform(y.transpose(1,0,2))
        if prec:
            result['posce'] = jnp.stack([jnp.linalg.inv(vec_to_sym_matrix(c)) for c in shrunk_connectivities],axis=-1)
        else:
            result['posce'] = jnp.stack([vec_to_sym_matrix(c) for c in shrunk_connectivities],axis=-1)
    except: pass
    
    
    return result

def evaluate(methods,sigma,ord=2,prec=False):
    N = sigma.shape[2]

    result = {}
    for key in methods.keys():
        if prec:
            try:
                result[key] = [jnp.linalg.norm(
                    jnp.linalg.inv(methods[key][:,:,i]) - jnp.linalg.inv(sigma[:,:,i]), ord=ord
                ) for i in range(N)]
            except: continue
        else:
            result[key] = [jnp.linalg.norm(
                methods[key][:,:,i] - sigma[:,:,i], ord=ord
            ) for i in range(N)]

    return result

import numpy as np
from scipy.stats import rankdata


#  %%
def evaluate_var_smoothness(x,y,methods):
    if x.shape[1] == 2:
        idx = rankdata(x, method='dense',axis=0)-1

        y_reshaped = {}
        for m in range(len(y)):
            a = np.nan*np.zeros((idx.max(0)+1).tolist() + [y[0].shape[1]])
            
            for i in range(len(x)): 
                a[idx[i][0],idx[i][1]] = y[m][i]
        
            y_reshaped[methods[m]] = a.copy()
            
        
        mse = {}
        for m in methods:
            mse[m] = np.median(((
                y_reshaped[m].flatten()-
                y_reshaped['empirical'].flatten()
            )**2))
        return mse
    
    if len(x.shape) == 1 or x.shape[1]==1:
        y_reshaped = {}
        for m in range(len(y)):
            y_reshaped[methods[m]] = y[m]

        mse = {}
        for m in methods:
            mse[m] = np.median(((
                y_reshaped[m].flatten()-
                y_reshaped['empirical'].flatten()
            )**2))
        return mse


# %%
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import linalg

def fisher_information(x,mu_prime,sigma):
    fi = np.array([
        mu_prime[:,[i]].T@np.linalg.inv(sigma[i])@mu_prime[:,[i]] for i in range(len(x))
    ]).squeeze()
    
    return fi

def top_eig_overlap(x,mu_prime,sigma):
    overlap = np.array([abs(cosine_similarity(
            mu_prime[:,[i]].T,
            linalg.eigsh(np.array(sigma[i]),1)[1].T
        )) for i in range(len(x))
    ]).squeeze()

    return overlap