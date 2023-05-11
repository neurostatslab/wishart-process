# -*- coding: utf-8 -*-
"""
@authors: Amin, Isabel
"""

# %%
import jax.numpy as jnp
from sklearn.covariance import LedoitWolf
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import EmpiricalCovariance

# %%

def compare(y,prec=False):
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
                GraphicalLasso().fit(y[:, i, :]).precision_ for i in range(y.shape[1])
            ], axis=-1)
        else:
            result['lasso'] = jnp.stack([
                GraphicalLasso().fit(y[:, i, :]).covariance_ for i in range(y.shape[1])
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
            
        
        corr = {}
        for m in methods:
            corr[m] = ((
                y_reshaped[m].flatten()-
                y_reshaped['empirical'].flatten()
            )**2).mean()
        return corr