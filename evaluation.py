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

def compare(y):
    result = {}

    result['lw'] = jnp.stack([
        LedoitWolf().fit(y[:,i,:]).covariance_ for i in range(y.shape[1])
    ], axis=-1)
    result['lasso'] = jnp.stack([
        GraphicalLasso().fit(y[:, i, :]).covariance_ for i in range(y.shape[1])
    ], axis=-1)
    result['empirical'] = jnp.stack([
        EmpiricalCovariance().fit(y[:, i, :]).covariance_ for i in range(y.shape[1])
    ], axis=-1)

    return result

def evaluate(methods,sigma):
    N = sigma.shape[2]

    result = {}
    for key in methods.keys():
        result[key] = [jnp.linalg.norm(
            methods[key][:,:,i] - sigma[:,:,i], ord=2
        ) for i in range(N)]

    return result