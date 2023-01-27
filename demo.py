# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import visualizations
import models
import inference
import jax.numpy as jnp
import jax
from numpyro import optim


# %% Make a fake dataset
N = 20  # time points
D = 2

rng = [-3,3]
key = jax.random.PRNGKey(111)
x = jnp.linspace(rng[0], rng[1], N)
sigma_m = .3
sigma_c = .6

# Prior
gp = models.GaussianProcess(kernel=lambda x, y: jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_m**2)),num_dims=D)
wp = models.WishartProcess(kernel=lambda x, y: jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_c**2)),nu=5,V=.001*jnp.eye(D))

# Likelihood
mu = gp.sample(key, x)
sigma = wp.sample(key, x)

keys = jax.random.split(key, N)
y = models.ConditionalLikelihood(mu,sigma).sample(keys,num_samples=10)

print(y.shape)
visualizations.visualize_pc(
    mu.transpose(2,1,0),sigma.transpose(2,0,1),pc=y.reshape(y.shape[0]*y.shape[1],-1),
    save=True,file='../results/true'
)

# Joint
vwp = models.NormalGaussianWishartProcess(gp,wp) 

# %% Inference

# varfam = models.VariationalDetla(vwp.model)
varfam = inference.VariationalNormal(vwp.model)

adam = optim.Adam(1e-3)
varfam.infer(adam,x,y,n_iter=10000,key=jax.random.PRNGKey(0))

visualizations.plot_loss(
    [varfam.losses],xlabel='Iteration',ylabel='ELBO',titlestr='Training Loss',colors=['k'],
    save=True,file='../results/losses'
)

for i in range(3):
    mu_hat,sigma_hat = varfam.sample(jax.random.PRNGKey(i))

    visualizations.visualize_pc(
        mu_hat.transpose(2,1,0),sigma_hat.transpose(2,0,1),pc=y.reshape(y.shape[0]*y.shape[1],-1),
            save=True,file='../results/inferred'+str(i)
    )
