# %%
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
import evaluation


# %% Make a fake dataset
N = 20  # time points
D = 3
seed = 0
num_samples = 20

rng = [-100,100]
key = jax.random.PRNGKey(seed)

x = jnp.linspace(rng[0], rng[1], N)
sigma_m = 50.
sigma_c = 100.

# Prior
kernel_gp = lambda x, y: 1e4*(1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_m**2)))
kernel_wp = lambda x, y: 1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_c**2))

gp = models.GaussianProcess(kernel=kernel_gp,num_dims=D)
wp = models.WishartProcess(kernel=kernel_wp,nu=D+1,V=1e2*jnp.eye(D))

# Likelihood
key, _key = jax.random.split(key)
mu = gp.sample(_key, x)

key, _key = jax.random.split(key)
sigma = wp.sample(_key, x)

keys = jax.random.split(key, N+1)
key = keys[0]
y = models.ConditionalLikelihood(mu,sigma).sample(keys[1:],num_samples=num_samples)

print(y.shape)
visualizations.visualize_pc(
    mu.transpose(2,1,0),sigma.transpose(2,0,1),pc=y.reshape(y.shape[0]*y.shape[1],-1),
    save=False,file='../results/true'
)

# %% Joint

# vwp = models.NormalGaussianWishartProcess(gp,wp) # true joint
# Model misspecification
sigma_empirical = jnp.stack([jnp.cov(y[:,i,:].T) for i in range(y.shape[1])]).mean(0)
gp_ = models.GaussianProcess(
    kernel=lambda x, y: 1e10*(1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_m**2))),num_dims=D
)
wp_ = models.WishartProcess(
    kernel=lambda x, y: 1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_c**2)),
    nu=2*D,V=wp.L.T@wp.L
)
vwp_ = models.NormalGaussianWishartProcess(gp_,wp_) 

# %% Inference
# varfam = inference.VariationalDelta(vwp_.model)
varfam = inference.VariationalNormal(vwp_.model)

adam = optim.Adam(1e-1)
key, _key = jax.random.split(key)
varfam.infer(adam,x,y,n_iter=20000,key=_key)

# %% Visualization
visualizations.plot_loss(
    [varfam.losses],xlabel='Iteration',ylabel='ELBO',titlestr='Training Loss',colors=['k'],
    save=False,file='../results/losses'
)

# %%
key, _key = jax.random.split(key)
mu_hat,sigma_hat = varfam.sample(key=_key)

visualizations.visualize_pc(
    mu_hat.transpose(2,1,0),sigma_hat.transpose(2,0,1),pc=y.reshape(y.shape[0]*y.shape[1],-1),
        save=False,file='../results/inferred'+str(i)
)

# %% Evaluation

compared = evaluation.compare(y)
compared['wishart'] = sigma_hat
performance = evaluation.evaluate(compared,sigma)

visualizations.plot_box(performance)
