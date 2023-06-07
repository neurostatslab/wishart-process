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
import numpyro

# %load_ext autoreload
# %autoreload 2

# %% Make a fake dataset
N = 20  # time points
D = 2
seed = 1
num_samples = 20

rng = [-10,10]

x = jnp.linspace(rng[0], rng[1], N)
sigma_m = 10.
sigma_c = 10.

# Prior
kernel_gp = lambda x, y: 1e1*(1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_m**2)))
kernel_wp = lambda x, y: 1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_c**2))

gp = models.GaussianProcess(kernel=kernel_gp,num_dims=D)
wp = models.WishartProcess(kernel=kernel_wp,nu=D+1,V=1e-2*jnp.eye(D))

# Likelihood
likelihood = models.NormalConditionalLikelihood()

with numpyro.handlers.seed(rng_seed=seed):
    mu = gp.sample(x)
    sigma = wp.sample(x)
    data = [likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(num_samples)]
    y = jnp.stack(data)

visualizations.visualize_pc(
    mu[:,None],sigma,pc=y.reshape(y.shape[0]*y.shape[1],-1)
)

# %% Joint
joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

# %% Inference
varfam = inference.VariationalNormal(joint.model)

adam = optim.Adam(1e-1)
key = jax.random.PRNGKey(seed)
varfam.infer(adam,x,y,n_iter=20000,key=key)
joint.update_params(varfam.posterior)

# %% Visualization
visualizations.plot_loss(
    [varfam.losses],xlabel='Iteration',ylabel='ELBO',titlestr='Training Loss',colors=['k'],
)

# %%
posterior = models.NormalGaussianWishartPosterior(joint,varfam,x)
with numpyro.handlers.seed(rng_seed=seed):
    mu_hat, sigma_hat, F_hat = posterior.sample(x)

visualizations.visualize_pc(
    mu_hat[:,None],sigma_hat,pc=y.reshape(y.shape[0]*y.shape[1],-1)
)
# %% Evaluation

compared = evaluation.compare(y)
compared['wishart'] = sigma_hat.transpose(1,2,0)
performance = evaluation.evaluate(compared,sigma.transpose(1,2,0))

visualizations.plot_box(performance)
