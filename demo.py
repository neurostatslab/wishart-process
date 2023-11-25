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

# %% Make a fake dataset
C = 20 # conditions (time points)
N = 2  # number of neurons (dimensions) 
K = 20 # number of trials

data_seed = 1 # seed for generating data

rng = [-10,10] # range of input conditions

x = jnp.linspace(rng[0], rng[1], C)
sigma_m = 10. # smoothness of GP
sigma_c = 10. # smoothness of WP

# Prior distribution (GP and WP)
kernel_gp = lambda x, y: 1e1*(1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_m**2)))
kernel_wp = lambda x, y: 1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_c**2))

gp = models.GaussianProcess(kernel=kernel_gp,N=N)
wp = models.WishartProcess(kernel=kernel_wp,P=N+1,V=1e-2*jnp.eye(N))

# Likelihood model (Multivariate Normal)
likelihood = models.NormalConditionalLikelihood(N)

with numpyro.handlers.seed(rng_seed=data_seed):
    mu = gp.sample(x)
    sigma = wp.sample(x)
    data = [likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(K)]
    y = jnp.stack(data)

visualizations.visualize_pc(
    mu[:,None],sigma,pc=y.reshape(y.shape[0]*y.shape[1],-1)
)

# %% Joint distribution
joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

# %% Inference (Mean Field Variational)
inference_seed = 2
varfam = inference.VariationalNormal(joint.model)

adam = optim.Adam(1e-1)
key = jax.random.PRNGKey(inference_seed)
varfam.infer(adam,x,y,n_iter=20000,key=key)
joint.update_params(varfam.posterior)

# %% Visualization
visualizations.plot_loss(
    [varfam.losses],xlabel='Iteration',ylabel='ELBO',titlestr='Training Loss',colors=['k'],
)

# %% Sampling from the posterior over means and covariances
posterior = models.NormalGaussianWishartPosterior(joint,varfam,x)
with numpyro.handlers.seed(rng_seed=inference_seed):
    mu_hat, sigma_hat, F_hat = posterior.sample(x)

visualizations.visualize_pc(
    mu_hat[:,None],sigma_hat,pc=y.reshape(y.shape[0]*y.shape[1],-1)
)
# %% Evaluation and comparison with other methods
compared = evaluation.compare(y)
compared['wishart'] = sigma_hat.transpose(1,2,0)
performance = evaluation.evaluate(compared,sigma.transpose(1,2,0))

visualizations.plot_box(performance)
