# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

# %%
from jax import random
import jax.numpy as np

import visualizations
import models
import inference
import jax.numpy as jnp
import jax
from numpyro import optim
import evaluation
import numpyro
import loader

import argparse
import yaml

import os

# %%
def get_args():
    '''Parsing the arguments when this file is called from console
    '''
    parser = argparse.ArgumentParser(description='Runner for CCM',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c', metavar='Configuration',help='Configuration file address',default='/')
    parser.add_argument('--output', '-o', metavar='Output',help='Folder to save the output and results',default='/')
    
    return parser.parse_args()

# %%
if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as stream: pm = yaml.safe_load(stream)

    dataset_params, model_params, variational_params, visualization_params = pm['dataset_params'], pm['model_params'], pm['variational_params'], pm['visualization_params']

    seed = pm['model_params']['seed']
    file = args.output

    if not os.path.exists(file): os.makedirs(file)

    # %%
    dataloader = eval('loader.'+dataset_params['name'])(params=dataset_params)
    x,y = dataloader.load_data()
    
    
    # %%
    gp_kernel = loader.get_kernel(model_params['gp_kernel'])
    D = y.shape[2]
    
    print('Trials, Conditions, Neurons: ', y.shape)
    
    gp = models.GaussianProcess(kernel=gp_kernel,num_dims=D)
    empirical = jnp.cov((y - y.mean(0)[None]).reshape(y.shape[0]*y.shape[1],y.shape[2]).T)
    
    wp_kernel = loader.get_kernel(model_params['wp_kernel'])

    nu = model_params['nu'] if 'nu' in model_params.keys() else model_params['nu_scale']*D
    wp = models.WishartProcess(
        kernel=wp_kernel,nu=nu,
        V=empirical
    )

    likelihood = eval('models.'+model_params['likelihood']+'()')
    joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

    # %%
    varfam = eval('inference.'+variational_params['guide'])(
        joint.model,init={'G':y.mean(0).T[:,None]}
    )
    optimizer = eval('optim.'+variational_params['optimizer']['type'])(
        variational_params['optimizer']['step_size']
    )
    key = jax.random.PRNGKey(seed)

    if pm['load']:
        varfam.load(file+'varfam.npy')
    else:
        varfam.infer(
            optimizer,x,y,
            n_iter=variational_params['n_iter'],key=key,
            num_particles=variational_params['num_particles']
        )
        varfam.update_params(joint)
        varfam.save(file+'varfam')

    # %% Visualization
    if 'visualize_pc' in visualization_params:
        visualizations.visualize_pc(
            dataloader.mu[:,None],dataloader.sigma,
            pc=y.reshape(y.shape[0]*y.shape[1],-1),
            save=True,file=file+'true'
        )

    if 'plot_tuning' in visualization_params:
        visualizations.plot_tuning(
            x, dataloader.mu, titlestr='Tuning', 
            save=True,file=file+'tuning'
        )
    if 'plot_loss' in visualization_params:
        visualizations.plot_loss(
            [varfam.losses],xlabel='Iteration',ylabel='ELBO',
            titlestr='Training Loss',colors=['k'],
            save=True,file=file+'losses'
        )

    if 'visualize_pc' in visualization_params:

        with numpyro.handlers.seed(rng_seed=seed):
            F,mu_hat,sigma_hat = varfam.sample()

        visualizations.visualize_pc(
            mu_hat[:,None],sigma_hat,
            pc=y.reshape(y.shape[0]*y.shape[1],-1),
            save=True,file=file+'inferred'
        )


    if 'plot_box' in visualization_params:
        compared = evaluation.compare(y)
        compared['wishart'] = sigma_hat.transpose(1,2,0)
        performance = evaluation.evaluate(compared,dataloader.sigma.transpose(1,2,0))

        visualizations.plot_box(
            performance,titlestr='Difference Operator Norm',
            save=True,file=file+'cov_comparison'
        )

        mse = lambda x,y: jnp.sqrt(((x-y)**2).sum(-1))

        mu_empirical = y.mean(0)
        performance_mean = {}
        performance_mean['empirical'] = mse(mu_empirical,dataloader.mu)
        performance_mean['wishart'] = mse(mu_hat,dataloader.mu)

        visualizations.plot_box(
            performance_mean,titlestr='Mean MSE',
            save=True,file=file+'mean_comparison'
        )

    x_test, y_test = dataloader.load_test_data()
    posterior = models.NormalGaussianWishartPosterior(joint,varfam,x)
    
    if 'visualize_pc' in visualization_params and 'x_new' in y_test.keys():
        with numpyro.handlers.seed(rng_seed=seed):
            
            for i in range(3):
                mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(x_test)
                visualizations.visualize_pc(
                    mu_test_hat[:,None],sigma_test_hat,
                    pc=y_test['x_new'].reshape(y_test['x_new'].shape[0]*y_test['x_new'].shape[1],-1),
                    save=True,file=file+'test_inferred_'+str(i)
                )

                visualizations.visualize_pc(
                    mu_test_hat[:,None],sigma_test_hat,
                    pc=y.reshape(y.shape[0]*y.shape[1],-1),
                    save=True,file=file+'test_inferred_train_data_'+str(i)
                )

        visualizations.visualize_pc(
            dataloader.mu_test[:,None],dataloader.sigma_test,
            pc=y_test['x_new'].reshape(y_test['x_new'].shape[0]*y_test['x_new'].shape[1],-1),
            save=True,file=file+'test_true'
        )


    if 'plot_box' in visualization_params:
        # posterior predictive likelihood
        lpp = {}

        if 'lw' in compared.keys():
            lpp['lw'] = likelihood.log_prob(y_test['x'],mu_empirical,compared['lw'].transpose(2,0,1)).flatten()
        if 'lasso' in compared.keys():
            lpp['lasso'] = likelihood.log_prob(y_test['x'],mu_empirical,compared['lasso'].transpose(2,0,1)).flatten()
        if 'empirical' in compared.keys():
            lpp['empirical'] = likelihood.log_prob(y_test['x'],mu_empirical,compared['empirical'].transpose(2,0,1)).flatten()

        with numpyro.handlers.seed(rng_seed=seed):
            lpp['w test'] = posterior.log_prob(x, y_test['x'], vi_samples=20, gp_samples=1).flatten()
            if  'x_new' in y_test.keys():
                lpp['w test ho'] = posterior.log_prob(x_test, y_test['x_new'], vi_samples=20, gp_samples=1).flatten()
        
        lpp['train'] = likelihood.log_prob(y,dataloader.mu,dataloader.sigma).flatten()

        visualizations.plot_box(
            lpp,titlestr='Log Posterior Predictive',
            save=True,file=file+'lpp'
        )





