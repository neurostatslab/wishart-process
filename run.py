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
    mu_empirical = y.mean(0)
    
    # %%
    gp_kernel = loader.get_kernel(model_params['gp_kernel'],model_params['gp_kernel_diag'])
    D = y.shape[2]
    
    print('Trials, Conditions, Neurons: ', y.shape)
    
    gp = models.GaussianProcess(kernel=gp_kernel,num_dims=D)

    prec = True if 'Precision' in model_params['likelihood']  else False

    empirical = jnp.cov((y - y.mean(0)[None]).reshape(y.shape[0]*y.shape[1],y.shape[2]).T)
    if prec: empirical = jnp.linalg.inv(empirical)

    wp_kernel = loader.get_kernel(model_params['wp_kernel'],model_params['wp_kernel_diag'])

    nu = model_params['nu'] if 'nu' in model_params.keys() else model_params['nu_scale']*D
    optimize_L = True if 'optimize_L' in model_params.keys() and model_params['optimize_L'] else False

    wp_sample_diag = model_params['wp_sample_diag'] if 'wp_sample_diag' in model_params else 1e0
    # TODO: Wishart vs. WishartGamma should be a parameter
    wp = models.WishartGammaProcess(
        kernel=wp_kernel,nu=nu,
        V=empirical+wp_sample_diag*jnp.eye(D), optimize_L=optimize_L,
        diag_scale=wp_sample_diag
    )

    if 'likelihood_params' in model_params:
        likelihood = eval('models.'+model_params['likelihood']+'(**model_params[\'likelihood_params\'])')
    else:
        likelihood = eval('models.'+model_params['likelihood']+'()')
    joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

    if 'init_gt' in model_params and model_params['init_gt']: 
        init = {
            'G':dataloader.mu.T[:,None],
            'F':dataloader.F
        }
    else: 
        init = {
            'G':y.mean(0).T[:,None]
        }
    # %%
    varfam = eval('inference.'+variational_params['guide'])(
        joint.model,init=init
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

        joint.update_params(varfam.posterior)
        varfam.save(file+'varfam')

    x_test, y_test = dataloader.load_test_data()

    # %% Visualization
    if 'visualize_pc' in visualization_params:
        if hasattr(dataloader,'mu') and dataloader.mu is not None:
            visualizations.visualize_pc(
                dataloader.mu[:,None],dataloader.sigma,
                pc=y.reshape(y.shape[0]*y.shape[1],-1),
                save=True,file=file+'true'
            )

    if 'plot_tuning' in visualization_params:
        if hasattr(dataloader,'mu') and dataloader.mu is not None:
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
        posterior = models.NormalGaussianWishartPosterior(joint,varfam,x)
        with numpyro.handlers.seed(rng_seed=seed):
            mu_hat, sigma_hat, F_hat = posterior.sample(x)
            if len(x_test) > 0:
                mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(x_test)

        visualizations.visualize_pc(
            mu_hat[:,None],sigma_hat,
            pc=y.reshape(y.shape[0]*y.shape[1],-1),
            save=True,file=file+'inferred'
        )


    if 'plot_box' in visualization_params:
        compared = evaluation.compare(y,prec=prec)
        compared['grand-empirical'] = jnp.repeat(empirical[:,:,None],y.shape[1],2)
        compared['wishart'] = sigma_hat.transpose(1,2,0)
        
        if hasattr(dataloader,'sigma') and dataloader.sigma is not None:
            performance = evaluation.evaluate(compared,dataloader.sigma.transpose(1,2,0))

            visualizations.plot_box(
                performance,titlestr='Difference Operator Norm',
                save=True,file=file+'cov_comparison'
            )

            jnp.save(file+'performance',performance)

            if prec: 
                performance_cov = evaluation.evaluate(compared,dataloader.sigma.transpose(1,2,0),prec=prec)
                jnp.save(file+'performance_prec_cov',performance_cov)

            performance_fro = evaluation.evaluate(compared,dataloader.sigma.transpose(1,2,0),ord='fro')
            jnp.save(file+'performance_fro',performance_fro)

        mse = lambda x,y: jnp.sqrt(((x-y)**2).sum(-1))

        if hasattr(dataloader,'mu') and dataloader.mu is not None:
            performance_mean = {}
            performance_mean['empirical'] = mse(mu_empirical,dataloader.mu)
            performance_mean['wishart'] = mse(mu_hat,dataloader.mu)

            visualizations.plot_box(
                performance_mean,titlestr='Mean MSE',
                save=True,file=file+'mean_comparison'
            )

            jnp.save(file+'performance_mean',performance_mean)

        sigma_test_empirical = evaluation.compare(y_test['x'],prec=prec)['empirical']
        performance_test = evaluation.evaluate(compared,sigma_test_empirical)

        visualizations.plot_box(
            performance_test,titlestr='Difference Operator Norm (Test)',
            save=True,file=file+'cov_comparison_test'
        )

        jnp.save(file+'performance_test',performance_test)
    

    if 'plot_box' in visualization_params:
        # posterior predictive likelihood
        lpp = {}

        for key in compared.keys():
            lpp[key] = likelihood.log_prob(y_test['x'],mu_empirical,compared[key].transpose(2,0,1)).flatten()
        
        del lpp['wishart']
        if len(x_test) > 0:
            lpp['w test ho'] = likelihood.log_prob(y_test['x_test'], mu_test_hat, sigma_test_hat).flatten()
        lpp['w test'] = likelihood.log_prob(y_test['x'], mu_hat, sigma_hat).flatten()
        if hasattr(dataloader,'mu') and dataloader.mu is not None:
            lpp['train'] = likelihood.log_prob(y,dataloader.mu,dataloader.sigma).flatten()

        visualizations.plot_box(
            lpp,titlestr='Log Posterior Predictive',
            save=True,file=file+'lpp'
        )

        jnp.save(file+'lpp',lpp)

    if 'visualize_covariances' in visualization_params:
        for k in compared.keys():
            visualizations.visualize_covariances(
                compared[k].transpose(2,0,1)[:,None],
                titlestr=k,save=True,file=file+'covariance_image_'+k
            )
        if hasattr(dataloader,'sigma') and dataloader.sigma is not None:
            visualizations.visualize_covariances(
                dataloader.sigma[:,None],
                titlestr='true',save=True,file=file+'covariance_image_true'
            )

    if 'visualize_pc' in visualization_params and 'x_test' in y_test.keys():
        with numpyro.handlers.seed(rng_seed=seed):
            if len(x_test) > 0:
                for i in range(3):
                    mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(x_test)
                    visualizations.visualize_pc(
                        mu_test_hat[:,None],sigma_test_hat,
                        pc=y_test['x_test'].reshape(y_test['x_test'].shape[0]*y_test['x_test'].shape[1],-1),
                        save=True,file=file+'test_inferred_'+str(i)
                    )

                    visualizations.visualize_pc(
                        mu_test_hat[:,None],sigma_test_hat,
                        pc=y.reshape(y.shape[0]*y.shape[1],-1),
                        save=True,file=file+'test_inferred_train_data_'+str(i)
                    )

        if hasattr(dataloader,'mu_test') and dataloader.mu_test is not None:
            visualizations.visualize_pc(
                dataloader.mu_test[:,None],dataloader.sigma_test,
                pc=y_test['x_test'].reshape(y_test['x_test'].shape[0]*y_test['x_test'].shape[1],-1),
                save=True,file=file+'test_true'
            )


