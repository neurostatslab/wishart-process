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
import utils

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

    dataset_params, model_params, variational_params, visualization_params, comparison_params = pm['dataset_params'], pm['model_params'], pm['variational_params'], pm['visualization_params'], pm['comparison_params']

    seed = pm['model_params']['seed']
    file = args.output

    if not os.path.exists(file): os.makedirs(file)

    # %%
    dataloader = eval('loader.'+dataset_params['name'])(params=dataset_params)
    x,y = dataloader.load_data()
    mu_empirical = y.mean(0)
    
    # %%
    gp_kernel = utils.get_kernel(model_params['gp_kernel'],model_params['gp_kernel_diag'])
    N = y.shape[2]
    
    print('Trials, Conditions, Neurons: ', y.shape)
    
    gp = models.GaussianProcess(kernel=gp_kernel,N=N)

    prec = True if 'Precision' in model_params['likelihood']  else False

    empirical = jnp.cov((y - y.mean(0)[None]).reshape(y.shape[0]*y.shape[1],y.shape[2]).T)
    if prec: empirical = jnp.linalg.inv(empirical)

    wp_kernel = utils.get_kernel(model_params['wp_kernel'],model_params['wp_kernel_diag'])

    P = model_params['P'] if 'P' in model_params.keys() else model_params['P_scale']*N
    optimize_L = True if 'optimize_L' in model_params.keys() and model_params['optimize_L'] else False

    wp_sample_diag = model_params['wp_sample_diag'] if 'wp_sample_diag' in model_params else 1e0
    # TODO: Wishart vs. WishartLRD should be a parameter

    likelihood = eval('models.'+model_params['likelihood'])(N)

    if model_params['likelihood'] == 'PoissonConditionalLikelihood':
        # Fix this, for real data there's no dataloader.V
        V = dataloader.V
    if model_params['likelihood'] == 'NormalConditionalLikelihood':
        V = empirical+wp_sample_diag*jnp.eye(N)
    
    
    wp = eval('models.'+model_params['prior'])(
        kernel=wp_kernel,P=P,
        V=V, optimize_L=optimize_L,
        diag_scale=wp_sample_diag
    )

    
    
    if model_params['likelihood'] == 'PoissonConditionalLikelihood':
        likelihood.initialize_rate(y)

    joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

    if 'init_gt' in model_params and model_params['init_gt']: 
        init = {
            'G':dataloader.mu.T[:,None],
            'F':dataloader.F
        }
    else:
        if model_params['likelihood'] == 'PoissonConditionalLikelihood':
            init_G = likelihood.gain_inverse_fn(y.mean(0).T[:,None])-likelihood.rate[:,None,None]
            init = {
                'G':init_G,
                'g':init_G.transpose(1,2,0).repeat(y.shape[0],0),
            }
        if model_params['likelihood'] == 'NormalConditionalLikelihood':
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
            optimizer,x.squeeze(),y,
            n_iter=variational_params['n_iter'],key=key,
            num_particles=variational_params['num_particles']
        )

        joint.update_params(varfam.posterior)
        varfam.save(file+'varfam')

    x_test, y_test = dataloader.load_test_data()

    posterior = models.NormalGaussianWishartPosterior(joint,varfam,x)
    if model_params['likelihood'] == 'PoissonConditionalLikelihood':
        with numpyro.handlers.seed(rng_seed=seed):
            mu_hat = posterior.mean_stat(lambda x: x, x)
            sigma_hat = posterior.mean_stat(lambda x: jnp.einsum('cd,ck->cdk',x-mu_hat,x-mu_hat), x)
            if len(x_test) > 0:
                mu_test_hat = posterior.mean_stat(lambda x: x, x_test)
                sigma_test_hat = posterior.mean_stat(lambda x: jnp.einsum('cd,ck->cdk',x-mu_test_hat,x-mu_test_hat), x_test)
        
    if model_params['likelihood'] == 'NormalConditionalLikelihood':
        with numpyro.handlers.seed(rng_seed=seed):
            mu_hat, sigma_hat, F_hat = posterior.sample(x)
            if len(x_test) > 0:
                mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(x_test)

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
        visualizations.visualize_pc(
            mu_hat[:,None],sigma_hat,
            pc=y.reshape(y.shape[0]*y.shape[1],-1),
            save=True,file=file+'inferred'
        )


    if 'plot_box' in visualization_params:
        compared = evaluation.compare(y,prec=prec,params=comparison_params)
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

        if model_params['likelihood'] == 'PoissonConditionalLikelihood':
        
            with numpyro.handlers.seed(rng_seed=seed):
                mu_hat_g, sigma_hat_g, _ = posterior.mode(x)

            with numpyro.handlers.seed(rng_seed=seed):
                lpp['wishart'] = likelihood.log_prob(y_test['x'],mu_hat_g,sigma_hat_g).flatten()
                lpp['train'] = dataloader.likelihood.log_prob(y,dataloader.mu_g,dataloader.sigma_g).flatten()
                lpp['true'] = dataloader.likelihood.log_prob(y_test['x'],dataloader.mu_g,dataloader.sigma_g).flatten()

                if len(x_test) > 0:
                    with numpyro.handlers.seed(rng_seed=seed):
                        mu_test_hat_g, sigma_test_hat_g, _ = posterior.mode(x_test)
                    lpp['wishart ho'] = likelihood.log_prob(y_test['x_test'], mu_test_hat_g, sigma_test_hat_g).flatten()


        if model_params['likelihood'] == 'NormalConditionalLikelihood':
            for key in compared.keys():
                lpp[key] = likelihood.log_prob(y_test['x'],mu_empirical,compared[key].transpose(2,0,1)).flatten()
            
            if len(x_test) > 0:
                lpp['wishart ho'] = likelihood.log_prob(y_test['x_test'], mu_test_hat, sigma_test_hat).flatten()
                
            if hasattr(dataloader,'mu') and dataloader.mu is not None:
                try: lpp['train'] = dataloader.likelihood.log_prob(y,dataloader.mu,dataloader.sigma).flatten()
                except: pass
                try: lpp['true'] = dataloader.likelihood.log_prob(y_test['x'],dataloader.mu,dataloader.sigma).flatten()
                except: pass

        visualizations.plot_box(
            lpp,titlestr='Log Posterior Predictive',
            save=True,file=file+'lpp'
        )

        jnp.save(file+'lpp',lpp)
    
    if 'visualize_qda' in visualization_params:
        # QDA Analysis
        if model_params['likelihood'] == 'NormalConditionalLikelihood':
            lpp = {}
            mu_empirical = y.mean(0)

            correct = {}
            for c in range(y_test['x'].shape[1]): 
                for key in compared.keys():
                    lpp[key] = likelihood.log_prob(
                        y_test['x'][:,c][:,None],
                        mu_empirical,compared[key].transpose(2,0,1)
                    )
                lpp['true'] = likelihood.log_prob(y_test['x'][:,c][:,None],dataloader.mu,dataloader.sigma)
                for key in lpp.keys():
                    if key not in correct.keys(): correct[key] = 0
                    correct[key] += (lpp[key].argmax(1) == c).sum()
            
            jnp.save(file+'correct',correct)

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
        try:
            with numpyro.handlers.seed(rng_seed=seed):
                if len(x_test) > 0:
                    for i in range(2):
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
        except:
            pass

        if hasattr(dataloader,'mu_test') and dataloader.mu_test is not None:
            try:
                if len(x_test) > 0:
                    visualizations.visualize_pc(
                        dataloader.mu_test[:,None],dataloader.sigma_test,
                        pc=y_test['x_test'].reshape(y_test['x_test'].shape[0]*y_test['x_test'].shape[1],-1),
                        save=True,file=file+'test_true'
                    )
            except:
                pass
    

