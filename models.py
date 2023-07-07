# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import numpy as onp
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from jax import vmap

# %%
class WishartProcess:
    def __init__(self, kernel, nu, V, optimize_L=False, diag_scale=1e-1):
        self.kernel = kernel
        self.nu = nu
        self.num_dims = V.shape[0]
        # Wishart mean is V/nu
        self.L = jnp.linalg.cholesky(V/nu)
        self.optimize_L = optimize_L
        self.diag_scale=diag_scale

    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(ys)

    def f2sigma(self, F, L=None):
        if L is None: L = self.L
        diag = self.diag_scale*jnp.eye(self.num_dims)[:,:,None]
        fft = jnp.einsum('abn,cbn->acn',F[:,:-1],F[:,:-1]) + diag
        afft = jnp.einsum('ab,bcn->acn',L,fft) 
        sigma = jnp.einsum('abn,bc->nac',afft,L.T) 
        return sigma

    def sample(self, x):
        N = x.shape[0]
        L = numpyro.param('L', self.L) if self.optimize_L else self.L

        c_f = self.evaluate_kernel(x,x)

        F = numpyro.sample(
            'F',dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_f),
            sample_shape=(self.num_dims,self.nu)
        )
        self.F = F

        sigma = self.f2sigma(F,L)

        return sigma
    
    
    def posterior(self, X, Y, sigma, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y, sigma

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        
        f = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)
        K = K_x_x - K_X_x.T@Ki@K_X_x
        
        F = numpyro.sample(
            'F_test',dist.MultivariateNormal(f,covariance_matrix=K),
            sample_shape=(1,1)
        ).squeeze()

        sigma = self.f2sigma(F)

        return F, sigma
    
    def posterior_mode(self, X, Y, sigma, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y, sigma

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        
        F = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)
        sigma = self.f2sigma(F)

        return F, sigma

    
    def log_prob(self, x, F):
        # TODO: input to this fn must be sigma, not F
        N = x.shape[0]
        c_f = self.evaluate_kernel(x,x)
        LPF = dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_f).log_prob(F)
        return LPF

# %%
class WishartGammaProcess:
    def __init__(self, kernel, nu, V, optimize_L=False, diag_scale=1e-1):
        self.kernel = kernel
        self.nu = nu
        self.num_dims = V.shape[0]
        # Wishart mean is V/nu
        self.L = jnp.linalg.cholesky(V/max(nu,1))
        self.optimize_L = optimize_L
        self.diag_scale=diag_scale

    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(ys)

    def f2sigma(self, F, L=None):
        if L is None: L = self.L
        diag = self.diag_scale*jnp.stack([jnp.diag(jax.nn.softplus(F[:,-1,i])) for i in range(F.shape[2])],axis=-1)
        fft = jnp.einsum('abn,cbn->acn',F[:,:-1],F[:,:-1]) + diag
        afft = jnp.einsum('ab,bcn->acn',L,fft) 
        sigma = jnp.einsum('abn,bc->nac',afft,L.T) 
        
        return sigma

    def sample(self, x):
        N = x.shape[0]
        L = numpyro.param('L', self.L) if self.optimize_L else self.L

        c_f = self.evaluate_kernel(x,x)

        F = numpyro.sample(
            'F',dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_f),
            sample_shape=(self.num_dims,self.nu+1)
        )
        self.F = F

        sigma = self.f2sigma(F,L)

        return sigma

    
    def posterior(self, X, Y, sigma, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y, sigma

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        
        f = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)
        K = K_x_x - K_X_x.T@Ki@K_X_x
        
        F = numpyro.sample(
            'F_test',dist.MultivariateNormal(f,covariance_matrix=K),
            sample_shape=(1,1)
        )[0,0]

        sigma = self.f2sigma(F)

        return F, sigma
    
    def posterior_mode(self, X, Y, sigma, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y, sigma

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        
        F = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)

        sigma = self.f2sigma(F)

        return F, sigma
    
    def log_prob(self, x, F):
        # TODO: input to this fn must be sigma, not F
        N = x.shape[0]
        c_f = self.evaluate_kernel(x,x)
        LPF = dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_f).log_prob(F)
        return LPF
    
# %%
class GaussianProcess:
    def __init__(self, kernel, num_dims):
        self.kernel = kernel
        self.num_dims = num_dims


    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(ys)


    def sample(self, x):
        N = x.shape[0]
        c_g = self.evaluate_kernel(x,x)
        G = numpyro.sample(
            'G',dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_g),
            sample_shape=(self.num_dims,1)
        ).squeeze().T
        return G
    
    def posterior(self, X, Y, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)
        
        Ki   = jnp.linalg.inv(K_X_X)
        f = jnp.einsum('ij,jm->mi',(K_X_x.T@Ki),Y)
        
        K = K_x_x - K_X_x.T@Ki@K_X_x

        G_new = numpyro.sample(
            'G_test',dist.MultivariateNormal(f,covariance_matrix=K),
            sample_shape=(1,1)
        ).squeeze().T
        
        return G_new
    
    def posterior_mode(self, X, Y, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)
        
        Ki   = jnp.linalg.inv(K_X_X)
        G = jnp.einsum('ij,jm->mi',(K_X_x.T@Ki),Y)
        
        return G
    
    def log_prob(self, x, G):
        N = x.shape[0]
        c_g = self.evaluate_kernel(x,x)
        LPG = dist.MultivariateNormal(jnp.zeros((N)),covariance_matrix=c_g).log_prob(G)
        return LPG

# %% 
class NeuralTuningProcess:
    def __init__(self, num_dims, spread, amp):
        self.num_dims = num_dims
        self.spread = spread
        self.amp = amp

    def sample(self, x):
        # generate num_dims random phases
        # generate cosine response curves with the given spread
        p = numpyro.sample('phase', dist.Uniform(),sample_shape=(self.num_dims,))
        a = numpyro.sample('amp', dist.Uniform(low=.5,high=1.5),sample_shape=(self.num_dims,))
        return self.amp*a*(1+jnp.cos(((jnp.pi*x[:,None]/360.)-(p[None])*jnp.pi)/self.spread))
        


# %%
class NormalConditionalLikelihood:
    def sample(self,mu,sigma,ind=None,y=None):
        Y = numpyro.sample(
            'y',dist.MultivariateNormal(mu[ind,...],sigma[ind,...]),
            obs=y
        )
        return Y
    
    def log_prob(self,Y,mu,sigma,ind=None):
        LPY = dist.MultivariateNormal(mu[ind,...],sigma[ind,...]).log_prob(Y)
        return LPY
    
# %%
class PoissonConditionalLikelihood:
    def __init__(self,rate):
        self.rate = jnp.array(rate)

    def sample(self,mu,sigma,ind=None,y=None):
        rate = self.rate
        # rate = numpyro.param('rate', self.rate)

        G = numpyro.sample('g',dist.MultivariateNormal(mu[ind,...],sigma[ind,...]))
        Y = numpyro.sample('y',dist.Poisson(jax.nn.softplus(G[ind,...]+rate[None])).to_event(1),obs=y)
        
        self.G = G
        return Y
    
    def log_prob(self,G,Y,mu,sigma,ind=None):
        # TODO: sample from G, the input to this fn must be only Y
        LPG = dist.MultivariateNormal(mu[ind,...],sigma[ind,...]).log_prob(G)
        # jax.nn.softplus
        LPY = dist.Poisson(jax.nn.softplus(G[ind,...]+self.rate[None])).to_event(1).log_prob(Y)
        
        return LPG + LPY
    

# %%
class JointGaussianWishartProcess:
    def __init__(self, gp, wp, likelihood):
        self.gp = gp
        self.wp = wp
        self.likelihood = likelihood

    def model(self, x, y):
        B,N,D = y.shape

        sigma = self.wp.sample(x)
        G = self.gp.sample(x)

        with numpyro.plate('obs', N) as ind:
            self.likelihood.sample(G,sigma,ind,y=y)
        return
    
    def log_prob(self, G, F, sigma, x, y):
        B,N,D = y.shape

        LPW = self.wp.log_prob(x,F.transpose(1,2,0))
        LPG = self.gp.log_prob(x,G.T)

        with numpyro.plate('obs', N) as ind:
            LPL = self.likelihood.log_prob(y,G,sigma,ind)

        return LPW.sum() + LPG.sum() + LPL.sum()
    
    def update_params(self, posterior):
        params = [k for k in posterior.keys() if 'auto' not in k]
        for p in params:
            if hasattr(self.wp,p): exec('self.wp.'+p+'=posterior[\''+p+'\']')
            if hasattr(self.gp,p): exec('self.gp.'+p+'=posterior[\''+p+'\']')
            if hasattr(self.likelihood,p): exec('self.likelihood.'+p+'=posterior[\''+p+'\']')

    

# %%
class NormalPrecisionConditionalLikelihood:
    def sample(self,mu,sigma,ind=None,y=None):
        Y = numpyro.sample(
            'y',dist.MultivariateNormal(mu[ind,...],precision_matrix=sigma[ind,...]),
            obs=y
        )
        return Y
    
    def log_prob(self,Y,mu,sigma,ind=None):
        LPY = dist.MultivariateNormal(mu[ind,...],precision_matrix=sigma[ind,...]).log_prob(Y)
        return LPY


# %%
class NormalGaussianWishartPosterior:
    def __init__(self, joint, posterior, x):
        self.joint = joint
        self.posterior = posterior
        self.x = x

    def mean_stat(self,fun,x,vi_samples=1,y_samples=100):
        '''returns monte carlo estimate of a function expectation
        '''

        ys = []

        for _ in range(vi_samples):
            F,G = self.posterior.sample()
            sigma = self.joint.wp.f2sigma(F)
            mu_ = self.joint.gp.posterior(self.x, G.squeeze().T, x)
            _, sigma_ = self.joint.wp.posterior(self.x, F, sigma, x)
            for _ in range(y_samples):
                y = self.joint.likelihood.sample(mu_,sigma_)
                ys.append(y[0,0])
        
        return jnp.array([fun(y) for y in ys]).mean(0)

    def mode(self,x):
        F,G = self.posterior.mode()
        sigma = self.joint.wp.f2sigma(F)

        mu_ = self.joint.gp.posterior_mode(self.x, G.squeeze().T, x)
        F_, sigma_ = self.joint.wp.posterior_mode(self.x, F, sigma, x) 

        return mu_, sigma_, F_

    def sample(self, x):
        F,G = self.posterior.sample()
        
        sigma = self.joint.wp.f2sigma(F)

        mu_ = self.joint.gp.posterior(self.x, G.squeeze().T, x)
        F_, sigma_ = self.joint.wp.posterior(self.x, F, sigma, x) 
        return mu_, sigma_, F_
    
    def log_prob(self,x,y,vi_samples=10,gp_samples=1):
        # TODO: we need to exponentiate log_prob before summing!
        '''returns monte carlo estimate of log posterior predictive
        '''
        LPL = []
        for i in range(vi_samples):
            F,G = self.posterior.sample()
            sigma = self.joint.wp.f2sigma(F)
            for j in range(gp_samples):
                mu_ = self.joint.gp.posterior(self.x, G.squeeze().T, x)
                _, sigma_ = self.joint.wp.posterior(self.x, F, sigma, x) 
                lpl = self.joint.likelihood.log_prob(y,mu_,sigma_)
                LPL.append(lpl)
        
        LPP = jax.nn.logsumexp(jnp.stack(LPL),axis=0) - jnp.log(vi_samples) - jnp.log(gp_samples)
        return LPP

