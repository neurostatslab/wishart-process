# Estimating Noise Correlations in Neural Populations with Wishart Processes

![Estimating Noise Correlations in Neural Populations with Wishart Processes](https://github.com/neurostatslab/wishart-process/assets/5959554/7a1c2a46-1f2c-4107-ab00-906d9344332e)

The signaling capacity of a neural population depends on the scale and orientation of its covariance across trials. Estimating this ''noise'' covariance is challenging---a recording of $N$ neurons has on the order of $N^2$ unknown parameters---and is thought to require a large number of stereotyped trials.

This code package introduces a generative model based on Gaussian and Wishart processes and develops a mean field variational inference procedure for inferring neural means and covariances given a dataset consisting of condition variables and their corresponding multi-trial neural firing rates. 

See **[our paper](https://openreview.net/forum?id=3ucmcMzCXD)** for further details:

```
@inproceedings{
      nejatbakhsh2023estimating,
      title={Estimating Noise Correlations Across Continuous Conditions With Wishart Processes},
      author={Amin Nejatbakhsh and Isabel Garon and Alex H Williams},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
      url={https://openreview.net/forum?id=3ucmcMzCXD}
}
```

**Note:** This research code remains a work-in-progress to some extent. It could use more documentation and examples. Please use at your own risk and reach out to us (anejatbakhsh@flatironinstitute.org) if you have questions.

## A short and preliminary guide

### Installation Instructions

1. Download and install [**anaconda**](https://docs.anaconda.com/anaconda/install/index.html)
2. Create a **virtual environment** using anaconda and activate it

```
conda create -n jaxenv
conda activate jaxenv
```

3. Install [**JAX**](https://github.com/google/jax) package

4. Install other requirements (matplotlib, scipy, sklearn, numpyro)

5. Run either using the demo file or the run script via the following commands

```
python run.py -c configs/GPWP.yaml -o ../results/
```


Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.

### Generative model, and sampling from it

We start by creating an instance of our prior and likelihood models.

```python
# Given
# -----
# N : integer, number of neurons.
# K : integer, number of trials.
# C : integer, number of stimulus conditions.
# seed : integer, random seed for reproducibility.
# sigma_m : float, prior kernel smoothness.

# Create instances of prior and likelihood distributions and generate some synthetic data.

import jax
from numpyro import optim

import models
import inference
import jax.numpy as jnp
import numpyro

x = jnp.linspace(-1, 1, C) # condition space

# RBF kernel
kernel_rbf = lambda x, y: 1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_m**2))

# Prior models
gp = models.GaussianProcess(kernel=kernel_rbf,N=N) # N is the number of neurons
wp = models.WishartLRDProcess(kernel=kernel_rbf,P=2,V=jnp.eye(N))

# Likelihood model
likelihood = models.NormalConditionalLikelihood(N)

# Sample from the generative model and create synthetic dataset
with numpyro.handlers.seed(rng_seed=seed):
    mu = gp.sample(x)
    sigma = wp.sample(x)
    y = jnp.stack([
        likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(K) 
    ]) # K is the number of trials
```


Now we are ready to fit the model to data and infer posterior distributions over neural means and covariances. Then we can sample from the inferred posterior and compute their likelihoods.

```python
# Given
# -----
# x : ndarray, (num_conditions x num_variables), stimulus conditions.
# y : ndarray, (num_trials x num_conditions x num_neurons), neural firing rates across C conditions repeated for K trials.

# Infer a posterior over neural means and covariances per condition.

# Joint distribution
joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

# Mean field variational family
varfam = inference.VariationalNormal(joint.model)

# Running inference
varfam.infer(
    optim=optim.Adam(1e-1),
    x=x,y=y,
    n_iter=20000,
    key=jax.random.PRNGKey(seed)
)
joint.update_params(varfam.posterior)
```

We can sample from the inferred posterior, compute likelihoods and summary statistics, evaluate its mode, compute derivatives, and more.

```python

# Posterior distribution
posterior = models.NormalGaussianWishartPosterior(joint,varfam,x)

# Sample from the posterior
with numpyro.handlers.seed(rng_seed=seed):
    mu_hat, sigma_hat, F_hat = posterior.sample(x)

# Evaluate posterior mode
with numpyro.handlers.seed(rng_seed=seed):
    mu_hat, sigma_hat, F_hat = posterior.mode(x)

# Evaluate the function derivative of the posterior mode 
with numpyro.handlers.seed(rng_seed=seed):
    mu_prime, sigma_prime = posterior.derivative(x)

# For the Poisson model, compute summary statistics (such as mean firing rate)
with numpyro.handlers.seed(rng_seed=seed):
    mu_hat = posterior.mean_stat(lambda x: x, x)
```


Since we use GP and WP as underlying models it's very easy to sample means and covariances in unseen test conditions:

```python
# Given
# -----
# X_test : ndarray, (num_test_conditions x num_variables), test data from first network.

# Interpolate covariances in unseen test conditions
with numpyro.handlers.seed(rng_seed=seed):
    mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(x_test)
```

