# Estimating Noise Correlations in Neural Populations with Wishart Processes: Installation Instructions

1. Download and install [**anaconda**](https://docs.anaconda.com/anaconda/install/index.html)
2. Create a **virtual environment** using anaconda and activate it

```
conda create -n jaxenv
conda activate jaxenv
```

3. Install [**JAX**](https://github.com/google/jax) package

4. Install other requirements (matplotlib, scipy, sklearn, numpyro)

- Run demo.py

```
python demo.py
```

- Run using config files

```
python run.py -c configs/GPWP.yaml -o ../results/
```