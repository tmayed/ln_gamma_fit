# ln-gamma-fit

A high-performance Python package for fitting a two-component mixture of **Lognormal** and **Gamma** distributions using **JAX**.

## Overview

`ln-gamma-fit` is designed for Maximum Likelihood Estimation (MLE) on large-scale datasets. By leveraging JAX's XLA compilation and auto-differentiation, it achieves significant speedups (50x-200x) over traditional R implementations, fitting millions of rows in seconds.

### Key Features
- **Fast & Scalable**: JAX-accelerated fitting capable of handling 5M+ samples.
- **Robust Optimization**: Multi-start L-BFGS-B strategy with exact gradients to avoid local optima.
- **High Precision**: Full `float64` support for numerical stability.
- **Mean Constraint**: Optional penalty to align the theoretical mixture mean with the empirical data mean.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and a `Makefile` for standard operations.

```bash
# Install the package and sync dependencies
make install
```

## Usage

### 1. As a Library
You can import the fitting function into your own scripts:

```python
from ln_gamma_fit.fit_ln_gamma import fit_ln_gamma

# Your data as a list or array
data = [1.2, 3.4, 0.5, ...]

# Perform the fit
fit = fit_ln_gamma(data, n_starts=10, mean_fit=True)

print(f"Log-Likelihood: {fit.log_likelihood}")
print(f"Weights: {fit.weights}")
print(f"Lognormal Params: {fit.components['lognormal']}")
print(f"Gamma Params: {fit.components['gamma']}")
```

### 2. Running the POC
The package includes a proof-of-concept script that handles data cleaning, fitting, and visualization (CDF/PDF plots).

```bash
# Run the POC with the default dataset
make poc
```


## Directory Structure
- `ln_gamma_fit/fit_ln_gamma.py`: The core fitting engine.
- `ln_gamma_fit/poc_fit_ln_gamma.py`: The execution and plotting script.
- `ln_gamma_fit/inputs/`: Place your source CSV files here.
- `ln_gamma_fit/outputs/`: Result CSVs and PNG plots are generated here.

## Requirements
- Python >= 3.12
- JAX & jaxlib
- SciPy
- NumPy
- Pandas
- Matplotlib
