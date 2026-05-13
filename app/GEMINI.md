# fit-ln-gamma

A high-performance Python implementation for fitting a two-component mixture of Lognormal and Gamma distributions using **JAX**.

## Project Overview
This project provides a robust, JAX-accelerated alternative to the original R proof-of-concept. It is designed to perform Maximum Likelihood Estimation (MLE) on large-scale datasets (5M+ rows) with high numerical precision and fast execution.

### Key Features
- **JAX-Powered**: Utilizes XLA compilation and auto-differentiation for exact gradients.
- **High Performance**: Fits 5 million samples in ~20 seconds using the L-BFGS-B optimizer.
- **Numerical Stability**: Implements `logsumexp` and `float64` precision to handle extreme likelihood values and match R's numerical accuracy.
- **Robust Optimization**: Features a multi-start strategy (default 20 starts) to reliably find global maxima on multi-modal likelihood surfaces.

## Project Structure
The repository is organized as a standard Python package:

```text
/home/tom/workspace/
├── pyproject.toml           # PEP 517/621 package metadata and dependencies
├── README.md                # General project overview
├── GEMINI.md                # Technical mandates and agent instructions (this file)
└── ln_gamma_fit/            # Main package directory
    ├── fit_ln_gamma.py      # Core fitting engine (LnGammaMixture class & solver)
    ├── poc_fit_ln_gamma.py  # Proof-of-concept execution script & visualization
    ├── inputs/              # Directory for source CSV data
    └── outputs/             # Directory for generated results and plots
```

## Technical Mandates

### Mathematical Constraints
- **Gamma Shape > 0**: Must be strictly enforced via `jnp.exp` in the unpacking logic.
- **Weights**: Sum to 1.0, enforced via the sigmoid transformation of a single logit parameter (if `w` is not fixed).
- **Mean Fit**: When `mean_fit=True`, a squared penalty is applied to the log-likelihood to constrain the mixture theoretical mean to the empirical mean of the data.

### Development Workflow
- **Precision**: Always use `jax_enable_x64=True` to maintain parity with the R implementation's double-precision results.
- **Initialization**: Use unbiased estimates (`ddof=1`) for standard deviation and variance in the data-driven heuristics.
- **Packaging**: The project uses `hatchling` as a build backend. Core logic must remain in `fit_ln_gamma.py` to allow direct script execution within the package folder.

## Environment & Execution
This project runs in a containerized environment. 

### Agent Instructions (Mandatory)
As an agent, you **MUST** use the host's `run-env` utility to execute commands inside the container. You should prioritize using the `Makefile` targets via `run-env` to ensure consistency.

- **Install/Sync**: `run-env make install`
- **Run POC**: `run-env make poc`
- **Clean**: `run-env make clean`

### User Instructions
Users interact directly with the application within the environment (e.g., calling `make install` or `make poc` directly). Do not include `run-env` in user-facing documentation like `README.md`.

## Testing & Validation
- **Empirical Mean**: The theoretical mean of the fitted mixture should align closely with the empirical mean (especially with `mean_fit=True`).
- **KS Statistic**: Used as the primary metric for fit quality (lower is better).
- **Benchmarking**: Periodically verify that optimization time scales linearly with dataset size.
