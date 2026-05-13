import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.stats import norm, gamma
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
import numpy as np
import time

class LnGammaMixture:
    def __init__(self, weights, ln_params, gamma_params, log_likelihood, n, convergence, params_internal, fixed_w, mean_fit, empirical_mean):
        self.weights = weights
        self.components = {
            "lognormal": ln_params,
            "gamma": gamma_params
        }
        self.log_likelihood = log_likelihood
        self.n = n
        self.distribution = "ln_gamma_mixture"
        self.convergence = convergence
        self.params_internal = params_internal
        self.fixed_w = fixed_w
        self.mean_fit = mean_fit
        self.empirical_mean = empirical_mean

    def log_lik(self):
        return self.log_likelihood

    def aic(self):
        df = 4 if self.fixed_w is not None else 5
        return 2 * df - 2 * self.log_likelihood

    def bic(self):
        df = 4 if self.fixed_w is not None else 5
        return df * jnp.log(self.n) - 2 * self.log_likelihood

    def __repr__(self):
        return f"LnGammaMixture(log_likelihood={self.log_likelihood:.4f}, convergence={self.convergence})"

def unpack_params(params, w=None):
    """Unpack unconstrained parameters into constrained ones."""
    if w is None:
        # Weight logit
        w_logit = jnp.clip(params[0], -20.0, 20.0)
        weight_ln = jax.nn.sigmoid(w_logit)
        weights = jnp.array([weight_ln, 1.0 - weight_ln])
        p_start = 1
    else:
        weights = jnp.array([w, 1.0 - w])
        p_start = 0

    ln_mu = params[p_start]
    # Match R's clip range exactly
    ln_sigma = jnp.exp(jnp.clip(params[p_start + 1], -15.0, 15.0))
    g_shape = jnp.exp(jnp.clip(params[p_start + 2], -15.0, 15.0))
    g_scale = jnp.exp(jnp.clip(params[p_start + 3], -15.0, 15.0))

    return weights, {"mu": ln_mu, "sigma": ln_sigma}, {"shape": g_shape, "scale": g_scale}

def neg_log_likelihood(params, data, w=None, mean_fit=False, emp_mean=None, penalty=True):
    """Negative log-likelihood with optional penalties."""
    weights, ln, g = unpack_params(params, w)
    
    # Lognormal log-pdf
    log_ln = norm.logpdf(jnp.log(data), loc=ln["mu"], scale=ln["sigma"]) - jnp.log(data)
    
    # Gamma log-pdf
    log_g = gamma.logpdf(data, g["shape"], scale=g["scale"])
    
    # Mixture log-likelihood
    comp_log = jnp.stack([
        jnp.log(weights[0]) + log_ln,
        jnp.log(weights[1]) + log_g
    ], axis=1)
    
    # Use logsumexp for numerical stability
    ll_vec = logsumexp(comp_log, axis=1)
    
    # Handle NaNs/Infs by replacing with large negative values, matching R's behavior
    ll_vec = jnp.where(jnp.isfinite(ll_vec), ll_vec, -1e20)
    ll = jnp.sum(ll_vec)
    
    if not penalty:
        return -ll

    # L2 Penalty on internal parameters
    p_val = 5e-4 * jnp.sum(params**2)
    
    if mean_fit and emp_mean is not None:
        # Theoretical mean
        theo_mean = weights[0] * jnp.exp(ln["mu"] + (ln["sigma"]**2) / 2.0) + \
                    weights[1] * (g["shape"] * g["scale"])
        
        mean_diff = jnp.abs(theo_mean - emp_mean)
        # Penalty if difference > 0.001
        p_val += jnp.where(mean_diff > 0.001, 1e5 * (mean_diff - 0.001)**2, 0.0)
        
    return -ll + p_val

def fit_ln_gamma(data, initial_ln=None, initial_gamma=None, w=None, n_starts=20, maxit=3000, mean_fit=False):

    """Fit a two-component mixture of lognormal and gamma distributions."""
    
    # 1. Data Cleaning
    data = jnp.array(data, dtype=jnp.float64)
    data = data[jnp.isfinite(data) & (data > 0)]
    if len(data) < 10:
        raise ValueError("Need at least 10 positive observations.")
    
    emp_mean = jnp.mean(data)
    
    # 2. Initialization Heuristics
    data_sorted = jnp.sort(data)
    n = len(data)
    idx_mid = n // 2
    data1 = data_sorted[:idx_mid]
    data2 = data_sorted[idx_mid:]
    
    # Initial Lognormal
    if initial_ln is not None:
        ln_mu = initial_ln["mu"]
        ln_sigma = initial_ln["sigma"]
    else:
        ln_mu = jnp.mean(jnp.log(data1))
        # Use ddof=1 to match R's sd()
        ln_sigma = jnp.std(jnp.log(data1), ddof=1)
        if ln_sigma < 1e-3: ln_sigma = 0.1
        
    # Initial Gamma
    if initial_gamma is not None:
        g_shape = initial_gamma["shape"]
        g_scale = initial_gamma["scale"]
    else:
        m2 = jnp.mean(data2)
        # Use ddof=1 to match R's var()
        v2 = jnp.var(data2, ddof=1)
        if v2 < 1e-6: v2 = m2**2 * 0.1
        g_shape = jnp.maximum(m2**2 / v2, 0.1)
        g_scale = v2 / m2

    # 3. Optimization Setup
    if w is None:
        base_start = jnp.array([
            0.0,                    # weight logit
            ln_mu, jnp.log(ln_sigma), # lognormal: mu, log(sigma)
            jnp.log(g_shape), jnp.log(g_scale) # gamma: log(shape), log(scale)
        ])
    else:
        base_start = jnp.array([
            ln_mu, jnp.log(ln_sigma), # lognormal: mu, log(sigma)
            jnp.log(g_shape), jnp.log(g_scale) # gamma: log(shape), log(scale)
        ])

    starts = [base_start]
    if n_starts > 1:
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (n_starts - 1, len(base_start))) * 1.0
        for i in range(n_starts - 1):
            starts.append(base_start + noise[i])

    # 4. Run Optimizations
    best_fit_res = None
    best_value = float('inf')
    
    # Pre-compile the value and grad function using a closure
    def loss_fn(p):
        return neg_log_likelihood(p, data, w, mean_fit, emp_mean, penalty=True)
    
    loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for i, start in enumerate(starts):
        try:
            def objective(p):
                val, grad = loss_grad_fn(p)
                return float(val), np.array(grad, dtype=np.float64)

            res = minimize(
                fun=objective,
                x0=np.array(start, dtype=np.float64),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': maxit}
            )
            
            if res.success and res.fun < best_value:
                best_fit_res = res
                best_value = res.fun
            elif not res.success:
                print(f"Start {i} failed: {res.message}")
        except Exception as e:
            print(f"Start {i} raised exception: {e}")
            continue

    if best_fit_res is None:
        raise RuntimeError("Optimization failed to converge for all starting values.")

    # 5. Result Assembly
    final_params = best_fit_res.x
    weights, ln, g = unpack_params(final_params, w)
    
    # Final log-likelihood (without penalty)
    final_ll = -float(neg_log_likelihood(final_params, data, w, mean_fit, emp_mean, penalty=False))

    return LnGammaMixture(
        weights={"lognormal": float(weights[0]), "gamma": float(weights[1])},
        ln_params={"mu": float(ln["mu"]), "sigma": float(ln["sigma"])},
        gamma_params={"shape": float(g["shape"]), "scale": float(g["scale"])},
        log_likelihood=final_ll,
        n=len(data),
        convergence=int(best_fit_res.status),
        params_internal=final_params.tolist(),
        fixed_w=w,
        mean_fit=mean_fit,
        empirical_mean=float(emp_mean)
    )

if __name__ == "__main__":
    # Simple test
    key = jax.random.PRNGKey(0)
    data = jnp.concatenate([
        jnp.exp(jax.random.normal(key, (1000,)) * 0.5 + 1.0),
        jax.random.gamma(key, 2.0, shape=(1000,)) * 5.0
    ])
    
    print(f"Fitting on {len(data)} samples...")
    start_time = time.time()
    fit = fit_ln_gamma(data, n_starts=3)
    end_time = time.time()
    
    print(f"Fit complete in {end_time - start_time:.2f}s")
    print(f"Log-Likelihood: {fit.log_likelihood:.4f}")
    print(f"Weights: {fit.weights}")
    print(f"Lognormal: {fit.components['lognormal']}")
    print(f"Gamma: {fit.components['gamma']}")
