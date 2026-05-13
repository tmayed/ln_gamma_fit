import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from fit_ln_gamma import fit_ln_gamma, unpack_params
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.stats import norm, gamma
import time

def ln_gamma_cdf(x, weights, ln_params, gamma_params):
    """Cumulative Distribution Function for the LN-Gamma mixture."""
    # Lognormal CDF: plnorm(x, mu, sigma)
    # JAX norm.cdf(log(x), mu, sigma)
    ln_cdf = norm.cdf(jnp.log(x), loc=ln_params["mu"], scale=ln_params["sigma"])
    
    # Gamma CDF: pgamma(x, shape, scale)
    # JAX gamma.cdf(x, shape, scale=scale)
    g_cdf = gamma.cdf(x, gamma_params["shape"], scale=gamma_params["scale"])
    
    return weights["lognormal"] * ln_cdf + weights["gamma"] * g_cdf

def ln_gamma_pdf(x, weights, ln_params, gamma_params):
    """Probability Density Function for the LN-Gamma mixture."""
    # Lognormal PDF: dlnorm(x, mu, sigma)
    ln_pdf = norm.pdf(jnp.log(x), loc=ln_params["mu"], scale=ln_params["sigma"]) / x
    
    # Gamma PDF: dgamma(x, shape, scale)
    g_pdf = gamma.pdf(x, gamma_params["shape"], scale=gamma_params["scale"])
    
    return weights["lognormal"] * ln_pdf + weights["gamma"] * g_pdf

def run_poc():
    # Configuration
    date_str = "2026_03_30"
    ln_weight = 0.25
    mean_fit = True
    
    # 1. Load data
    input_file = f"inputs/router_traffic_{date_str}.csv"
    if not os.path.exists(input_file):
        input_file = f"poc/inputs/router_traffic_{date_str}.csv"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}. Generating synthetic data for POC.")
        # Generate synthetic data
        np.random.seed(42)
        n_synth = 5000
        # 25% Lognormal (mu=1, sigma=0.5), 75% Gamma (shape=2, scale=5)
        data_ln = np.random.lognormal(1.0, 0.5, int(n_synth * 0.25))
        data_g = np.random.gamma(2.0, 5.0, int(n_synth * 0.75))
        data_clean = np.concatenate([data_ln, data_g])
        input_name = "synthetic_router_traffic"
    else:
        df = pd.read_csv(input_file)
        data_raw = df['traffic'].values
        data_clean = data_raw[np.isfinite(data_raw) & (data_raw > 0)]
        input_name = os.path.splitext(os.path.basename(input_file))[0]

    output_dir = f"outputs/{input_name}_fit_ln_gamma"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Take a subset
    np.random.seed(42)
    subset_size = min(2000, len(data_clean))
    sample_data = np.random.choice(data_clean, subset_size, replace=False)

    print(f"Loaded {len(data_clean)} observations, using subset of {subset_size}")

    # 3. Fit the dedicated lognormal-gamma mixture
    print("\nFitting lognormal-gamma mixture using fit_ln_gamma()...")
    start_time = time.time()
    best_fit = fit_ln_gamma(sample_data, w=ln_weight, mean_fit=mean_fit)
    elapsed = time.time() - start_time
    
    print(f"Fit completed in {elapsed:.2f}s")

    # 4. Display fit summary and collect results
    print("\n=== Fit Summary ===")
    print(f"Distribution: {best_fit.distribution}")
    print(f"Log-Likelihood: {best_fit.log_likelihood:.4f}")
    
    aic = best_fit.aic()
    bic = best_fit.bic()
    
    print(f"AIC: {aic:.4f}")
    print(f"BIC: {bic:.4f}")
    print(f"Convergence: {best_fit.convergence}")

    norm_loglik = best_fit.log_likelihood / len(sample_data)
    
    # KS Statistic
    sorted_data = np.sort(sample_data)
    theo_cdf = ln_gamma_cdf(sorted_data, best_fit.weights, best_fit.components["lognormal"], best_fit.components["gamma"])
    emp_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ks_stat = np.max(np.abs(theo_cdf - emp_cdf))

    print(f"Normalized Log-Likelihood: {norm_loglik:.6f}")
    print(f"KS Statistic: {ks_stat:.6f}")

    # 5. Results Table
    results = []
    results.append({"distribution": "metrics", "param": "bic", "value": bic})
    results.append({"distribution": "metrics", "param": "norm_loglik", "value": norm_loglik})
    results.append({"distribution": "metrics", "param": "ks_stat", "value": ks_stat})
    results.append({"distribution": "metrics", "param": "sample_size", "value": len(sample_data)})

    print("\nComponents:")
    # Lognormal
    ln = best_fit.components["lognormal"]
    w_ln = best_fit.weights["lognormal"]
    ln_mean = np.exp(ln["mu"] + (ln["sigma"]**2)/2)
    ln_exp_term = ln["mu"] + (ln["sigma"]**2)/2
    
    print(f"Component 1 (lognormal):")
    print(f"  Weight: {w_ln:.4f}")
    print(f"  mu: {ln['mu']:.4f}")
    print(f"  sigma: {ln['sigma']:.4f}")
    print(f"  Component Mean: {ln_mean:.4f}")
    
    results.append({"distribution": "lognormal", "param": "weight", "value": w_ln})
    results.append({"distribution": "lognormal", "param": "mu", "value": ln["mu"]})
    results.append({"distribution": "lognormal", "param": "sigma", "value": ln["sigma"]})
    results.append({"distribution": "lognormal", "param": "component_mean", "value": ln_mean})
    results.append({"distribution": "lognormal", "param": "log_mean_term", "value": ln_exp_term})

    # Gamma
    g = best_fit.components["gamma"]
    w_g = best_fit.weights["gamma"]
    g_mean = g["shape"] * g["scale"]
    g_exp_term = np.log(g["shape"] * g["scale"])
    
    print(f"\nComponent 2 (gamma):")
    print(f"  Weight: {w_g:.4f}")
    print(f"  shape: {g['shape']:.4f}")
    print(f"  scale: {g['scale']:.4f}")
    print(f"  Component Mean: {g_mean:.4f}")

    results.append({"distribution": "gamma", "param": "weight", "value": w_g})
    results.append({"distribution": "gamma", "param": "shape", "value": g["shape"]})
    results.append({"distribution": "gamma", "param": "scale", "value": g["scale"]})
    results.append({"distribution": "gamma", "param": "component_mean", "value": g_mean})
    results.append({"distribution": "gamma", "param": "log_mean_term", "value": g_exp_term})

    # Mean Comparison
    theo_mean = w_ln * ln_mean + w_g * g_mean
    emp_mean = np.mean(sample_data)
    print(f"\n=== Mean Comparison ===")
    print(f"Mixture Theoretical Mean: {theo_mean:.4f}")
    print(f"Empirical Mean:          {emp_mean:.4f}")

    results.append({"distribution": "mixture", "param": "theoretical_mean", "value": theo_mean})
    results.append({"distribution": "mixture", "param": "empirical_mean", "value": emp_mean})

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "results.csv"), index=False)
    print(f"\nResults exported to: {os.path.join(output_dir, 'results.csv')}")

    # 6. Plots
    print("\nGenerating plots...")
    
    # CDF Plot
    plt.figure(figsize=(10, 6))
    plt.step(sorted_data, emp_cdf, label="Empirical CDF", where="post", color="gray", alpha=0.5)
    plt.plot(sorted_data, theo_cdf, label="Fitted LN-Gamma CDF", color="blue", linewidth=2)
    plt.title(f"Lognormal-Gamma Mixture Fit: {input_name}")
    plt.xlabel("Traffic")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "cdf.png"))
    plt.close()

    # Log CDF Plot
    plt.figure(figsize=(10, 6))
    plt.step(sorted_data, emp_cdf, label="Empirical CDF", where="post", color="gray", alpha=0.5)
    plt.plot(sorted_data, theo_cdf, label="Fitted LN-Gamma CDF", color="blue", linewidth=2)
    plt.xscale("log")
    plt.title(f"Lognormal-Gamma Mixture Fit (Log Scale): {input_name}")
    plt.xlabel("Traffic (log scale)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig(os.path.join(output_dir, "cdf_log.png"))
    plt.close()

    # PDF Plot
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(sample_data, bins=50, density=True, alpha=0.3, color="gray", label="Empirical Histogram")
    x_range = np.linspace(sorted_data[0], sorted_data[-1], 500)
    pdf_vals = ln_gamma_pdf(x_range, best_fit.weights, best_fit.components["lognormal"], best_fit.components["gamma"])
    plt.plot(x_range, pdf_vals, color="blue", linewidth=2, label="Fitted LN-Gamma PDF")
    plt.axvline(emp_mean, color="red", linestyle="--", label=f"Empirical Mean ({emp_mean:.2f})")
    plt.axvline(theo_mean, color="green", linestyle=":", label=f"Fitted Mean ({theo_mean:.2f})")
    plt.title(f"Lognormal-Gamma Mixture PDF Fit: {input_name}")
    plt.xlabel("Traffic")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "pdf.png"))
    plt.close()

    print(f"\nProcessing complete. All outputs saved to: {output_dir}/")

if __name__ == "__main__":
    run_poc()
