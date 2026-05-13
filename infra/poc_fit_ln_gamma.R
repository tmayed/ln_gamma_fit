# POC: Fit the dedicated lognormal-gamma mixture to CSV data

# Configuration
date_str <- "2026_03_30"
ln_weight <- 0.25
mean_fit <- FALSE

# Initialize renv
if (file.exists("../renv/activate.R")) {
  Sys.setenv(RENV_PROJECT = normalizePath(".."))
  source("../renv/activate.R")
}

source("../pkg/rfit.R")
source("../pkg/plots/cdf_plot.R")
source("../pkg/plots/pdf_plot.R")
source("../pkg/plots/diag_plot.R")

# Dedicated CDF for the fit_ln_gamma() result object.
fit_ln_gamma_cdf <- function(x, fit) {
  ln_cdf <- plnorm(
    x,
    meanlog = fit$components$lognormal$mu,
    sdlog = fit$components$lognormal$sigma
  )

  gamma_cdf <- pgamma(
    x,
    shape = fit$components$gamma$shape,
    scale = fit$components$gamma$scale
  )

  fit$weights["lognormal"] * ln_cdf +
    fit$weights["gamma"] * gamma_cdf
}

# Dedicated PDF for the fit_ln_gamma() result object.
fit_ln_gamma_pdf <- function(x, fit) {
  ln_pdf <- dlnorm(
    x,
    meanlog = fit$components$lognormal$mu,
    sdlog = fit$components$lognormal$sigma
  )

  gamma_pdf <- dgamma(
    x,
    shape = fit$components$gamma$shape,
    scale = fit$components$gamma$scale
  )

  fit$weights["lognormal"] * ln_pdf +
    fit$weights["gamma"] * gamma_pdf
}

# Prevent Rplots.pdf from being created
if (!interactive()) {
  pdf(NULL)
}

# 1. Load data
input_file <- sprintf("inputs/router_traffic_%s.csv", date_str)
if (!file.exists(input_file)) {
  input_file <- sprintf("poc/inputs/router_traffic_%s.csv", date_str)
}
if (!file.exists(input_file)) {
  stop(sprintf("Input file not found: %s", input_file))
}

input_name <- tools::file_path_sans_ext(basename(input_file))
output_dir <- file.path("outputs", paste0(input_name, "_fit_ln_gamma"))

# Create or empty the output directory
if (dir.exists(output_dir)) {
  unlink(list.files(output_dir, full.names = TRUE), recursive = TRUE)
} else {
  dir.create(output_dir, recursive = TRUE)
}

data_raw <- read.csv(input_file)$traffic
data_clean <- data_raw[!is.na(data_raw) & data_raw > 0]

# 2. Take a subset robustly
set.seed(42)
subset_size <- min(2000, length(data_clean))
sample_data <- sample(data_clean, subset_size)

cat(sprintf("Loaded %d observations from %s, using subset of %d\n", 
            length(data_clean), input_file, subset_size))

# 3. Fit the dedicated lognormal-gamma mixture
cat("\nFitting lognormal-gamma mixture using fit_ln_gamma()...\n")
best_fit <- fit_ln_gamma(sample_data, w=ln_weight, mean_fit = mean_fit)
best_name <- best_fit$distribution

# 4. Display fit summary and collect results
cat("\n=== Fit Summary ===\n")
cat(sprintf("Distribution: %s\n", best_name))
cat(sprintf("Log-Likelihood: %.4f\n", best_fit$log_likelihood))
cat(sprintf("AIC: %.4f\n", AIC(best_fit)))
cat(sprintf("BIC: %.4f\n", BIC(best_fit)))
cat(sprintf("Convergence: %d\n", best_fit$convergence))

# Calculate Tracking Metrics for Pipeline Monitoring
norm_loglik <- best_fit$log_likelihood / length(sample_data)

# Calculate KS Statistic (Max distance between empirical and theoretical CDF)
# We use a sorted version of the data for the empirical CDF comparison
sorted_data <- sort(sample_data)
theoretical_cdf <- fit_ln_gamma_cdf(sorted_data, best_fit)
empirical_cdf <- seq_along(sorted_data) / length(sorted_data)
ks_stat <- max(abs(theoretical_cdf - empirical_cdf))

cat(sprintf("Normalized Log-Likelihood: %.6f\n", norm_loglik))
cat(sprintf("KS Statistic: %.6f\n", ks_stat))

fit_results_df <- data.frame(
  distribution = character(),
  param = character(),
  value = numeric(),
  stringsAsFactors = FALSE
)

# Add Tracking Metrics to output
fit_results_df <- rbind(fit_results_df, data.frame(distribution = "metrics", param = "bic", value = BIC(best_fit)))
fit_results_df <- rbind(fit_results_df, data.frame(distribution = "metrics", param = "norm_loglik", value = norm_loglik))
fit_results_df <- rbind(fit_results_df, data.frame(distribution = "metrics", param = "ks_stat", value = ks_stat))
fit_results_df <- rbind(fit_results_df, data.frame(distribution = "metrics", param = "sample_size", value = length(sample_data)))

cat("\nComponents:\n")
component_means <- numeric(2)
names(component_means) <- c("lognormal", "gamma")

# Lognormal component
cat("Component 1 (lognormal):\n")
cat(sprintf("  Weight: %.4f\n", best_fit$weights["lognormal"]))
fit_results_df <- rbind(fit_results_df, data.frame(distribution = "lognormal", param = "weight", value = best_fit$weights["lognormal"]))
for (p in names(best_fit$components$lognormal)) {
  cat(sprintf("  %s: %.4f\n", p, best_fit$components$lognormal[[p]]))
  fit_results_df <- rbind(fit_results_df, data.frame(distribution = "lognormal", param = p, value = best_fit$components$lognormal[[p]]))
}
comp_mean_ln <- lognormal_2p_mean(best_fit$components$lognormal)
component_means["lognormal"] <- comp_mean_ln
cat(sprintf("  Component Mean: %.4f\n", comp_mean_ln))

# Calculate Lognormal exponent term (mu + sigma^2/2)
ln_exp_term <- best_fit$components$lognormal$mu + (best_fit$components$lognormal$sigma^2) / 2
cat(sprintf("  Log-Mean (mu + sigma^2/2): %.4f\n", ln_exp_term))

fit_results_df <- rbind(fit_results_df, data.frame(distribution = "lognormal", param = "component_mean", value = comp_mean_ln))
fit_results_df <- rbind(fit_results_df, data.frame(distribution = "lognormal", param = "log_mean_term", value = ln_exp_term))

# Gamma component
cat("\nComponent 2 (gamma):\n")
cat(sprintf("  Weight: %.4f\n", best_fit$weights["gamma"]))
fit_results_df <- rbind(fit_results_df, data.frame(distribution = "gamma", param = "weight", value = best_fit$weights["gamma"]))
for (p in names(best_fit$components$gamma)) {
  cat(sprintf("  %s: %.4f\n", p, best_fit$components$gamma[[p]]))
  fit_results_df <- rbind(fit_results_df, data.frame(distribution = "gamma", param = p, value = best_fit$components$gamma[[p]]))
}
comp_mean_g <- gamma_2p_mean(best_fit$components$gamma)
component_means["gamma"] <- comp_mean_g
cat(sprintf("  Component Mean: %.4f\n", comp_mean_g))

# Calculate Gamma exponent term ln(shape * scale)
gamma_exp_term <- log(best_fit$components$gamma$shape * best_fit$components$gamma$scale)
cat(sprintf("  Log-Mean ln(shape*scale): %.4f\n", gamma_exp_term))

fit_results_df <- rbind(fit_results_df, data.frame(distribution = "gamma", param = "component_mean", value = comp_mean_g))
fit_results_df <- rbind(fit_results_df, data.frame(distribution = "gamma", param = "log_mean_term", value = gamma_exp_term))

# Mean comparison
mixture_theoretical_mean <- sum(best_fit$weights * component_means)
empirical_mean <- mean(sample_data)

cat("\n=== Mean Comparison ===\n")
cat(sprintf("Mixture Theoretical Mean: %.4f\n", mixture_theoretical_mean))
cat(sprintf("Empirical Mean:          %.4f\n", empirical_mean))

# Add means to the results dataframe
fit_results_df <- rbind(fit_results_df, data.frame(
  distribution = "mixture", param = "theoretical_mean", value = mixture_theoretical_mean
))
fit_results_df <- rbind(fit_results_df, data.frame(
  distribution = "mixture", param = "empirical_mean", value = empirical_mean
))

# Save results to CSV
csv_file <- file.path(output_dir, "results.csv")
write.csv(fit_results_df, csv_file, row.names = FALSE)
cat(sprintf("\nResults exported to: %s\n", csv_file))

# 7. Create Plots
cat("\nGenerating plots...\n")

plot_cdf_comparison(
  sample_data = sample_data,
  fit = best_fit,
  dist_cdf = fit_ln_gamma_cdf,
  output_dir = output_dir,
  output_file = "cdf",
  title = "Lognormal-Gamma Mixture Fit",
  x_label = "Traffic"
)

plot_cdf_comparison(
  sample_data = sample_data,
  fit = best_fit,
  dist_cdf = fit_ln_gamma_cdf,
  output_dir = output_dir,
  output_file = "cdf_log",
  title = "Lognormal-Gamma Mixture Fit (Log Scale)",
  x_label = "Traffic",
  log_x = TRUE
)

plot_pdf_comparison(
  sample_data = sample_data,
  fit = best_fit,
  dist_pdf = fit_ln_gamma_pdf,
  output_dir = output_dir,
  output_file = "pdf",
  title = "Lognormal-Gamma Mixture PDF Fit",
  x_label = "Traffic",
  empirical_mean = empirical_mean,
  fitted_mean = mixture_theoretical_mean
)

# 8. Create Diagnostic Plots
cat("\nGenerating diagnostic plots...\n")

# Bridge the specialized fit object to be compatible with plot_diag
diag_fit <- best_fit
diag_fit$distribution <- "mixture"
diag_fit$dist_names <- c("lognormal_2p", "gamma_2p")
diag_fit$components <- list(
  best_fit$components$lognormal,
  best_fit$components$gamma
)
diag_fit$weights <- as.numeric(best_fit$weights)

tryCatch({
  plot_diag(
    sample_data = sample_data,
    fit = diag_fit,
    output_dir = output_dir,
    output_file = "diag"
  )
}, error = function(e) {
  cat(sprintf("Warning: Diagnostic plots failed: %s\n", e$message))
})

cat(sprintf("\nProcessing complete. All outputs saved to: %s/\n", output_dir))
