#' @title fit_ln_gamma
#' @description Fit a two-component mixture of lognormal and gamma
#'   distributions by maximum likelihood.
#' @param data A numeric vector of positive observations.
#' @param initial_ln Optional named list with lognormal starting values
#'   `mu` and `sigma`.
#' @param initial_gamma Optional named list with gamma starting values
#'   `shape` and `scale`.
#' @param w Optional fixed weight for the lognormal component (0 < w < 1). 
#'   If NULL, the weights are fitted.
#' @param n_starts Number of optimization starts. Defaults to 5.
#' @param maxit Maximum optimizer iterations per start. Defaults to 3000.
#' @param mean_fit Logical; if TRUE, constrains the theoretical mean to be 
#'   within 0.01 of the empirical mean.
#' @return A fitted mixture object of class `ln_gamma_mixture`.
#' @details
#' This implementation is fully self-contained. It uses a multi-start strategy
#' with a baseline initialized from data-driven heuristics. Parameters are 
#' optimized on an unconstrained internal scale. The function requires at 
#' least one start to converge (code 0) or it will stop with an error.
#'
#' @section Design Constraints & Maintenance Mandate:
#' To ensure the mathematical integrity and physical interpretability of the 
#' mixture model, the following constraints are STRICTLY enforced and MUST be 
#' preserved in all future updates:
#' \itemize{
#'   \item \bold{gamma shape > 0}: Standard Gamma distribution parameter constraint.
#' }
#' These conditions are implemented using appropriate transformations (e.g., exp)
#' in the unpacking logic. Any modification to the parameter mapping must 
#' explicitly maintain these lower bounds.
#' @export
fit_ln_gamma <- function(data,
                          initial_ln = NULL,
                          initial_gamma = NULL,
                          w = NULL,
                          n_starts = 5L,
                          maxit = 3000L,
                          mean_fit = FALSE) {
  # --- 1. Data Cleaning ---
  data <- data[is.finite(data) & !is.na(data) & data > 0]
  if (length(data) < 10L) {
    stop("Need at least 10 positive observations.")
  }

  n_starts <- as.integer(n_starts[1])
  maxit <- as.integer(maxit[1])
  if (is.na(n_starts) || n_starts < 1L) stop("`n_starts` must be an integer >= 1.")
  if (is.na(maxit) || maxit < 1L) stop("`maxit` must be an integer >= 1.")

  if (!is.null(w)) {
    if (!is.numeric(w) || w <= 0 || w >= 1) stop("`w` must be a numeric value between 0 and 1.")
  }
  
  emp_mean <- mean(data)

  # --- 2. Initializer Validation ---
  if (!is.null(initial_ln)) {
    if (!is.list(initial_ln) || is.null(initial_ln$mu) || is.null(initial_ln$sigma))
      stop("`initial_ln` must be a list with `mu` and `sigma`.")
    if (!is.finite(initial_ln$mu) || !is.finite(initial_ln$sigma) || initial_ln$sigma <= 0)
      stop("`initial_ln` parameters must be finite and `sigma > 0`.")
  }
  if (!is.null(initial_gamma)) {
    if (!is.list(initial_gamma) || is.null(initial_gamma$shape) || is.null(initial_gamma$scale))
      stop("`initial_gamma` must be a list with `shape` and `scale`.")
    if (!is.finite(initial_gamma$shape) || !is.finite(initial_gamma$scale) || initial_gamma$shape <= 0 || initial_gamma$scale <= 0)
      stop("`initial_gamma` parameters must be finite, `shape > 0` and `scale > 0`.")
  }

  # --- 3. Self-Contained Initialization ---
  data_sorted <- sort(data)
  n <- length(data)
  idx1 <- 1:floor(n/2)
  idx2 <- (floor(n/2)+1):n
  
  ln_mu <- mean(log(data_sorted[idx1]))
  ln_sigma <- stats::sd(log(data_sorted[idx1]))
  if (is.na(ln_sigma) || ln_sigma < 1e-3) ln_sigma <- 0.1
  if (!is.null(initial_ln)) {
    ln_mu <- initial_ln$mu
    ln_sigma <- initial_ln$sigma
  }

  m2 <- mean(data_sorted[idx2])
  v2 <- stats::var(data_sorted[idx2])
  if (is.na(v2) || v2 < 1e-6) v2 <- m2^2 * 0.1
  g_shape <- max(m2^2 / v2, 0.1)
  g_scale <- v2 / m2
  if (!is.null(initial_gamma)) {
    g_shape <- initial_gamma$shape
    g_scale <- initial_gamma$scale
  }

  # --- 4. Optimization Setup ---
  if (is.null(w)) {
    base_start <- c(
      0,                                # weight logit (w1 vs w2)
      ln_mu, log(ln_sigma),             # lognormal: mu, log(sigma)
      log(g_shape), log(g_scale)        # gamma: log(shape), log(scale)
    )
  } else {
    base_start <- c(
      ln_mu, log(ln_sigma),             # lognormal: mu, log(sigma)
      log(g_shape), log(g_scale)        # gamma: log(shape), log(scale)
    )
  }

  starts <- list(base_start)
  if (n_starts > 1) {
    set.seed(42)
    for (i in 2:n_starts) {
      starts[[i]] <- base_start + stats::rnorm(length(base_start), sd = 0.25)
    }
  }

  # --- 5. Run Optimizations ---
  best_fit <- NULL
  best_value <- Inf

  for (start in starts) {
    fit_try <- tryCatch(
      optim(
        par = start,
        fn = .fit_ln_gamma_neg_log_likelihood,
        data = data,
        w = w,
        mean_fit = mean_fit,
        emp_mean = emp_mean,
        method = "Nelder-Mead",
        control = list(maxit = maxit)
      ),
      error = function(e) NULL
    )

    if (!is.null(fit_try) && is.finite(fit_try$value) && fit_try$convergence == 0) {
      if (fit_try$value < best_value) {
        best_fit <- fit_try
        best_value <- fit_try$value
      }
    }
  }

  if (is.null(best_fit)) {
    stop("Optimization failed to converge for all starting values.")
  }

  # --- 6. Result Assembly ---
  final <- .fit_ln_gamma_unpack(best_fit$par, w = w)
  log_lik <- .fit_ln_gamma_mixture_log_likelihood(best_fit$par, data, w = w, penalty = FALSE)

  result <- list(
    weights = stats::setNames(final$weights, c("lognormal", "gamma")),
    components = list(
      lognormal = final$lognormal,
      gamma = final$gamma
    ),
    log_likelihood = log_lik,
    n = length(data),
    distribution = "ln_gamma_mixture",
    convergence = best_fit$convergence,
    params_internal = best_fit$par,
    fixed_w = w,
    mean_fit = mean_fit,
    empirical_mean = emp_mean
  )

  class(result) <- "ln_gamma_mixture"
  result
}

#' @export
logLik.ln_gamma_mixture <- function(object, ...) {
  # df: (1 if w fitted else 0) + 2 (LN) + 2 (Gamma)
  df <- if (is.null(object$fixed_w)) 5 else 4
  structure(object$log_likelihood, df = df, nobs = object$n, class = "logLik")
}

.fit_ln_gamma_unpack <- function(params, w = NULL) {
  if (is.null(w)) {
    # Logit weights
    w_logit <- pmax(-20, pmin(20, params[1]))
    weights <- c(exp(w_logit), 1) / (1 + exp(w_logit))
    p_start <- 2
  } else {
    weights <- c(w, 1 - w)
    p_start <- 1
  }

  # Parameters with exp() constraints
  list(
    weights = weights,
    lognormal = list(
      mu = params[p_start],
      sigma = exp(pmax(-15, pmin(15, params[p_start + 1])))
    ),
    gamma = list(
      shape = exp(pmax(-15, pmin(15, params[p_start + 2]))),
      scale = exp(pmax(-15, pmin(15, params[p_start + 3])))
    )
  )
}

.fit_ln_gamma_mixture_log_likelihood <- function(params, data, w = NULL, mean_fit = FALSE, emp_mean = NULL, penalty = TRUE) {
  fit <- .fit_ln_gamma_unpack(params, w = w)
  
  log_ln <- dlnorm(data, meanlog = fit$lognormal$mu, sdlog = fit$lognormal$sigma, log = TRUE)
  log_g  <- dgamma(data, shape = fit$gamma$shape, scale = fit$gamma$scale, log = TRUE)
  
  comp_log <- cbind(
    log(fit$weights[1]) + log_ln,
    log(fit$weights[2]) + log_g
  )
  
  row_max <- pmax(comp_log[,1], comp_log[,2])
  valid_rows <- is.finite(row_max)
  
  ll_vec <- rep(-1e20, length(data))
  if (any(valid_rows)) {
    ll_vec[valid_rows] <- row_max[valid_rows] + log(
      exp(comp_log[valid_rows, 1] - row_max[valid_rows]) + 
      exp(comp_log[valid_rows, 2] - row_max[valid_rows])
    )
  }
  
  ll <- sum(ll_vec)
  if (!is.finite(ll)) return(-1e20)
  
  if (penalty) {
    p_val <- 5e-4 * sum(params^2)
    
    if (mean_fit && !is.null(emp_mean)) {
      # Theoretical mean of LN-Gamma mixture
      theo_mean <- fit$weights[1] * exp(fit$lognormal$mu + fit$lognormal$sigma^2 / 2) +
                   fit$weights[2] * (fit$gamma$shape * fit$gamma$scale)
      
      # Penalty if difference > 0.001
      mean_diff <- abs(theo_mean - emp_mean)
      if (mean_diff > 0.001) {
        # Squared penalty for smooth optimization
        p_val <- p_val + 1e5 * (mean_diff - 0.001)^2
      }
    }
    
    return(ll - p_val)
  }
  ll
}

.fit_ln_gamma_neg_log_likelihood <- function(params, data, w = NULL, mean_fit = FALSE, emp_mean = NULL) {
  - .fit_ln_gamma_mixture_log_likelihood(params, data, w = w, mean_fit = mean_fit, emp_mean = emp_mean, penalty = TRUE)
}
