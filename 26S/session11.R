# session11.R
# Session 11: Causal Machine Learning — CATE, Causal Forests, Policy Learning
# End-to-end on a simulated heterogeneous-treatment-effect dataset.
#
# True effect:  tau(X) = 2 * I(age > 40) - 1 * I(income > median)
#   -> some people benefit, some are hurt -> a non-trivial policy exists.
#
# Required packages:
#   install.packages(c("grf", "policytree", "DiagrammeR",
#                      "causaldata", "boot"))    # causaldata + boot are
#                                                # used by the bootstrap
#                                                # AIPW section at the end

library(grf)
set.seed(2026)

# ============================================================
# 1. Generate a panel with heterogeneous treatment effect
# ============================================================
n      <- 4000
age    <- runif(n, 20, 70)
income <- rlnorm(n, meanlog = 10, sdlog = 0.5)
educ   <- sample(10:18, n, replace = TRUE)
X      <- cbind(age, income, educ)

# True conditional average treatment effect (CATE)
tau_true <- 2 * (age > 40) - 1 * (income > median(income))

# Treatment assignment depends on covariates (not random).
e_true   <- plogis(-1 + 0.04 * age - 0.3 * (educ > 12))
W        <- rbinom(n, 1, e_true)

# Observed outcome
Y0 <- 5 + 0.05 * age + 0.0001 * income + rnorm(n)
Y  <- Y0 + W * tau_true

# ============================================================
# 2. Fit a causal forest
# ============================================================
cf <- causal_forest(X = X, Y = Y, W = W,
                    num.trees = 2000,
                    honesty   = TRUE,
                    seed      = 2026)

# Out-of-bag CATE predictions
tau_hat <- predict(cf)$predictions
cat("---- Causal Forest CATE estimates ----\n")
cat(sprintf("  RMSE vs true CATE: %.3f\n",
            sqrt(mean((tau_hat - tau_true)^2))))

# Doubly-robust ATE
ate <- average_treatment_effect(cf, target.sample = "all")
cat(sprintf("  Overall AIPW ATE:  %+.3f  (SE = %.3f)\n", ate[1], ate[2]))

# ============================================================
# 3. Diagnostics: best linear projection + variable importance
# ============================================================
cat("\n---- Best Linear Projection of CATE on X ----\n")
print(best_linear_projection(cf, X))

cat("\n---- Variable importance (heterogeneity drivers) ----\n")
vi <- variable_importance(cf)
print(setNames(round(as.numeric(vi), 3), colnames(X)))

# ============================================================
# 4. Policy Learning via policytree
# ------------------------------------------------------------
# policytree finds a SHALLOW decision tree that maximises the doubly-robust
# value of the policy. The tree rule itself is the prescription.
# ============================================================
library(policytree)

# DR scores from the causal forest are the labels for policy learning.
dr <- double_robust_scores(cf)
pt <- policy_tree(X, dr, depth = 2)

cat("\n---- Optimal depth-2 policy tree ----\n")
print(pt)

# ============================================================
# 5. Evaluate the learned policy vs. naive baselines
# ============================================================
pi_tree     <- predict(pt, X) - 1            # 0 = don't treat, 1 = treat
pi_treatall <- rep(1, n)
pi_none     <- rep(0, n)

policy_value <- function(pi) mean(pi * dr[, 2] + (1 - pi) * dr[, 1])

cat("\n---- DR policy value comparison ----\n")
cat(sprintf("  Treat nobody     : %+.3f\n", policy_value(pi_none)))
cat(sprintf("  Treat everybody  : %+.3f\n", policy_value(pi_treatall)))
cat(sprintf("  Treat-if tau>0   : %+.3f\n", policy_value(as.integer(tau_hat > 0))))
cat(sprintf("  Learned policy   : %+.3f  <- best\n", policy_value(pi_tree)))

# Optional: visualise the policy tree
# plot(pt, leaf.labels = c("don't treat", "treat"))


# ============================================================
# 6. Bootstrap CI for AIPW ATE on the LaLonde NSW data
# ------------------------------------------------------------
# Demonstrates the bootstrap recipe from the slides on a separate, real
# observational dataset (LaLonde NSW). Each replicate refits the entire
# AIPW pipeline; the percentile interval is the resulting 95% CI.
# Required: install.packages(c("causaldata", "boot"))
# ============================================================
cat("\n", strrep("=", 60), "\n", sep = "")
cat("Bootstrap CI for AIPW ATE (LaLonde NSW)\n")
cat(strrep("=", 60), "\n", sep = "")

library(causaldata)
library(boot)

nsw    <- nsw_mixtape
covars <- c("age", "educ", "black", "marr", "nodegree", "re74", "re75")

aipw_estimator <- function(d, idx) {
  d <- d[idx, ]
  # Propensity model
  ps   <- glm(reformulate(covars, "treat"), family = binomial, data = d)
  e    <- pmin(pmax(predict(ps, type = "response"), 0.01), 0.99)
  # Two outcome models, one per arm
  fm   <- reformulate(covars, "re78")
  mu1  <- predict(lm(fm, data = d[d$treat == 1, ]), newdata = d)
  mu0  <- predict(lm(fm, data = d[d$treat == 0, ]), newdata = d)
  mean(mu1 - mu0
       + d$treat       * (d$re78 - mu1) / e
       - (1 - d$treat) * (d$re78 - mu0) / (1 - e))
}

tau_point <- aipw_estimator(nsw, seq_len(nrow(nsw)))
cat(sprintf("  Point estimate AIPW tau: %+8.1f\n", tau_point))

set.seed(2026)
boot_out <- boot(nsw, aipw_estimator, R = 500)
ci       <- quantile(boot_out$t, probs = c(0.025, 0.975))
cat(sprintf("  95%% percentile CI:       [%+8.1f, %+8.1f]   (B = 500)\n",
            ci[1], ci[2]))
cat(sprintf("  Bootstrap SE:            %8.1f\n", sd(boot_out$t)))
