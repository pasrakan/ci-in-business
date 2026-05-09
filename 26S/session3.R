# session3.R
# Session 3: Subclassification, Regression Foundations, Regression in RCTs
# Three demos:
#   A. Manual subclassification on a binary stratifier.
#   B. Regression as automated subclassification (same answer, less labour).
#   C. Why regression covariates help even in a randomised experiment
#      (variance reduction).
#
# Required packages:
#   install.packages(c("causaldata", "tidyverse"))

library(causaldata)
library(tidyverse)
set.seed(2026)

# ============================================================
# Part A. Manual subclassification on NSW (LaLonde job training)
# ------------------------------------------------------------
# We stratify on `nodegree` (a binary baseline covariate) and compute the
# stratum-weighted ATE. This is the simplest possible adjustment estimator.
# ============================================================
df <- nsw_mixtape

strata <- df %>%
  group_by(nodegree) %>%
  summarise(n_t = sum(treat == 1),
            n_c = sum(treat == 0),
            ybar_t = mean(re78[treat == 1]),
            ybar_c = mean(re78[treat == 0]),
            diff   = ybar_t - ybar_c,
            n_total = n_t + n_c,
            .groups = "drop")

ate_manual <- with(strata, sum(diff * n_total) / sum(n_total))
cat("---- Manual subclassification on `nodegree` ----\n")
print(strata)
cat(sprintf("\n  Weighted ATE (manual): %+.1f\n", ate_manual))

# ============================================================
# Part B. Same idea via OLS regression
#         lm(Y ~ T + nodegree)  recovers the same adjusted effect.
# ============================================================
m_reg <- lm(re78 ~ treat + nodegree, data = df)
cat(sprintf("\n  Regression coefficient on treat: %+.1f  (SE = %.1f)\n",
            coef(m_reg)["treat"], sqrt(diag(vcov(m_reg)))["treat"]))
# Note: regression weighting differs slightly from the simple stratum-size
# weighting (regression weights by within-stratum variance of T) — close,
# not identical. Both target the same identified estimand under ignorability.

# ============================================================
# Part C. Regression in an RCT — why covariates still help
# ------------------------------------------------------------
# Simulate an RCT with a strong predictor X of Y. Treatment is assigned
# randomly, so OLS (Y ~ T) is unbiased — but adding X as a control
# REDUCES the standard error sharply.
# ============================================================
n  <- 500
X  <- rnorm(n)
T  <- rbinom(n, 1, 0.5)                # truly random assignment
Y  <- 2 + 5 * X + 1.0 * T + rnorm(n)   # true ATE = 1.0

m_naive <- lm(Y ~ T)
m_adj   <- lm(Y ~ T + X)

cat("\n---- RCT: covariate adjustment for variance reduction ----\n")
cat(sprintf("  Naive RCT estimate:   tau = %+.3f  (SE = %.3f)\n",
            coef(m_naive)["T"], sqrt(diag(vcov(m_naive)))["T"]))
cat(sprintf("  With covariate X:     tau = %+.3f  (SE = %.3f)  <- tighter\n",
            coef(m_adj)["T"],   sqrt(diag(vcov(m_adj)))["T"]))
