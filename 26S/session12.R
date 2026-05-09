# session12.R
# Session 12: Sensitivity Analysis for Hidden Confounding
# Three tools, three perspectives:
#   A. Rosenbaum bounds on a matched study (rbounds).
#   B. E-values for a risk-ratio claim (EValue).
#   C. Stratified binary sensitivity (sensitivity2x2xk).
#
# Required packages:
#   install.packages(c("causaldata", "MatchIt", "rbounds", "EValue",
#                      "sensitivity2x2xk"))

library(causaldata)
library(MatchIt)
library(rbounds)
library(EValue)

# ============================================================
# Part A. Rosenbaum bounds on the matched NSW analysis
# ------------------------------------------------------------
# Workflow: match on covariates -> compute pair differences -> ask
# "How large would unobserved confounding (Gamma) need to be before our
# significance disappears?"
# ============================================================
df <- nsw_mixtape

# 1. Match (pair-level) on baseline covariates
m <- matchit(treat ~ age + educ + black + hisp + marr +
                    nodegree + re74 + re75,
             data = df, method = "nearest", caliper = 0.2,
             ratio = 1, replace = FALSE)
matched <- match.data(m)

# 2. Extract treated and control outcomes in matched-pair order
pair    <- split(matched, matched$subclass)
keep    <- sapply(pair, function(p) sum(p$treat == 1) == 1 &&
                                    sum(p$treat == 0) == 1)
pair    <- pair[keep]
y_treat <- sapply(pair, function(p) p$re78[p$treat == 1])
y_ctrl  <- sapply(pair, function(p) p$re78[p$treat == 0])

cat("---- Matched-pair summary ----\n")
cat(sprintf("  n pairs = %d   |   mean diff = %.1f   |   median = %.1f\n",
            length(y_treat), mean(y_treat - y_ctrl), median(y_treat - y_ctrl)))

# 3. Rosenbaum sensitivity bounds (Wilcoxon signed-rank by default)
# psens API: psens(x = treated outcomes, y = matched control outcomes, ...)
cat("\n---- Rosenbaum sensitivity bounds (psens) ----\n")
print(psens(x = y_treat, y = y_ctrl, Gamma = 2.0, GammaInc = 0.1))
# Read the table: as Gamma rises, the upper p-value bound grows.
# The smallest Gamma at which the upper bound crosses 0.05 is the
# study's "breakpoint" — interpret as the minimum hidden-confounding
# strength that could overturn the conclusion.

# ============================================================
# Part B. E-value for an observed risk ratio
# ------------------------------------------------------------
# Suppose a published RR = 1.8 (95% CI 1.3-2.5). What is the smallest
# hidden-confounder association with both T and Y that could explain it?
# ============================================================
cat("\n---- E-value for RR = 1.8 (CI: 1.3, 2.5) ----\n")
print(evalues.RR(est = 1.8, lo = 1.3, hi = 2.5))
# The "E-value" is the minimum joint association (on the RR scale) that an
# unmeasured confounder must have with BOTH T and Y to explain away the
# observed effect. The "E-value for the CI" applies to the lower bound.

# ============================================================
# Part C. Stratified 2x2xK binary sensitivity (Mantel-Haenszel)
# ------------------------------------------------------------
# Use sensitivity2x2xk when your data are K stratified 2x2 tables (e.g.,
# treatment x outcome within each subgroup).
# ============================================================
# Toy example: 3 strata, treatment effect on a binary outcome.
tabs <- array(c(40, 20, 30, 50,
                35, 25, 20, 40,
                30, 30, 25, 45),
              dim = c(2, 2, 3))
# Note: sensitivity2x2xk usage varies by version; if installed, see:
#   ?sensitivityMH ; vignette("sensitivity2x2xk")
cat("\n---- (See sensitivity2x2xk::sensitivityMH for stratified MH bounds) ----\n")
