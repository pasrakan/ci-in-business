# session5.R
# Session 5: IPW & Doubly Robust Methods (incl. modern weighting)
# End-to-end example on the Dehejia-Wahba LaLonde-PSID1 sample (185 NSW
# treated + 2,490 PSID-1 observational controls): compares 5 estimators side
# by side, then demos modern weighting (entropy balancing, CBPS).
#
# Required packages:
#   install.packages(c("survey", "WeightIt", "cobalt", "ebal", "CBPS"))
#   # ebal/CBPS are pulled in by WeightIt's `method = "ebal"/"cbps"`

library(survey)
library(WeightIt)
library(cobalt)

# Load LaLonde-PSID1 data (data/ lives one level above this script).
data_path <- if (file.exists("../data/lalonde_psid.csv")) {
  "../data/lalonde_psid.csv"
} else if (file.exists("data/lalonde_psid.csv")) {
  "data/lalonde_psid.csv"
} else {
  "transcripts/code/data/lalonde_psid.csv"
}
df     <- read.csv(data_path)
covars <- c("age", "educ", "black", "marr", "nodegree", "re74", "re75")
form   <- as.formula(paste("treat ~", paste(covars, collapse = " + ")))

# ============================================================
# Common-support trim (ATT-style)
# A flat e in [0.1, 0.9] cutoff is wrong here: treatment is rare
# (185 / 2,675 = 6.9 %), and NSW treated were *selected* into training, so
# their propensity is naturally high. Trimming around the marginal rate
# would symmetrically drop most treated -- exactly the units we want to
# study. The textbook ATT fix (Imbens-Rubin 2015) is asymmetric:
# keep all treated and only trim controls that have no support in the
# treated propensity distribution. We use the 5th-95th percentile of
# treated e as the window (small buffer at the tails).
# ============================================================
ps_full <- glm(form, family = binomial, data = df)
df$e0   <- predict(ps_full, type = "response")

q       <- quantile(df$e0[df$treat == 1], probs = c(0.05, 0.95))
e_lo    <- q[[1]]; e_hi <- q[[2]]
df      <- subset(df, e0 >= e_lo & e0 <= e_hi)
cat(sprintf("ATT trim:  e in [%.4f, %.4f] (5/95 pct of treated propensity)\n",
            e_lo, e_hi))
cat(sprintf("  N = %d  (T = %d, C = %d)\n\n",
            nrow(df), sum(df$treat == 1), sum(df$treat == 0)))

# ============================================================
# Estimator 1: Outcome regression only (Session 3 baseline)
# ============================================================
m_reg <- lm(re78 ~ treat + age + educ + black + marr +
              nodegree + re74 + re75, data = df)
cat("Estimator 1  (Outcome regression):\n"); print(coef(summary(m_reg))["treat", ])

# ============================================================
# Estimator 2a: IPW (raw, unstabilized)
# ============================================================
ps_model <- glm(form, family = binomial, data = df)
df$e     <- predict(ps_model, type = "response")
df$w_raw <- ifelse(df$treat == 1, 1 / df$e, 1 / (1 - df$e))

design_raw <- svydesign(ids = ~1, weights = ~w_raw, data = df)
m_ipw      <- svyglm(re78 ~ treat, design = design_raw)
cat("\nEstimator 2a (IPW raw):\n"); print(coef(summary(m_ipw))["treat", ])

# ============================================================
# Estimator 2b: IPW with weight trimming (99th percentile cap)
# ============================================================
threshold  <- quantile(df$w_raw, 0.99)
df$w_trim  <- ifelse(df$w_raw > threshold, threshold, df$w_raw)
design_trm <- svydesign(ids = ~1, weights = ~w_trim, data = df)
m_trim     <- svyglm(re78 ~ treat, design = design_trm)
cat("\nEstimator 2b (IPW trimmed):\n"); print(coef(summary(m_trim))["treat", ])

# ============================================================
# Estimator 2c: Stabilized IPW (mean weight ≈ 1)
# ============================================================
p_treat   <- mean(df$treat); p_ctrl <- 1 - p_treat
df$w_stab <- ifelse(df$treat == 1, p_treat / df$e, p_ctrl / (1 - df$e))
design_st <- svydesign(ids = ~1, weights = ~w_stab, data = df)
m_stab    <- svyglm(re78 ~ treat, design = design_st)
cat("\nEstimator 2c (IPW stabilized):\n"); print(coef(summary(m_stab))["treat", ])

# ============================================================
# Estimator 3: Doubly Robust (IPW-weighted outcome regression)
# ============================================================
m_dr <- lm(re78 ~ treat + age + educ + black + marr +
             nodegree + re74 + re75,
           data = df, weights = w_stab)
cat("\nEstimator 3  (Doubly robust):\n"); print(coef(summary(m_dr))["treat", ])

# ============================================================
# Modern weighting: IPW vs Entropy Balancing vs CBPS via WeightIt
#
# On the LaLonde-PSID1 sample, treated and control covariate distributions
# barely overlap, so the propensity model exhibits perfect separation. Some
# WeightIt methods (notably ebal, cbps) can fail to converge on this data --
# we wrap each call in tryCatch so a crash in one method does not abort the
# script, and so students can see exactly which methods break under poor
# overlap.
# ============================================================
fit_weight <- function(method) {
  tryCatch(weightit(form, data = df, method = method),
           error = function(e) { message(sprintf("  [%s] failed: %s",
                                                  method, conditionMessage(e)))
                                 NULL })
}
w_ipw  <- fit_weight("ps")
w_eb   <- fit_weight("ebal")
w_cbps <- fit_weight("cbps")

methods_named <- Filter(Negate(is.null),
                        list(IPW = w_ipw, EBal = w_eb, CBPS = w_cbps))

cat("\n---- Max |SMD| across methods ----\n")
for (nm in names(methods_named)) {
  W   <- methods_named[[nm]]
  smd <- bal.tab(W)$Balance$Diff.Adj
  cat(sprintf("  %-5s  max|SMD| = %.3f\n", nm, max(abs(smd), na.rm = TRUE)))
}

# Love plot comparing surviving methods (skip if cobalt API mismatches).
if (length(methods_named) >= 1) {
  try(
    love.plot(bal.tab(methods_named),
              threshold = 0.1, abs = TRUE,
              title = "Modern weighting comparison"),
    silent = TRUE
  )
}

# Final ATE estimate via each weighting method
cat("\n---- Treatment effect under each method ----\n")
for (nm in names(methods_named)) {
  W <- methods_named[[nm]]$weights
  m <- lm(re78 ~ treat + age + educ + black + marr +
            nodegree + re74 + re75,
          data = df, weights = W)
  cat(sprintf("  %-5s  tau = %.1f  (SE = %.1f)\n", nm,
              coef(m)["treat"],
              summary(m)$coefficients["treat", "Std. Error"]))
}
