# session8.R
# Session 8: Instrumental Variables — 2SLS, weak-IV diagnostics, LATE
# End-to-end on Card (1995) "Using Geographic Variation in College Proximity"
# — the canonical IV teaching dataset.
#
#   Outcome Y    : lwage  (log hourly wage)
#   Treatment D  : educ   (years of schooling, endogenous)
#   Instrument Z : nearc4 (lives near a 4-year college)
#
# Required packages:
#   install.packages(c("causaldata", "AER"))   # AER provides ivreg() directly

library(causaldata)
library(AER)

# `causaldata::close_college` is the canonical Card (1995) college-proximity
# teaching dataset. The trimmed teaching slice drops the regional dummies
# (reg661-reg668), so we use the basic controls only.
card <- na.omit(close_college)
controls <- "exper + I(exper^2) + black + south + smsa + married"

# ============================================================
# 1. Naive OLS (likely biased: ability confounds educ and wage)
# ============================================================
f_ols <- as.formula(paste("lwage ~ educ +", controls))
m_ols <- lm(f_ols, data = card)
cat("---- OLS (likely biased) ----\n")
print(coef(summary(m_ols))["educ", ])

# ============================================================
# 2. Reduced form: does the instrument move the outcome?
# ============================================================
f_rf  <- as.formula(paste("lwage ~ nearc4 +", controls))
m_rf  <- lm(f_rf, data = card)
cat("\n---- Reduced form (Y on Z) ----\n")
print(coef(summary(m_rf))["nearc4", ])

# ============================================================
# 3. First stage: does the instrument move the treatment?
#    Rule of thumb: F > 10 (Stock-Yogo) for a single IV.
# ============================================================
f_fs  <- as.formula(paste("educ ~ nearc4 +", controls))
m_fs  <- lm(f_fs, data = card)
cat("\n---- First stage (D on Z) ----\n")
print(coef(summary(m_fs))["nearc4", ])
cat(sprintf("  First-stage F-stat: %.2f  (need > 10)\n",
            summary(m_fs)$fstatistic[1]))

# ============================================================
# 4. 2SLS via AER::ivreg
# ============================================================
f_iv <- as.formula(paste("lwage ~ educ +", controls,
                         "|", "nearc4 +", controls))
m_iv <- ivreg(f_iv, data = card)
cat("\n---- 2SLS estimate (LATE on compliers) ----\n")
print(summary(m_iv, diagnostics = TRUE))
# Diagnostics block reports:
#  * Weak instruments  (= first-stage F): want LARGE
#  * Wu-Hausman        (OLS vs IV): tests endogeneity
#  * Sargan            (only with overidentification): partial exclusion check

# ============================================================
# 5. Wald estimator: ratio of reduced-form to first-stage coefficients
#    (matches 2SLS exactly with one instrument and no covariates;
#     useful for didactic intuition.)
# ============================================================
m_rf0 <- lm(lwage ~ nearc4, data = card)
m_fs0 <- lm(educ  ~ nearc4, data = card)
wald  <- coef(m_rf0)["nearc4"] / coef(m_fs0)["nearc4"]
cat(sprintf("\n---- Wald estimator (no covariates): %.3f ----\n", wald))

# ============================================================
# 6. Reading the output (what to report)
#   * 2SLS coefficient on `educ` -> the LATE (effect for compliers,
#     i.e. those induced into more schooling by living near a college)
#   * 95% CI / robust SE
#   * First-stage F-stat (relevance check)
#   * Wu-Hausman p-value (endogeneity)
#   * If overidentified: Sargan p-value (partial exclusion check)
# ============================================================
