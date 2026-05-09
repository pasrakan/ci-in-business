# session8.py
# Session 8: Instrumental Variables — 2SLS, weak-IV diagnostics, LATE
# End-to-end on Card (1995) "Using Geographic Variation in College Proximity"
# — the canonical IV teaching dataset.
#
#   Outcome Y    : lwage  (log hourly wage)
#   Treatment D  : educ   (years of schooling, endogenous)
#   Instrument Z : nearc4 (lives near a 4-year college)
#
# Required packages:
#   pip install causaldata pandas numpy statsmodels linearmodels

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from causaldata import close_college      # Card (1995) college proximity

df = close_college.load_pandas().data.dropna().reset_index(drop=True).copy()
# Cast `exper` to float64 BEFORE squaring. `causaldata.close_college` ships
# `exper` as int8; for exper >= 12 the squared value silently overflows the
# int8 range [-128, 127] (e.g. 25**2 = 625 wraps to 113), which biases every
# downstream regression. R does not have this problem because it represents
# integers as 32-bit by default.
df["exper"]  = df["exper"].astype(float)
df["exper2"] = df["exper"] ** 2
# `causaldata.close_college` is the trimmed teaching slice of Card (1995):
# it drops the regional dummies (reg661-reg668), so we use the basic controls.
controls = ["exper", "exper2", "black", "south", "smsa", "married"]

# ============================================================
# 1. Naive OLS (likely biased: ability confounds educ and wage)
# ============================================================
X_ols = sm.add_constant(df[["educ"] + controls])
m_ols = sm.OLS(df["lwage"], X_ols).fit(cov_type="HC1")
print("---- OLS (likely biased) ----")
print(f"  beta_educ = {m_ols.params['educ']:+.4f}  "
      f"(SE = {m_ols.bse['educ']:.4f})")

# ============================================================
# 2. Reduced form: does the instrument move the outcome?
# ============================================================
X_rf = sm.add_constant(df[["nearc4"] + controls])
m_rf = sm.OLS(df["lwage"], X_rf).fit(cov_type="HC1")
print("\n---- Reduced form (Y on Z) ----")
print(f"  beta_Z    = {m_rf.params['nearc4']:+.4f}  "
      f"(SE = {m_rf.bse['nearc4']:.4f})")

# ============================================================
# 3. First stage: does the instrument move the treatment?
#    Rule of thumb: F > 10 (Stock-Yogo) for a single IV.
# ============================================================
X_fs = sm.add_constant(df[["nearc4"] + controls])
m_fs = sm.OLS(df["educ"], X_fs).fit(cov_type="HC1")
print("\n---- First stage (D on Z) ----")
print(f"  pi_Z      = {m_fs.params['nearc4']:+.4f}  "
      f"(SE = {m_fs.bse['nearc4']:.4f})")
# Robust F for the single excluded instrument:
F = (m_fs.params["nearc4"] / m_fs.bse["nearc4"]) ** 2
print(f"  First-stage F (single Z) = {F:.2f}   (need > 10)")

# ============================================================
# 4. 2SLS via linearmodels.iv.IV2SLS
# ============================================================
exog  = sm.add_constant(df[controls])
endog = df[["educ"]]
instr = df[["nearc4"]]
y     = df["lwage"]

m_iv = IV2SLS(y, exog, endog, instr).fit(cov_type="robust")
print("\n---- 2SLS estimate (LATE on compliers) ----")
print(m_iv.summary)
# Useful diagnostics:
print("\nFirst-stage diagnostics:")
print(m_iv.first_stage)
# m_iv.wu_hausman()      # endogeneity test (OLS vs IV)
# m_iv.sargan            # only meaningful when overidentified

# ============================================================
# 5. Wald estimator: ratio of reduced-form to first-stage coefficients
#    (matches 2SLS exactly with one instrument and no covariates)
# ============================================================
beta_rf = sm.OLS(df["lwage"], sm.add_constant(df["nearc4"])).fit().params["nearc4"]
pi_fs   = sm.OLS(df["educ"],  sm.add_constant(df["nearc4"])).fit().params["nearc4"]
print(f"\n---- Wald estimator (no covariates): {beta_rf / pi_fs:+.3f} ----")

# ============================================================
# 6. What to report:
#   * 2SLS coefficient on `educ` -> the LATE (effect for compliers,
#     i.e. those induced into more schooling by living near a college)
#   * Robust SE / 95% CI
#   * First-stage F (relevance)
#   * Wu-Hausman p (endogeneity)
#   * If overidentified: Sargan p (partial exclusion check)
# ============================================================
