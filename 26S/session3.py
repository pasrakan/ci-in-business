# session3.py
# Session 3: Subclassification, Regression Foundations, Regression in RCTs
# Three demos:
#   A. Manual subclassification on a binary stratifier.
#   B. Regression as automated subclassification (same answer, less labour).
#   C. Why regression covariates help even in a randomised experiment
#      (variance reduction).
#
# Required packages:
#   pip install causaldata pandas numpy statsmodels

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from causaldata import nsw_mixtape

rng = np.random.default_rng(2026)
df  = nsw_mixtape.load_pandas().data

# ============================================================
# Part A. Manual subclassification on `nodegree`
# ============================================================
strata = (df.groupby("nodegree")
            .apply(lambda g: pd.Series({
                "n_t":   (g.treat == 1).sum(),
                "n_c":   (g.treat == 0).sum(),
                "ybar_t": g.loc[g.treat == 1, "re78"].mean(),
                "ybar_c": g.loc[g.treat == 0, "re78"].mean(),
            }))
            .assign(diff = lambda d: d.ybar_t - d.ybar_c,
                    n    = lambda d: d.n_t + d.n_c))

ate_manual = (strata["diff"] * strata["n"]).sum() / strata["n"].sum()
print("---- Manual subclassification on `nodegree` ----")
print(strata)
print(f"\n  Weighted ATE (manual): {ate_manual:+.1f}")

# ============================================================
# Part B. Same idea via OLS regression
# ============================================================
m_reg = smf.ols("re78 ~ treat + nodegree", data=df).fit()
print(f"\n  Regression coefficient on treat: "
      f"{m_reg.params['treat']:+.1f}  (SE = {m_reg.bse['treat']:.1f})")
# Regression weighting differs slightly from naive stratum-size weighting
# (regression weights by within-stratum variance of T) — close, not identical.

# ============================================================
# Part C. Regression in an RCT — why covariates still help
# ============================================================
n = 500
X = rng.normal(size=n)
T = rng.binomial(1, 0.5, size=n)             # truly random assignment
Y = 2 + 5 * X + 1.0 * T + rng.normal(size=n)  # true ATE = 1.0
sim = pd.DataFrame({"Y": Y, "T": T, "X": X})

m_naive = smf.ols("Y ~ T",     data=sim).fit()
m_adj   = smf.ols("Y ~ T + X", data=sim).fit()

print("\n---- RCT: covariate adjustment for variance reduction ----")
print(f"  Naive RCT estimate:  tau = {m_naive.params['T']:+.3f}  "
      f"(SE = {m_naive.bse['T']:.3f})")
print(f"  With covariate X:    tau = {m_adj.params['T']:+.3f}  "
      f"(SE = {m_adj.bse['T']:.3f})  <- tighter")
