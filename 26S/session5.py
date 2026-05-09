# session5.py
# Session 5: IPW & Doubly Robust Methods
# End-to-end on the Dehejia-Wahba LaLonde-PSID1 sample (185 NSW treated +
# 2,490 PSID-1 observational controls): 5 estimators side by side, plus AIPW.
# This is the same dataset Session 4 matched on -- here we keep all rows and
# reweight instead of discarding unmatched units.
#
# Note: Python's modern-weighting ecosystem is less consolidated than R's
# WeightIt. We demo IPW + DR in pure statsmodels; for entropy balancing /
# CBPS analogues, R is the more mature choice (see session5.R).
#
# Required packages:
#   pip install pandas numpy statsmodels

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "lalonde_psid.csv"
df = pd.read_csv(DATA_PATH).reset_index(drop=True)
covars = ["age", "educ", "black", "marr", "nodegree", "re74", "re75"]

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
ps_full = sm.Logit(df["treat"], sm.add_constant(df[covars])).fit(disp=0)
df["e0"] = ps_full.predict(sm.add_constant(df[covars]))

e_t  = df.loc[df.treat == 1, "e0"]
e_lo, e_hi = e_t.quantile(0.05), e_t.quantile(0.95)
df = df.loc[(df["e0"] >= e_lo) & (df["e0"] <= e_hi)].reset_index(drop=True)
print(f"ATT trim:  e in [{e_lo:.4f}, {e_hi:.4f}] "
      f"(5/95 pct of treated propensity)")
print(f"  N = {len(df)}  (T = {int(df.treat.sum())}, "
      f"C = {int((1 - df.treat).sum())})\n")

# ============================================================
# Estimator 1: Outcome regression only
# ============================================================
X1   = sm.add_constant(df[["treat"] + covars])
m_reg = sm.OLS(df["re78"], X1).fit()
print(f"Estimator 1  (regression):     "
      f"tau = {m_reg.params['treat']:+8.1f}  SE = {m_reg.bse['treat']:.1f}")

# ============================================================
# Estimator 2: IPW family
# ============================================================
ps_model = sm.Logit(df["treat"], sm.add_constant(df[covars])).fit(disp=0)
df["e"]     = ps_model.predict(sm.add_constant(df[covars]))
df["w_raw"] = np.where(df.treat == 1, 1 / df.e, 1 / (1 - df.e))

# 2a. Horvitz-Thompson IPW (raw weights)
tau_ipw = ((df.treat * df.re78 / df.e).mean()
           - ((1 - df.treat) * df.re78 / (1 - df.e)).mean())
print(f"Estimator 2a (IPW raw):        tau = {tau_ipw:+8.1f}")

# 2b. Trimmed IPW (cap at 99th percentile)
thresh = df.w_raw.quantile(0.99)
df["w_trim"] = np.minimum(df.w_raw, thresh)
m_trim = sm.WLS(df["re78"], sm.add_constant(df["treat"]),
                weights=df["w_trim"]).fit()
print(f"Estimator 2b (IPW trimmed):    "
      f"tau = {m_trim.params['treat']:+8.1f}  SE = {m_trim.bse['treat']:.1f}")

# 2c. Stabilized IPW (mean weight = 1)
p_t = df.treat.mean()
df["w_stab"] = np.where(df.treat == 1, p_t / df.e, (1 - p_t) / (1 - df.e))
m_stab = sm.WLS(df["re78"], sm.add_constant(df["treat"]),
                weights=df["w_stab"]).fit()
print(f"Estimator 2c (IPW stabilized): "
      f"tau = {m_stab.params['treat']:+8.1f}  SE = {m_stab.bse['treat']:.1f}")

# ============================================================
# Estimator 3: Doubly Robust (IPW-weighted outcome regression)
# ============================================================
X_full = sm.add_constant(df[["treat"] + covars])
m_dr   = sm.WLS(df["re78"], X_full, weights=df["w_stab"]).fit()
print(f"Estimator 3  (DR weighted reg):"
      f" tau = {m_dr.params['treat']:+8.1f}  SE = {m_dr.bse['treat']:.1f}")

# ============================================================
# Bonus: AIPW (formal augmented IPW) computed by hand
# ============================================================
X_cov = sm.add_constant(df[covars])
mu1 = sm.OLS(df.loc[df.treat == 1, "re78"],
             X_cov.loc[df.treat == 1]).fit().predict(X_cov)
mu0 = sm.OLS(df.loc[df.treat == 0, "re78"],
             X_cov.loc[df.treat == 0]).fit().predict(X_cov)

aipw = (mu1 - mu0
        + df.treat       * (df.re78 - mu1) / df.e
        - (1 - df.treat) * (df.re78 - mu0) / (1 - df.e)).mean()
print(f"Bonus AIPW (hand-computed):    tau = {aipw:+8.1f}")
