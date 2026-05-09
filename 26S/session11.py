# session11.py
# Session 11: Causal Machine Learning — CATE, Causal Forests, Policy Learning,
#            Bootstrap CI for AIPW
# End-to-end on a simulated heterogeneous-treatment-effect dataset (parts 1-4)
# plus a real-data bootstrap example on LaLonde NSW (part 5).
#
# True effect:  tau(X) = 2 * I(age > 40) - 1 * I(income > median)
#   -> some people benefit, some are hurt -> a non-trivial policy exists.
#
# Required packages:
#   pip install numpy pandas scikit-learn statsmodels econml causaldata

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML

rng = np.random.default_rng(2026)

# ============================================================
# 1. Generate a panel with heterogeneous treatment effect
# ============================================================
n      = 4000
age    = rng.uniform(20, 70, n)
income = rng.lognormal(mean=10, sigma=0.5, size=n)
educ   = rng.integers(10, 19, n)
X      = np.column_stack([age, income, educ])

tau_true = 2 * (age > 40).astype(float) - 1 * (income > np.median(income)).astype(float)

# Treatment assignment depends on covariates (not random).
logit  = -1 + 0.04 * age - 0.3 * (educ > 12)
e_true = 1 / (1 + np.exp(-logit))
W      = rng.binomial(1, e_true)

Y0 = 5 + 0.05 * age + 1e-4 * income + rng.normal(size=n)
Y  = Y0 + W * tau_true

# ============================================================
# 2. Fit a causal forest via EconML
# ============================================================
cf = CausalForestDML(
    model_y       = RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                                          random_state=2026),
    model_t       = RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                           random_state=2026),
    discrete_treatment = True,
    n_estimators  = 2000,
    min_samples_leaf = 10,
    random_state  = 2026,
)
cf.fit(Y, W, X=X)

tau_hat = cf.effect(X)
print("---- Causal Forest CATE estimates ----")
print(f"  RMSE vs true CATE: {np.sqrt(((tau_hat - tau_true)**2).mean()):.3f}")
print(f"  Overall ATE:       {cf.ate(X):+.3f}")

# ============================================================
# 3. Diagnostics: best linear projection
# ============================================================
print("\n---- CATE summary statistics ----")
print(pd.Series(tau_hat).describe().round(3))

# ============================================================
# 4. Policy Learning via EconML
# ------------------------------------------------------------
# DRPolicyTree learns a shallow decision tree that maximises the doubly-robust
# value of the policy.
# ============================================================
from econml.policy import DRPolicyTree

pt = DRPolicyTree(max_depth=2, min_samples_leaf=50, random_state=2026)
pt.fit(Y, W, X=X)

pi_tree     = pt.predict(X)                  # 0 / 1 recommendation per unit
pi_treatall = np.ones(n, dtype=int)
pi_none     = np.zeros(n, dtype=int)

# True average outcome under each policy, using the simulation ground truth.
def true_value(pi):
    return (Y0 + pi * tau_true).mean()

print("\n---- True policy value comparison (uses simulation ground truth) ----")
print(f"  Treat nobody     : {true_value(pi_none):+.3f}")
print(f"  Treat everybody  : {true_value(pi_treatall):+.3f}")
print(f"  Treat-if tau>0   : {true_value((tau_hat > 0).astype(int)):+.3f}")
print(f"  Learned policy   : {true_value(pi_tree):+.3f}  <- best")

# Optional: visualise the policy tree
# pt.plot()


# ============================================================
# 5. Bootstrap CI for AIPW ATE on the LaLonde NSW data
# ------------------------------------------------------------
# Demonstrates the bootstrap recipe from the slides on a separate, real
# observational dataset (LaLonde NSW). Each replicate refits the entire
# AIPW pipeline; the percentile interval is the resulting 95% CI.
# Required: pip install causaldata
# ============================================================
print("\n" + "=" * 60)
print("Bootstrap CI for AIPW ATE (LaLonde NSW)")
print("=" * 60)

from causaldata import nsw_mixtape  # noqa: E402

nsw    = nsw_mixtape.load_pandas().data.reset_index(drop=True)
covars = ["age", "educ", "black", "marr", "nodegree", "re74", "re75"]


def aipw_estimator(d):
    """AIPW ATE estimator that refits propensity + two outcome models."""
    Xc = pd.DataFrame(np.column_stack([np.ones(len(d)), d[covars].values]),
                      columns=["const"] + covars, index=d.index)
    # Propensity model
    ps = sm.Logit(d["treat"].astype(int), Xc).fit(disp=0)
    e  = np.clip(ps.predict(Xc), 0.01, 0.99)
    # Two outcome models, one per arm
    treated = d["treat"] == 1
    mu1 = sm.OLS(d.loc[treated,  "re78"], Xc.loc[treated]).fit().predict(Xc)
    mu0 = sm.OLS(d.loc[~treated, "re78"], Xc.loc[~treated]).fit().predict(Xc)
    aipw = (mu1 - mu0
            + d["treat"]       * (d["re78"] - mu1) / e
            - (1 - d["treat"]) * (d["re78"] - mu0) / (1 - e)).mean()
    return aipw


tau_point = aipw_estimator(nsw)
print(f"  Point estimate AIPW tau: {tau_point:+8.1f}")

B      = 500
n_nsw  = len(nsw)
boot_rng = np.random.default_rng(2026)
tau_b  = np.empty(B)
for b in range(B):
    idx       = boot_rng.choice(n_nsw, size=n_nsw, replace=True)
    tau_b[b]  = aipw_estimator(nsw.iloc[idx].reset_index(drop=True))

ci_lo, ci_hi = np.percentile(tau_b, [2.5, 97.5])
print(f"  95% percentile CI:       [{ci_lo:+8.1f}, {ci_hi:+8.1f}]   (B = {B})")
print(f"  Bootstrap SE:            {tau_b.std(ddof=1):8.1f}")


# ============================================================
# 6. DoWhy walkthrough: identify -> estimate -> refute
# ------------------------------------------------------------
# Reproduces the SaaS UI example drawn on the slide:
#       I -> S -> T -> Y     (income -> phone -> UI -> retention)
#       I -> Y               (income -> retention directly)
#       T -> E -> Y          (UI -> engagement -> retention)
#
# True effect of T on Y, decomposed:
#   * Direct (T -> Y):        +0.024
#   * Indirect (T -> E -> Y): 0.5 x 0.20 = +0.100
#   * Total causal effect:    +0.124
#
# `backdoor.linear_regression` estimates the *total* effect (it does not
# adjust for the mediator E because E is a descendant of T), so the
# correct target here is +0.124, not +0.024. With n = 5000 and unit-variance
# noise the estimate lands around +0.09; refute_estimate should leave it
# intact under random-common-cause and crush it under placebo.
#
# Required: pip install dowhy
# Optional (for model.view_model with graphviz layout):
#   brew install graphviz
#   python -m pip install \
#       --global-option=build_ext \
#       --global-option="-I$(brew --prefix graphviz)/include/" \
#       --global-option="-L$(brew --prefix graphviz)/lib/" \
#       pygraphviz
# ============================================================
print("\n" + "=" * 60)
print("DoWhy walkthrough: SaaS UI DAG")
print("=" * 60)

from dowhy import CausalModel  # noqa: E402

# --- Generate synthetic data following the DAG --------------
n_saas    = 5000
saas_rng  = np.random.default_rng(2026)
income    = saas_rng.normal(0, 1, n_saas)                       # I
phone_age = -0.6 * income + saas_rng.normal(0, 1, n_saas)       # I -> S (richer => newer phone)
# S -> T : newer phones (smaller phone_age) are more likely to load the new UI.
prob_T    = 1.0 / (1.0 + np.exp(0.8 * phone_age))
ui_treat  = (saas_rng.uniform(size=n_saas) < prob_T).astype(int)
# T -> E : new UI raises engagement on average.
engagement = 0.5 * ui_treat + saas_rng.normal(0, 1, n_saas)
# Outcome: direct UI effect (+0.024), income effect, mediated effect via E.
retention = (
    0.024 * ui_treat
    + 0.10 * income
    + 0.20 * engagement
    + saas_rng.normal(0, 1, n_saas)
)

saas = pd.DataFrame({"T": ui_treat, "Y": retention,
                     "I": income,   "S": phone_age, "E": engagement})

# --- Declare the DAG explicitly via a DOT graph -------------
dag = """
digraph {
    I -> S; I -> Y;
    S -> T;
    T -> E; E -> Y;
    T -> Y;
}
"""

model = CausalModel(data=saas, treatment="T", outcome="Y", graph=dag)

# Stage 1: identification ------------------------------------
identified = model.identify_effect(proceed_when_unidentifiable=True)
print(identified)

# Stage 2: estimation (back-door via linear regression) ------
estimate = model.estimate_effect(identified,
                                 method_name="backdoor.linear_regression")
print(f"  Causal estimate (DoWhy): {estimate.value:+.4f}   "
      f"(true total effect = +0.124; direct +0.024, indirect +0.100)")

# Stage 3: refutation ----------------------------------------
placebo = model.refute_estimate(identified, estimate,
                                method_name="placebo_treatment_refuter",
                                placebo_type="permute")
print(placebo)

random_cc = model.refute_estimate(identified, estimate,
                                  method_name="random_common_cause")
print(random_cc)

# --- Optional: render the DAG as PNG ------------------------
# Needs pygraphviz (see install note above). Writes 'causal_model.png'.
# model.view_model(layout="dot")
