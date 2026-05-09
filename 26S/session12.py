# session12.py
# Session 12: Sensitivity Analysis for Hidden Confounding
#
# DIVERGENCES FROM session12.tex (slide deck):
#   * Part B (Rosenbaum). The slide bills the Python side as "a ~10-line
#     hand-rolled rosenbaum_upper_p that mirrors psens"; the code below
#     is more ambitious. It ports session12.R's full workflow: load
#     nsw_mixtape, fit a logistic propensity, do greedy 1:1 nearest-
#     neighbour matching on the logit-propensity scale with a 0.2-SD
#     caliper and no replacement, and run the Rosenbaum sweep on the
#     resulting pair diffs. Output is directly comparable to
#     session12.R's `rbounds::psens` table.
#   * Part B reports BOTH lower- and upper-bound p-values at each Gamma
#     (mirrors `psens`). The slide only spells out the upper bound.
#   * Part C (DoWhy). The slide's "Code: Sensitivity in Python" frame
#     uses `add_unobserved_common_cause` -- the proper Rosenbaum analog
#     in DoWhy -- but it requires hand-tuning the effect-strength
#     parameters. The code below runs the standard panel of robustness
#     refuters (placebo, random common cause, data subset) instead, which
#     are the more typical "did your estimate survive?" battery.
#   * The slide includes a binary-outcome McNemar helper
#     (`sens_mcnemar(D, Tobs, Gamma)`) demonstrated on the Hammond
#     smoking data. That example is left to session12.R for now.
#
# Three tools below:
#   A. E-value for a published risk ratio (closed-form formula).
#   B. Rosenbaum sweep on real NSW matched-pair diffs (mirrors session12.R).
#   C. DoWhy refute_estimate hooks (placebo, random common cause, subset).
#
# Required packages:
#   pip install causaldata pandas numpy scipy statsmodels scikit-learn dowhy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ============================================================
# Part A. E-value (VanderWeele & Ding, 2017)
# ------------------------------------------------------------
# E-value = the minimum association (on the risk-ratio scale) that a
# hidden confounder must have with BOTH T and Y to fully explain an
# observed RR. Closed-form formula:
#       E = RR + sqrt( RR * (RR - 1) )           if RR >= 1
#       E = 1/RR_inv + sqrt( 1/RR_inv * (1/RR_inv - 1) ) if RR < 1
# ============================================================
def e_value(rr):
    rr = max(rr, 1 / rr)
    return rr + np.sqrt(rr * (rr - 1))

print("---- E-values ----")
for rr in [1.2, 1.5, 1.8, 2.5, 3.9]:
    print(f"  RR = {rr:.1f}  ->  E-value = {e_value(rr):.2f}")
# Rule of thumb: E-value >= 2 means a confounder would need to roughly
# DOUBLE the risk on both sides to overturn the result — usually large.

# ============================================================
# Part B. Rosenbaum sweep on real NSW matched-pair diffs
# ------------------------------------------------------------
# Mirrors session12.R: load nsw_mixtape, fit logistic propensity, do
# greedy 1:1 NN matching on the logit-propensity scale with a 0.2-SD
# caliper and no replacement, take treated-minus-control re78 diffs,
# then sweep Gamma. Reports both lower- and upper-bound p-values
# (mirrors `rbounds::psens`).
#
# NN matching here is greedy in dataset order; R's MatchIt is also
# greedy by default but breaks ties slightly differently, so the matched
# pair set is similar but not identical between the two languages.
# ============================================================
from causaldata import nsw_mixtape  # noqa: E402

nsw    = nsw_mixtape.load_pandas().data.reset_index(drop=True)
covars = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]

# 1. Logistic propensity, then move to the logit scale (better for matching).
Xc      = sm.add_constant(nsw[covars])
e_hat   = sm.Logit(nsw["treat"], Xc).fit(disp=0).predict(Xc)
logit_e = np.log(e_hat / (1 - e_hat))
caliper = 0.2 * logit_e.std()

# 2. Greedy 1:1 NN matching on logit propensity, no replacement, with caliper.
treated_idx = np.where(nsw["treat"] == 1)[0]
controls    = list(np.where(nsw["treat"] == 0)[0])
pairs       = []
for t in treated_idx:
    if not controls:
        break
    dists = np.abs(logit_e.iloc[controls].to_numpy() - float(logit_e.iloc[t]))
    j = int(np.argmin(dists))
    if dists[j] <= caliper:
        pairs.append((int(t), int(controls[j])))
        controls.pop(j)

diffs = np.array([nsw.iloc[t]["re78"] - nsw.iloc[c]["re78"] for t, c in pairs])
print("\n---- Matched-pair summary (Python NN matching) ----")
print(f"  n pairs = {len(diffs)}   |   mean diff = ${diffs.mean():,.1f}   "
      f"|   median = ${np.median(diffs):,.1f}")


def rosenbaum_bounds(diffs, gamma):
    """Worst-case (upper) and best-case (lower) one-sided p-values under
    hidden bias of strength gamma. Mirrors `rbounds::psens` output."""
    s     = np.sign(diffs)
    rk    = stats.rankdata(np.abs(diffs))
    Tplus = np.sum(rk[s == 1])
    sum_r, sum_r2 = np.sum(rk), np.sum(rk ** 2)
    out = {}
    for label, p in (("upper", gamma / (1 + gamma)),    # worst case  -> larger p
                     ("lower", 1     / (1 + gamma))):   # best case   -> smaller p
        z = (Tplus - p * sum_r) / np.sqrt(p * (1 - p) * sum_r2)
        out[label] = 1 - stats.norm.cdf(z)
    return out


print("\n---- Rosenbaum sensitivity sweep ----")
print(f"  {'Gamma':>6}  {'Lower':>10}  {'Upper':>10}")
for g in np.arange(1.0, 2.01, 0.1):
    b = rosenbaum_bounds(diffs, g)
    print(f"  {g:>6.2f}  {b['lower']:>10.4f}  {b['upper']:>10.4f}")
# Breakpoint: smallest Gamma at which the Upper column crosses 0.05.

# ============================================================
# Part C. DoWhy refute_estimate hooks (random common cause, placebo,
#         data subset). These are standard automated robustness checks.
# ============================================================
try:
    from dowhy import CausalModel
    from causaldata import nsw_mixtape

    df = nsw_mixtape.load_pandas().data.copy()
    df["treat"] = df["treat"].astype(bool)
    model = CausalModel(
        data        = df,
        treatment   = "treat",
        outcome     = "re78",
        common_causes = ["age", "educ", "black", "hisp", "marr",
                         "nodegree", "re74", "re75"],
    )
    estimand = model.identify_effect()
    estimate = model.estimate_effect(
        estimand,
        method_name = "backdoor.linear_regression",
    )
    print(f"\n---- DoWhy point estimate: {estimate.value:+.1f} ----")

    print("\nRefutation 1: add a random common cause (should NOT change estimate):")
    print(model.refute_estimate(estimand, estimate, method_name="random_common_cause"))

    print("\nRefutation 2: placebo treatment (should yield ~0):")
    print(model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter"))

    print("\nRefutation 3: data subset (estimate should be stable):")
    print(model.refute_estimate(estimand, estimate, method_name="data_subset_refuter"))
except Exception as exc:
    print(f"\n(Skipping DoWhy section: {exc})")
