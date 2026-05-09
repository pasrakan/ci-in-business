# session7.py
# Session 7: Difference-in-Differences (incl. staggered DiD)
# Two parts:
#   A. Classic 2-period DiD via TWFE on organ_donations (Kessler & Roth 2014).
#   B. Staggered rollout via Callaway-Sant'Anna on a simulated panel.
#
# Required packages:
#   pip install causaldata pandas numpy linearmodels differences

import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from causaldata import organ_donations

# ============================================================
# Part A. Classic 2-period DiD
# ============================================================
od = organ_donations.load_pandas().data.copy()
post_quarters  = ["Q32011", "Q42011", "Q12012"]
od["Treated"] = ((od.State == "California") &
                 (od.Quarter.isin(post_quarters))).astype(int)
# linearmodels.PanelOLS requires the time index to be numeric / date-like.
# Map ordered quarter strings to integers 1..K.
od["q_num"] = (od["Quarter"].astype("category")
                            .cat.set_categories(sorted(od["Quarter"].unique()),
                                                ordered=True)
                            .cat.codes + 1)

od_p = od.set_index(["State", "q_num"])
mod  = PanelOLS.from_formula(
    "Rate ~ Treated + EntityEffects + TimeEffects", od_p)
res  = mod.fit(cov_type="clustered", cluster_entity=True)
print("---- Part A: classic DiD (CA active-choice) ----")
print(res.summary)

# ============================================================
# Part B. Staggered rollout simulation
# ============================================================
np.random.seed(2026)
n_units, n_periods = 60, 10

panel = pd.DataFrame({
    "id":   np.repeat(np.arange(n_units), n_periods),
    "year": np.tile(np.arange(1, n_periods + 1), n_units),
})
panel["treat_year"] = np.repeat(
    np.random.choice([0, 4, 6, 8], n_units), n_periods)
panel["D"] = ((panel.treat_year > 0) &
              (panel.year >= panel.treat_year)).astype(int)
# `differences.ATTgt` requires never-treated to be encoded as NaN, not 0.
panel["treat_year_did"] = np.where(panel["treat_year"] == 0, np.nan,
                                   panel["treat_year"])
panel["y"] = (5
              + 0.5 * panel.year
              + panel.id % 7
              + 2.0 * panel.D
              + np.random.normal(0, 1, len(panel)))

print(f"\n---- Part B: staggered rollout simulation ----")
print(f"  n = {len(panel)}, units = {n_units}, "
      f"periods = {n_periods}, true ATT = 2.0")

# B.0  Naive TWFE (the broken baseline)
panel_p = panel.set_index(["id", "year"])
mod_naive = PanelOLS.from_formula(
    "y ~ D + EntityEffects + TimeEffects", panel_p)
res_naive = mod_naive.fit(cov_type="clustered", cluster_entity=True)
print(f"\nNaive TWFE estimate (likely biased):"
      f"  D = {res_naive.params['D']:+.3f} "
      f"(SE = {res_naive.std_errors['D']:.3f})")

# B.1  Callaway & Sant'Anna via the `differences` package
try:
    from differences import ATTgt
    att = ATTgt(data          = panel.set_index(["id", "year"]),
                cohort_column = "treat_year_did")
    att.fit(formula = "y", control_group = "never_treated")
    print("\nCallaway-Sant'Anna overall ATT:")
    print(att.aggregate("simple"))
    print("\nDynamic event-study aggregation:")
    print(att.aggregate("event"))
except ImportError:
    print("\n(install `differences` to run Callaway-Sant'Anna)")

# Note: The Wooldridge ETWFE and Borusyak-Jaravel-Spiess imputation
# estimators currently have no mature Python equivalents. Use R for those
# (see session7.R). Translation: Python is fine for the canonical CS estimator
# but R is the better ecosystem for staggered DiD overall.
