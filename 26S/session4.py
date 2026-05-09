# session4.py
# Session 4: Matching Methods & Propensity Scores
# End-to-end runnable example using the Dehejia-Wahba LaLonde-PSID1 sample
# (185 NSW treated + 2,490 PSID-1 observational controls). The severe
# imbalance between these two groups is what makes matching interesting --
# the original NSW experimental sample is already balanced by randomization.
# Demonstrates: PSM, nearest-neighbour matching, balance diagnostics, ATT.
#
# We use the `causalinference` package (Wasserman / Causal Inference textbook
# companion) — a more mature option than psmpy for an econ/stats course.
#
# Required packages:
#   pip install causalinference pandas numpy statsmodels matplotlib

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from causalinference import CausalModel

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "lalonde_psid.csv"
df = pd.read_csv(DATA_PATH).reset_index(drop=True)
covars = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]

# ============================================================
# 1. Naive (unmatched) comparison
# ============================================================
y_t = df.loc[df.treat == 1, "re78"]
y_c = df.loc[df.treat == 0, "re78"]
print("---- Naive difference in means ----")
print(f"  Treated mean: {y_t.mean():.1f}")
print(f"  Control mean: {y_c.mean():.1f}")
print(f"  Diff:         {y_t.mean() - y_c.mean():+.1f}")

# ============================================================
# 2. Build a CausalModel object: (Y, D, X)
# ============================================================
Y = df["re78"].values
D = df["treat"].values
X = df[covars].values
cm = CausalModel(Y, D, X)

# Pre-matching covariate summary (Raw-diff and Norm-diff per covariate).
print("\n---- Pre-matching covariate summary ----")
print(cm.summary_stats)

# ============================================================
# 3. Estimate propensity score (logistic) and visualise overlap
# ============================================================
cm.est_propensity_s()           # stepwise logistic with quadratic terms
df["pscore"] = cm.propensity["fitted"]

fig, ax = plt.subplots(figsize=(6, 3))
df.loc[df.treat == 1, "pscore"].plot.kde(ax=ax, label="Treated")
df.loc[df.treat == 0, "pscore"].plot.kde(ax=ax, label="Control")
ax.set_title("Propensity score overlap")
ax.set_xlabel("p(T=1|X)"); ax.legend()

# ============================================================
# 4. Trim units outside the common-support region
#    (cuts treated and controls with extreme propensity scores).
# ============================================================
cm.cutoff = 0.1
cm.trim()
print(f"\n---- After trimming: N = {cm.raw_data['N']} ----")

# ============================================================
# 5. Nearest-neighbour matching on propensity score (1:1)
#    `est_via_matching` returns ATE / ATT / ATC with Abadie-Imbens SEs.
# ============================================================
cm.est_via_matching(matches=1, bias_adj=True)
print("\n---- Matching estimates (Abadie-Imbens bias-adjusted) ----")
print(cm.estimates)

# ============================================================
# 6. Balance diagnostics: standardized differences before vs. after
#    matching, plus a manual Love plot.
#
# `causalinference` does not expose a public matched-index attribute, so we
# reconstruct 1-NN propensity matches with sklearn. This gives us a long-
# format matched DataFrame, which we use for post-matching SMDs.
# ============================================================
from sklearn.neighbors import NearestNeighbors  # noqa: E402

def smd(t, c):
    return (t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)

treated_idx = np.where(D == 1)[0]
control_idx = np.where(D == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(
    df.loc[control_idx, "pscore"].to_numpy().reshape(-1, 1))
_, nn_idx = nn.kneighbors(
    df.loc[treated_idx, "pscore"].to_numpy().reshape(-1, 1))
matched_ctrl_idx = control_idx[nn_idx.flatten()]
matched_idx = np.concatenate([treated_idx, matched_ctrl_idx])
matched = df.iloc[matched_idx].copy()

print("\n---- Standardized mean differences: before vs. after matching ----")
print(f"  {'covariate':<10}  {'before':>8}  {'after':>8}")
smd_before, smd_after = {}, {}
for v in covars:
    b = smd(df.loc[df.treat == 1, v],     df.loc[df.treat == 0, v])
    a = smd(matched.loc[matched.treat == 1, v],
            matched.loc[matched.treat == 0, v])
    smd_before[v], smd_after[v] = b, a
    flag = "OK" if abs(a) < 0.1 else ("WARN" if abs(a) < 0.2 else "BAD")
    print(f"  {v:<10}  {b:+8.3f}  {a:+8.3f}  [{flag}]")

# ----- Manual Love plot (matplotlib) ---------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter([abs(v) for v in smd_before.values()], covars,
           marker="x", color="#0065BD", label="Before matching")
ax.scatter([abs(v) for v in smd_after.values()],  covars,
           marker="o", color="#E37222", label="After matching")
ax.axvline(0.1, ls="--", color="grey")
ax.set_xlabel("|Standardized mean difference|")
ax.set_title("Love plot: NSW matching")
ax.legend()
fig.tight_layout()

# ============================================================
# 7. Doubly robust preview (full data, IPW-weighted regression).
#    Detailed coverage in Session 5.
# ============================================================
df["w"] = np.where(df.treat == 1, 1 / df.pscore, 1 / (1 - df.pscore))
X_full  = sm.add_constant(df[["treat"] + covars])
fit_dr  = sm.WLS(df["re78"], X_full, weights=df["w"]).fit()
print(f"\n---- Doubly robust (IPW-weighted regression) ----")
print(f"  tau = {fit_dr.params['treat']:+8.1f}  (SE = {fit_dr.bse['treat']:.1f})")
