# session9.py
# Session 9: Regression Discontinuity Design (Sharp + Fuzzy)
# Replicates the workflow from The Effect Ch. 25 on Manacorda et al. (2011)
# Government Transfers data (Uruguay PANES program).
#
# Required packages:
#   pip install causaldata rdrobust pandas numpy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causaldata import gov_transfers
from rdrobust import rdrobust, rdplot
from rddensity import rddensity

# ============================================================
# 1. Load data
# ============================================================
gt = gov_transfers.load_pandas().data
print(f"---- gov_transfers data ----")
print(f"  N = {len(gt)}   |   share treated = "
      f"{(gt['Income_Centered'] <= 0).mean():.3f}")

# Cast to writable numpy arrays once: rdrobust/rdplot internally do in-place
# assignments and cannot accept read-only pandas-backed arrays in some versions.
y_support = np.array(gt["Support"], copy=True)
x_income  = np.array(gt["Income_Centered"], copy=True)

# ============================================================
# 2. Visual diagnostic: binned-means RD plot
# ------------------------------------------------------------
# `rdplot` in `rdrobust 1.3.0` (current PyPI latest as of 2024-09) does
# in-place writes on internal slices that surface as a "read-only buffer"
# error on numpy >= 1.25 / pandas >= 2.0. The bug is upstream and not yet
# patched on PyPI; we already pass writable np.array copies above, but
# rdplot still trips internally, so we wrap it. The R port
# `rdrobust::rdplot()` does not have this issue.
# ============================================================
try:
    rdplot(y = y_support, x = x_income,
           c = 0, p = 1, nbins = [20, 20],
           title   = "Binned means: political support vs. income score",
           x_label = "Income score (centered)", y_label = "Support")
    plt.show()
except (ValueError, IndexError, TypeError) as exc:
    print(f"  (skipping rdplot due to package compatibility issue: {exc})")

# ============================================================
# 3. Sharp RD via local linear regression with optimal bandwidth
# ============================================================
print("\n---- Sharp RD: local linear, MSE-optimal bandwidth ----")
m = rdrobust(y = y_support, x = x_income, c = 0)
print(m)

print("\n---- Sharp RD: local quadratic, triangular kernel ----")
m_alt = rdrobust(y = y_support, x = x_income,
                 c = 0, p = 2, kernel = "triangular")
print(m_alt)

# ============================================================
# 4. Falsification 1: Placebo outcome (Age should show no jump)
# ============================================================
if "Age" in gt.columns:
    print("\n---- Placebo: Age as outcome (should be ~0) ----")
    placebo = rdrobust(y = np.array(gt["Age"], copy=True),
                       x = x_income, c = 0)
    print(placebo)

# ============================================================
# 5. Falsification 2: McCrary-style density test
# ------------------------------------------------------------
# `rddensity 2.4.6` (current PyPI latest as of 2025-01) is broken under
# pandas >= 2.0 in two distinct places:
#   * a leftover `Series._append` call (the method was removed in pandas 2.0)
#   * `X.min()[0]` style positional access on a pandas Series, which
#     started raising KeyError once Series stopped supporting integer
#     position indexing by default.
# Either error can surface depending on the data path, so we catch a
# small bouquet of exception types. The R port `rddensity::rddensity()`
# has neither issue and is the recommended fallback. Run `session9.R`
# for a working density test on the same data.
# ============================================================
print("\n---- Density test (rddensity) ----")
try:
    dens = rddensity(X = x_income, c = 0)
    print(dens)
except (AttributeError, KeyError, IndexError) as exc:
    print(f"  (skipping rddensity due to pandas compatibility issue: {exc!r})")
    print(f"  Run `session9.R` for the working density test on the same data.")

# ============================================================
# 6. Fuzzy RD sketch
# ------------------------------------------------------------
# Pass `fuzzy = D` (your actual treatment indicator) to rdrobust.
# Identification then targets the LATE on compliers — see session11.py
# for the IV machinery underneath.
#   m_fuzzy = rdrobust(y = Y, x = running_var, fuzzy = D, c = 0)
# ============================================================
