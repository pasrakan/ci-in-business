# session10.py
# Session 10: DAGs, the Three Building Blocks, and Identification
# Three parts:
#   A. Simulate Chain / Fork / Collider — show what controlling does to the
#      observed correlation between X and Y in each structure.
#   B. Build a DAG with `networkx` and check d-separation with `pgmpy`.
#   C. End-to-end DoWhy pipeline on the same DAG:
#      identification -> estimation -> refutation. This is the canonical
#      Pearl-style workflow and bridges S9 (DAG) -> S10 (flowchart) -> S12
#      (sensitivity / refutation).
#
# Required packages:
#   pip install numpy pandas statsmodels networkx pgmpy dowhy

import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(2026)
n   = 5000

def partial_corr(x, y, z):
    """Correlation of (x, y) after linearly removing z (a 1-D array)."""
    z = sm.add_constant(z)
    rx = x - sm.OLS(x, z).fit().predict(z)
    ry = y - sm.OLS(y, z).fit().predict(z)
    return np.corrcoef(rx, ry)[0, 1]

# ============================================================
# Part A. The three building blocks
# ============================================================

# --- 1. CHAIN: X -> M -> Y ---
X = rng.normal(size=n)
M = 0.8 * X + rng.normal(size=n)
Y = 0.8 * M + rng.normal(size=n)
print("---- Chain: X -> M -> Y ----")
print(f"  Marginal cor(X, Y)      = {np.corrcoef(X, Y)[0,1]:+.3f}")
print(f"  Partial   cor(X, Y | M) = {partial_corr(X, Y, M):+.3f}  (should be ~0)")

# --- 2. FORK: X <- C -> Y ---
C = rng.normal(size=n)
X = 0.8 * C + rng.normal(size=n)
Y = 0.8 * C + rng.normal(size=n)        # X has no direct effect
print("\n---- Fork: X <- C -> Y ----")
print(f"  Marginal cor(X, Y)      = {np.corrcoef(X, Y)[0,1]:+.3f}  (spurious!)")
print(f"  Partial   cor(X, Y | C) = {partial_corr(X, Y, C):+.3f}  (should be ~0)")

# --- 3. COLLIDER: X -> S <- Y ---
X = rng.normal(size=n)
Y = rng.normal(size=n)
S = 0.8 * X + 0.8 * Y + rng.normal(size=n)
print("\n---- Collider: X -> S <- Y ----")
print(f"  Marginal cor(X, Y)      = {np.corrcoef(X, Y)[0,1]:+.3f}  (independent)")
print(f"  Partial   cor(X, Y | S) = {partial_corr(X, Y, S):+.3f}  (induced!)")

# Lesson: condition on chains and forks; NEVER on colliders or their descendants.

# ============================================================
# Part B. DAGs with networkx + d-separation via pgmpy
# ============================================================
# Same DAG as session9.R: T -> Y, confounder L, mediator M, collider C.
try:
    from pgmpy.base import DAG

    g = DAG([("L", "T"), ("L", "Y"),
             ("T", "M"), ("M", "Y"),
             ("T", "C"), ("Y", "C")])

    print("\n---- DAG analysis (pgmpy) ----")
    print("Minimal adjustment set for total effect of T on Y:")
    print(g.minimal_dseparator(start="T", end="Y"))

    print("\nIs {L} sufficient to d-separate T and Y (given the back-door)?")
    # We test active trails with conditioning set = {L}
    print("  active trail T-Y | {L} :",
          g.active_trail_nodes("T", observed=["L"])["T"].__contains__("Y"))

    print("\nIf we (wrongly) also condition on the collider C, we open a path:")
    print("  active trail T-Y | {L, C} :",
          g.active_trail_nodes("T", observed=["L", "C"])["T"].__contains__("Y"))
except ImportError:
    print("\n(install `pgmpy` to run the DAG analysis section)")

# ============================================================
# Part C. End-to-end DoWhy pipeline: DAG -> identify -> estimate -> refute
# ------------------------------------------------------------
# Same DAG as Part B: L confounds T->Y, M is a mediator, C is a collider.
# Simulate data so the TRUE total effect of T on Y is 1.0 and check that
# DoWhy (a) finds the right back-door adjustment set, (b) recovers the
# effect, and (c) survives standard refutations.
# ============================================================
try:
    from dowhy import CausalModel

    n = 5000
    L = rng.normal(size=n)
    T = (0.5 * L + rng.normal(size=n) > 0).astype(int)
    M = 0.8 * T + rng.normal(size=n)
    Y = 0.6 * T + 0.5 * M + 0.7 * L + rng.normal(size=n)   # true total = 0.6 + 0.5*0.8 = 1.0
    Cv = 0.4 * T + 0.4 * Y + rng.normal(size=n)            # collider; never condition on it
    df = pd.DataFrame({"L": L, "T": T, "M": M, "Y": Y, "C": Cv})

    # Hand DoWhy the DAG (DOT string). It auto-detects back-door / front-door / IV.
    # NB: DoWhy 0.14's DOT parser silently rejects *named* digraphs
    # (`digraph G {...}`) -- only the anonymous form `digraph { ... }`
    # parses. Leading whitespace before the `digraph` keyword is also
    # not allowed in some sub-paths. Stick with the form below.
    dot_graph = """
    digraph {
      L -> T; L -> Y;
      T -> M; M -> Y;
      T -> C; Y -> C;
    }
    """
    model = CausalModel(data=df, treatment="T", outcome="Y", graph=dot_graph)

    # Visualise the DAG (writes causal_model.png to the working directory).
    # model.view_model()

    # 1. Identification: which adjustment set does the DAG license?
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("\n---- DoWhy: identified estimand ----")
    print(estimand)

    # 2. Estimation: back-door linear regression on the identified set.
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    print(f"\nDoWhy point estimate of T -> Y: {estimate.value:+.3f}   "
          f"(true total effect = 1.000)")

    # 3. Refutation: three standard automated robustness checks.
    print("\n---- Refutation 1: random common cause (estimate should be stable) ----")
    print(model.refute_estimate(estimand, estimate,
                                method_name="random_common_cause"))

    print("\n---- Refutation 2: placebo treatment (estimate should ~0) ----")
    print(model.refute_estimate(estimand, estimate,
                                method_name="placebo_treatment_refuter"))

    print("\n---- Refutation 3: random data subset (estimate should be stable) ----")
    print(model.refute_estimate(estimand, estimate,
                                method_name="data_subset_refuter"))
except ImportError:
    print("\n(install `dowhy` to run the end-to-end pipeline section)")
