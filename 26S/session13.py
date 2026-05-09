# session13.py
# Session 13: Causal Discovery
# Two demos:
#   A. PC algorithm on simulated Gaussian data -> recover the CPDAG.
#   B. DirectLiNGAM on simulated linear non-Gaussian data
#      -> recover the FULL DAG (no Markov-equivalence ambiguity).
#
# Required packages:
#   pip install causal-learn numpy pandas

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.utils.cit import fisherz

rng = np.random.default_rng(2026)
n   = 2000

# ============================================================
# Part A. PC algorithm on a Gaussian DGP
# ------------------------------------------------------------
# True graph:  X1 -> X2,  X1 -> X3,  X2 -> X4,  X3 -> X4
# PC returns a CPDAG: undirected edges flag Markov-equivalent orientations
# that observational data alone cannot distinguish.
# ============================================================
X1 = rng.normal(size=n)
X2 = 0.8 * X1 + rng.normal(size=n)
X3 = 0.8 * X1 + rng.normal(size=n)
X4 = 0.6 * X2 + 0.6 * X3 + rng.normal(size=n)
data_gauss = np.column_stack([X1, X2, X3, X4])

cg = pc(data_gauss, alpha=0.01, indep_test=fisherz)
print("---- PC algorithm CPDAG (rows/cols = X1..X4) ----")
print("Adjacency matrix (1 = edge, 0 = no edge; symmetry => undirected):")
print(cg.G.graph)
# Read: edges between (X2, X4) and (X3, X4) should be present and directed
# IF there is enough info. The X1-X2-X3 'fork' often stays undirected.

# ============================================================
# Part B. DirectLiNGAM on linear NON-Gaussian noise
# ------------------------------------------------------------
# Same DAG structure, but with non-Gaussian (Laplace) noise. Under the
# LiNGAM assumption (linear + non-Gaussian), the full DAG is identifiable
# — no Markov-equivalence problem.
# ============================================================
X1 = rng.laplace(size=n)
X2 = 0.8 * X1 + rng.laplace(size=n)
X3 = 0.8 * X1 + rng.laplace(size=n)
X4 = 0.6 * X2 + 0.6 * X3 + rng.laplace(size=n)
data_ng = np.column_stack([X1, X2, X3, X4])

model = DirectLiNGAM()
model.fit(data_ng)

print("\n---- DirectLiNGAM output ----")
print("Causal order (parents come before children):", model.causal_order_)
print("Estimated adjacency matrix B (B[i, j] = effect of j -> i):")
print(np.round(model.adjacency_matrix_, 3))

# ============================================================
# Reading the output
# ------------------------------------------------------------
# * PC ouput is a CPDAG: some edges may stay undirected because the data
#   are Markov-equivalent across orientations. That is a feature, not a bug.
# * LiNGAM exploits non-Gaussianity to break that tie and return one DAG.
#   In return, you must believe in linearity AND non-Gaussianity of errors.
# * Causal discovery is a HYPOTHESIS GENERATOR. Always combine with domain
#   knowledge and follow-up experiments before acting on a discovered edge.
# ============================================================
