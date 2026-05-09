# session13.R
# Session 13: Causal Discovery
# Two demos:
#   A. PC algorithm on a small simulated DAG -> recover the CPDAG.
#   B. LiNGAM is Python-native; see session13.py for the linear non-Gaussian
#      identification demo.
#
# Required packages:
#   install.packages(c("pcalg", "lingam"))                 # core
#   # pcalg pulls in BiocManager-only deps (graph, RBGL); install via:
#   #   if (!require("BiocManager")) install.packages("BiocManager")
#   #   BiocManager::install(c("graph", "RBGL"))

library(pcalg)
set.seed(2026)

# ============================================================
# Part A. PC algorithm on a known DGP
# ------------------------------------------------------------
# True graph:  X1 -> X2,  X1 -> X3,  X2 -> X4,  X3 -> X4   (a "diamond")
# We simulate Gaussian data, then run PC and inspect the recovered CPDAG.
# Markov-equivalent edges remain undirected — that is a fundamental limit
# of constraint-based discovery from observational data alone.
# ============================================================
n  <- 2000
X1 <- rnorm(n)
X2 <- 0.8 * X1 + rnorm(n)
X3 <- 0.8 * X1 + rnorm(n)
X4 <- 0.6 * X2 + 0.6 * X3 + rnorm(n)
data <- cbind(X1, X2, X3, X4)

suffStat <- list(C = cor(data), n = n)
fit <- pc(suffStat, indepTest = gaussCItest,
          alpha = 0.01, labels = colnames(data))

cat("---- PC algorithm output (CPDAG adjacency) ----\n")
print(as(fit, "amat"))
cat("\nDirected edges (and undirected, indicating Markov equivalence):\n")
print(fit@graph)

# ============================================================
# Part B. GES (score-based) and LiNGAM
# ------------------------------------------------------------
# GES is in pcalg::ges; it greedily searches the CPDAG space using a
# BIC-style score. LiNGAM (linear non-Gaussian) is best run in Python
# (causal-learn's DirectLiNGAM); see session13.py.
#
# Example for GES:
#   score <- new("GaussL0penObsScore", data)
#   ges_fit <- ges(score)
#   plot(ges_fit$essgraph)
# ============================================================
cat("\n---- See session13.py for DirectLiNGAM (linear non-Gaussian DAG ID). ----\n")
