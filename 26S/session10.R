# session10.R
# Session 10: DAGs, the Three Building Blocks, and Identification
# Two parts:
#   A. Simulate Chain / Fork / Collider — show what controlling does to the
#      observed correlation between X and Y in each structure.
#   B. Use dagitty to specify a DAG and let it find the adjustment set.
#
# Required packages:
#   install.packages(c("dagitty", "ggdag"))

set.seed(2026)
n <- 5000

# ============================================================
# Part A. The three building blocks
# ============================================================

# --- 1. CHAIN: X -> M -> Y ---
# Conditioning on M blocks the path, killing the X-Y correlation.
X <- rnorm(n)
M <- 0.8 * X + rnorm(n)
Y <- 0.8 * M + rnorm(n)
cat("---- Chain: X -> M -> Y ----\n")
cat(sprintf("  Marginal cor(X, Y)            = %+.3f\n", cor(X, Y)))
cat(sprintf("  Partial   cor(X, Y | M)       = %+.3f  (should be ~0)\n",
            cor(resid(lm(X ~ M)), resid(lm(Y ~ M)))))

# --- 2. FORK: X <- C -> Y (classic confounding) ---
# X and Y share a common cause C. Conditioning on C removes the spurious link.
C <- rnorm(n)
X <- 0.8 * C + rnorm(n)
Y <- 0.8 * C + rnorm(n)        # X has no direct effect on Y
cat("\n---- Fork: X <- C -> Y ----\n")
cat(sprintf("  Marginal cor(X, Y)            = %+.3f  (spurious!)\n", cor(X, Y)))
cat(sprintf("  Partial   cor(X, Y | C)       = %+.3f  (should be ~0)\n",
            cor(resid(lm(X ~ C)), resid(lm(Y ~ C)))))

# --- 3. COLLIDER: X -> S <- Y (the trap!) ---
# X and Y are independent. Conditioning on the collider S OPENS a spurious
# correlation — the famous "selection bias" or "Berkson's paradox".
X <- rnorm(n)
Y <- rnorm(n)
S <- 0.8 * X + 0.8 * Y + rnorm(n)
cat("\n---- Collider: X -> S <- Y ----\n")
cat(sprintf("  Marginal cor(X, Y)            = %+.3f  (independent)\n", cor(X, Y)))
cat(sprintf("  Partial   cor(X, Y | S)       = %+.3f  (induced!)\n",
            cor(resid(lm(X ~ S)), resid(lm(Y ~ S)))))

# Lesson: condition on chains and forks; NEVER on colliders or their descendants.

# ============================================================
# Part B. Letting dagitty find the adjustment set
# ============================================================
library(dagitty)

# Example DAG: T -> Y, with confounder L, mediator M, and collider C
g <- dagitty('
dag {
  L -> T
  L -> Y
  T -> M
  M -> Y
  T -> C
  Y -> C
}
')

cat("\n---- DAG analysis ----\n")
cat("Adjustment set(s) for total effect of T on Y (back-door):\n")
print(adjustmentSets(g, exposure = "T", outcome = "Y", type = "all"))

cat("\nMinimal sufficient adjustment set:\n")
print(adjustmentSets(g, exposure = "T", outcome = "Y", type = "minimal"))

cat("\nWhat happens if you (wrongly) also condition on the collider C?\n")
cat("  -> opens a non-causal path. dagitty will tell you the set is no longer valid:\n")
print(isAdjustmentSet(g, Z = c("L", "C"), exposure = "T", outcome = "Y"))

# Optional: visualise
# library(ggdag); ggdag(g) + theme_dag()
