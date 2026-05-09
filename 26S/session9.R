# session9.R
# Session 9: Regression Discontinuity Design (Sharp + Fuzzy)
# Replicates the workflow from The Effect Ch. 25 on Manacorda et al. (2011)
# Government Transfers data (Uruguay PANES program).
#
# Required packages:
#   install.packages(c("causaldata", "rdrobust", "rddensity", "tidyverse"))

library(causaldata)
library(rdrobust)
library(rddensity)
library(tidyverse)

# ============================================================
# 1. Load data and centre the running variable at the cutoff
# ============================================================
gt <- gov_transfers
# Income score `Income_Centered` is already centred at the cutoff (=0).
# Outcome `Support` = political support indicator (0/1).

cat("---- gov_transfers data ----\n")
cat(sprintf("  N = %d   |   share treated = %.3f\n",
            nrow(gt), mean(gt$Income_Centered <= 0)))

# ============================================================
# 2. Visual diagnostic: binned-means RD plot
# ============================================================
rdplot(y = gt$Support, x = gt$Income_Centered,
       c = 0, p = 1, nbins = c(20, 20),
       title = "Binned means: political support vs. income score",
       x.label = "Income score (centered)", y.label = "Support")

# ============================================================
# 3. Sharp RD via local linear regression with optimal bandwidth
#    (rdrobust picks h via Calonico-Cattaneo-Titiunik MSE-optimal rule)
# ============================================================
cat("\n---- Sharp RD: local linear, MSE-optimal bandwidth ----\n")
m <- rdrobust(y = gt$Support, x = gt$Income_Centered, c = 0)
print(summary(m))

# Re-run with a triangular kernel and a quadratic local polynomial:
m_alt <- rdrobust(y = gt$Support, x = gt$Income_Centered,
                  c = 0, p = 2, kernel = "triangular")
cat("\n---- Sharp RD: local quadratic, triangular kernel ----\n")
print(summary(m_alt))

# ============================================================
# 4. Falsification 1: Placebo outcome (should show NO jump)
# ============================================================
# Use a baseline covariate that should not respond to the treatment.
if ("Age" %in% names(gt)) {
  cat("\n---- Placebo: Age as outcome (should be ~0) ----\n")
  placebo <- rdrobust(y = gt$Age, x = gt$Income_Centered, c = 0)
  print(summary(placebo))
}

# ============================================================
# 5. Falsification 2: McCrary density discontinuity test
#    Tests for manipulation of the running variable around the cutoff.
# ============================================================
cat("\n---- Density test (rddensity) ----\n")
dens <- rddensity(X = gt$Income_Centered, c = 0)
print(summary(dens))
# rdplotdensity(dens, X = gt$Income_Centered)   # uncomment to plot

# ============================================================
# 6. Optional: Fuzzy RD sketch
# ------------------------------------------------------------
# Sharp RD assumes everyone below the cutoff is treated. In a Fuzzy design
# the cutoff only *changes the probability* of treatment — you instrument
# the actual treatment with the eligibility rule:
#
#   m_fuzzy <- rdrobust(y = Y, x = running_var, fuzzy = D, c = 0)
#
# Identification then targets the LATE on compliers (those who actually
# took up treatment because they crossed the cutoff). See session11.R for
# the IV machinery underneath.
# ============================================================
