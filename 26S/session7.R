# session7.R
# Session 7: Difference-in-Differences (incl. staggered DiD)
# Two parts:
#   A. Classic 2-period DiD via TWFE on organ_donations (Kessler & Roth 2014).
#   B. Staggered rollout via Goodman-Bacon decomposition + 3 modern estimators
#      on a simulated panel.
#
# Required packages:
#   install.packages(c("causaldata", "fixest", "did", "ggplot2"))         # core
#   # Optional (script wraps each in tryCatch, will skip if missing):
#   install.packages(c("bacondecomp", "etwfe", "didimputation"))

library(causaldata)
library(fixest)

# ============================================================
# Part A. Classic 2-period DiD
# ============================================================
od <- organ_donations
od$Treated <- (od$State == "California") &
              (od$Quarter %in% c("Q32011", "Q42011", "Q12012"))

# TWFE with state-level clustering
m_twfe <- feols(Rate ~ Treated | State + Quarter,
                data = od, cluster = ~State)
cat("---- Part A: classic DiD (CA active-choice) ----\n")
print(summary(m_twfe))

# Dynamic treatment effects (event study)
# IMPORTANT: order quarters chronologically -- as.integer(factor(Quarter))
# alone uses alphabetical order ("Q12011" < "Q12012" < "Q22011" ...) which
# silently mislabels the periods even though the regression coefficients
# themselves are correct.
quarter_order <- c("Q42010","Q12011","Q22011","Q32011","Q42011","Q12012")
od$Q_num <- as.integer(factor(od$Quarter, levels = quarter_order))
m_dyn <- feols(Rate ~ i(Q_num, State == "California", ref = 3) |
                 State + Q_num,
               data = od, cluster = ~State)
cat("\n---- Event study (dynamic DiD; ref = Q22011) ----\n")
print(summary(m_dyn))
# coefplot(m_dyn)   # uncomment to visualise

# ============================================================
# Part B. Staggered rollout — simulated panel
# ============================================================
set.seed(2026)
n_units   <- 60
n_periods <- 10
# Build the panel with year varying fastest, so row order is
#   (id=1, year=1..10), (id=2, year=1..10), ...  -- same as Python's np.repeat.
# `expand.grid(id, year)` instead lets id vary fastest, which combined with
# `rep(..., each = n_periods)` would scramble treat_year across rows of the
# same id and break the "once treated, always treated" structure.
panel <- expand.grid(year = 1:n_periods, id = 1:n_units)
panel <- panel[order(panel$id, panel$year), c("id", "year")]
rownames(panel) <- NULL
panel$treat_year <- rep(sample(c(0, 4, 6, 8), n_units, replace = TRUE),
                        each = n_periods)
panel$D <- with(panel, treat_year > 0 & year >= treat_year)
panel$y <- 5 +
           0.5 * panel$year +
           panel$id %% 7 +
           2.0 * panel$D +
           rnorm(nrow(panel), sd = 1)
panel$D <- as.integer(panel$D)
cat("\n---- Part B: staggered rollout simulation ----\n")
cat(sprintf("n = %d, units = %d, periods = %d, true ATT = 2.0\n",
            nrow(panel), n_units, n_periods))

# B.0  Naive TWFE on the staggered panel (the broken baseline)
m_naive_twfe <- feols(y ~ D | id + year, data = panel, cluster = ~id)
cat("\nNaive TWFE estimate (likely biased):\n")
print(summary(m_naive_twfe))

# B.1  Goodman-Bacon decomposition  (skipped if package missing)
if (requireNamespace("bacondecomp", quietly = TRUE)) {
  library(bacondecomp)
  bb <- tryCatch(
    bacon(y ~ D, data = panel, id_var = "id", time_var = "year"),
    error = function(e) {
      cat("\n(bacondecomp::bacon failed: ", conditionMessage(e), ")\n", sep = "")
      NULL
    })
  if (!is.null(bb)) {
    cat("\nGoodman-Bacon decomposition (weights x estimates):\n"); print(bb)
  }
} else {
  cat("\n(install `bacondecomp` to run Goodman-Bacon decomposition)\n")
}

# B.2  Callaway & Sant'Anna group-time ATT
library(did)
att_gt <- att_gt(yname = "y", tname = "year", idname = "id",
                 gname = "treat_year", data = panel,
                 control_group = "nevertreated")
cat("\nCallaway-Sant'Anna group-time ATT:\n")
print(summary(att_gt))

agg <- aggte(att_gt, type = "simple", na.rm = TRUE)
cat("\nC-S overall ATT:\n")
print(summary(agg))

# B.3  Wooldridge Extended TWFE (ETWFE)  (skipped if package missing)
if (requireNamespace("etwfe", quietly = TRUE)) {
  library(etwfe)
  m_etwfe <- etwfe(fml = y ~ 1, tvar = year, gvar = treat_year, data = panel)
  cat("\nWooldridge ETWFE estimate:\n"); print(emfx(m_etwfe))
} else {
  cat("\n(install `etwfe` to run Wooldridge ETWFE)\n")
}

# B.4  Borusyak-Jaravel-Spiess imputation estimator  (skipped if missing)
if (requireNamespace("didimputation", quietly = TRUE)) {
  library(didimputation)
  m_bjs <- tryCatch(
    did_imputation(data = panel, yname = "y", gname = "treat_year",
                   tname = "year", idname = "id"),
    error = function(e) {
      cat("\n(didimputation::did_imputation failed: ",
          conditionMessage(e), ")\n", sep = "")
      NULL
    })
  if (!is.null(m_bjs)) { cat("\nBJS imputation estimator:\n"); print(m_bjs) }
} else {
  cat("\n(install `didimputation` to run BJS imputation)\n")
}
