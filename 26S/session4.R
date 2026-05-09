# session4.R
# Session 4: Matching Methods & Propensity Scores
# End-to-end runnable example using the Dehejia-Wahba LaLonde-PSID1 sample
# (185 NSW treated + 2,490 PSID-1 observational controls). The severe
# imbalance between these two groups is what makes matching interesting --
# the original NSW experimental sample is already balanced by randomization.
# Demonstrates: PSM, optimal matching, balance diagnostics, inference,
# multiple controls, Mahalanobis hybrid, and DR-style inference.
#
# Required packages (install if missing):
#   install.packages(c("MatchIt", "optmatch", "cobalt", "ggplot2"))

library(MatchIt)
library(cobalt)
library(ggplot2)

# ============================================================
# 1. Load LaLonde-PSID1 data (data/ lives one level above this script)
# ============================================================
data_path <- if (file.exists("../data/lalonde_psid.csv")) {
  "../data/lalonde_psid.csv"
} else if (file.exists("data/lalonde_psid.csv")) {
  "data/lalonde_psid.csv"
} else {
  "transcripts/code/data/lalonde_psid.csv"
}
df <- read.csv(data_path)
covars <- c("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75")
form   <- as.formula(paste("treat ~", paste(covars, collapse = " + ")))

# ============================================================
# 2. Naive (unmatched) comparison
# ============================================================
cat("---- Naive difference in means (re78) ----\n")
print(t.test(re78 ~ treat, data = df))

# ============================================================
# 3. Logistic propensity score and overlap inspection
# ============================================================
ps_model    <- glm(form, data = df, family = binomial)
df$pscore   <- predict(ps_model, type = "response")
df$logit_ps <- predict(ps_model)         # logit scale: better for matching

ggplot(df, aes(x = pscore, fill = factor(treat))) +
  geom_density(alpha = 0.4) +
  labs(title = "Propensity score overlap", fill = "Treated")

# ============================================================
# 4. Optimal pair matching with a 0.2 SD caliper on logit propensity
# ============================================================
m1 <- matchit(form, data = df,
              method   = "optimal",
              distance = "glm",         # logistic propensity
              caliper  = 0.2)
cat("\n---- Optimal 1:1 PS matching ----\n")
print(summary(m1)$nn)

# ============================================================
# 5. Balance diagnostics: standardized differences + Love plot
# ============================================================
cat("\n---- Standardized differences before/after ----\n")
print(bal.tab(m1, m.threshold = 0.1))

love.plot(m1, threshold = 0.1, abs = TRUE,
          colors = c("#0065BD", "#E37222"),
          shapes = c("circle", "triangle"),
          title  = "Love plot: NSW matching")

# ============================================================
# 6. Multiple controls per treated (1:k matching)
# k=2 trades a bit of bias for ~25% variance reduction (Notes 6).
# ============================================================
m2 <- matchit(form, data = df,
              method   = "nearest",
              ratio    = 2,
              distance = "glm",
              caliper  = 0.2,
              replace  = FALSE)
cat("\n---- 1:2 nearest-neighbour matching ----\n")
print(summary(m2)$nn)

# ============================================================
# 7. Mahalanobis distance + propensity caliper hybrid
# Match locally on key prognostic vars, globally on propensity.
# ============================================================
m3 <- matchit(form, data = df,
              method       = "nearest",
              distance     = "glm",       # propensity for caliper
              mahvars      = ~ age + educ + re74 + re75,
              caliper      = 0.2)
cat("\n---- Mahalanobis + propensity caliper hybrid ----\n")
print(summary(m3)$nn)

# ============================================================
# 8. Inference: regression on the matched dataset
# ============================================================
matched <- match.data(m1)
fit <- lm(re78 ~ treat + age + educ + black + hisp + marr +
            nodegree + re74 + re75,
          data    = matched,
          weights = weights)
cat("\n---- Treatment effect on matched data ----\n")
print(coef(summary(fit))["treat", ])

# ============================================================
# 9. Doubly robust preview (full IPW + outcome regression)
#    Detailed coverage in Session 5.
# ============================================================
df$w <- ifelse(df$treat == 1, 1 / df$pscore, 1 / (1 - df$pscore))
fit_dr <- lm(re78 ~ treat + age + educ + black + hisp + marr +
               nodegree + re74 + re75,
             data    = df,
             weights = w)
cat("\n---- Doubly robust (IPW-weighted regression) ----\n")
print(coef(summary(fit_dr))["treat", ])
