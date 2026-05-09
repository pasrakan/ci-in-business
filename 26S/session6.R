# session6.R
# Session 6: Panel Data, Fixed Effects & Random Effects
# Uses Gapminder via causaldata: log GDP per capita -> life expectancy.
# Demonstrates the four standard ways to estimate fixed effects, plus RE.
#
# Required packages:
#   install.packages(c("causaldata", "tidyverse", "fixest", "lme4",
#                      "modelsummary"))

library(tidyverse)
library(fixest)
library(lme4)
library(causaldata)

gm <- gapminder
gm$log_gdp <- log(gm$gdpPercap)

# ============================================================
# Method 1: Manual within-demeaning (didactic)
# Subtract each unit's mean -> isolate within-unit variation.
# ============================================================
gm <- gm %>%
  group_by(country) %>%
  mutate(lifeExp_within = lifeExp - mean(lifeExp),
         log_gdp_within = log_gdp - mean(log_gdp)) %>%
  ungroup()

m_within <- lm(lifeExp_within ~ log_gdp_within, data = gm)
cat("Method 1 (within demean):\n")
print(coef(summary(m_within))["log_gdp_within", ])

# ============================================================
# Method 2: LSDV (Least Squares Dummy Variables)
# Adds one binary indicator per country. Slow at scale but didactic.
# ============================================================
m_lsdv <- lm(lifeExp ~ log_gdp + factor(country), data = gm)
cat("\nMethod 2 (LSDV - factor(country)):\n")
print(coef(summary(m_lsdv))["log_gdp", ])

# ============================================================
# Method 3: One-way fixed effects via fixest::feols
# Production-grade, fast, clusters SEs by first FE by default.
# ============================================================
m_fe1 <- feols(lifeExp ~ log(gdpPercap) | country, data = gm)
cat("\nMethod 3 (one-way FE via feols):\n")
print(summary(m_fe1))

# ============================================================
# Method 4: Two-Way Fixed Effects (country + year)
# Controls for all time-invariant country traits AND common time shocks.
# ============================================================
m_twfe <- feols(lifeExp ~ log(gdpPercap) | country + year, data = gm)
cat("\nMethod 4 (TWFE country + year):\n")
print(summary(m_twfe))

# Two-way clustering (country AND year) for inference robustness.
m_twfe_cl2 <- feols(lifeExp ~ log(gdpPercap) | country + year,
                    data = gm, cluster = ~country + year)
cat("\nMethod 4 with two-way clustered SEs:\n")
print(summary(m_twfe_cl2))

# ============================================================
# Method 5: Random Effects via lme4::lmer
# ------------------------------------------------------------
# NOTE FOR STUDENTS — this is *advanced self-study material*.
# This course is not a multilevel-modeling course; we cover RE only
# briefly so you know it exists and where it fits.
#
# What RE does differently from FE:
#   * RE assumes each country effect is drawn from a normal distribution
#     (instead of being its own free parameter as in FE).
#   * This pools information across countries — efficient if the assumption
#     holds, biased if it doesn't.
#   * KEY IDENTIFYING ASSUMPTION: cov(country_effect, log(gdpPercap)) = 0.
#     In observational economics this is rarely defensible — that's why the
#     applied causal-inference literature defaults to FE/TWFE, not RE.
#
# Where RE is genuinely useful (worth learning on your own time):
#   * Hierarchical / nested designs (students within schools within districts).
#   * Small-T panels where FE eats too many degrees of freedom.
#   * Bayesian extensions and partial-pooling shrinkage estimators.
#
# Recommended starting points if you want to go deeper:
#   * Gelman & Hill, *Data Analysis Using Regression and Multilevel Models*.
#   * McElreath, *Statistical Rethinking* (Ch. 12-14) — Bayesian flavour.
#   * The lme4 vignettes (run `vignette("lmer", package = "lme4")`).
#
# We will not test you on the lmer output below.
# ============================================================
m_re <- lmer(lifeExp ~ log(gdpPercap) +
               (1 | country) + (1 | year),
             data = gm)
cat("\nMethod 5 (Random Effects via lmer):\n")
print(summary(m_re))
