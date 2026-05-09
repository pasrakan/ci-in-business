# session6.py
# Session 6: Panel Data, Fixed Effects & Random Effects
# Uses Gapminder via causaldata.
#
# Required packages:
#   pip install causaldata pandas numpy statsmodels linearmodels

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels import PanelOLS, RandomEffects
from causaldata import gapminder

gm = gapminder.load_pandas().data
gm["log_gdp"] = np.log(gm["gdpPercap"])

# ============================================================
# Method 1: Manual within-demeaning (didactic)
# ============================================================
gm[["lifeExp_within", "log_gdp_within"]] = (
    gm.groupby("country")[["lifeExp", "log_gdp"]]
      .transform(lambda x: x - x.mean())
)
m_within = smf.ols("lifeExp_within ~ log_gdp_within", data=gm).fit()
print("Method 1 (within demean):")
print(f"  log_gdp_within coef: {m_within.params['log_gdp_within']:.3f} "
      f"(SE = {m_within.bse['log_gdp_within']:.3f})")

# ============================================================
# Method 2: LSDV with C(country)
# ============================================================
m_lsdv = smf.ols("lifeExp ~ log_gdp + C(country)", data=gm).fit()
print(f"\nMethod 2 (LSDV): log_gdp coef = "
      f"{m_lsdv.params['log_gdp']:.3f} "
      f"(SE = {m_lsdv.bse['log_gdp']:.3f})")

# ============================================================
# Method 3 + 4: FE / TWFE via linearmodels.PanelOLS
# ============================================================
gm_p = gm.set_index(["country", "year"])

mod_fe1 = PanelOLS.from_formula(
    "lifeExp ~ log_gdp + EntityEffects", gm_p)
fe1 = mod_fe1.fit(cov_type="clustered", cluster_entity=True)
print(f"\nMethod 3 (one-way FE country, clustered):")
print(f"  log_gdp coef: {fe1.params['log_gdp']:.3f} "
      f"(SE = {fe1.std_errors['log_gdp']:.3f})")

mod_twfe = PanelOLS.from_formula(
    "lifeExp ~ log_gdp + EntityEffects + TimeEffects", gm_p)
twfe = mod_twfe.fit(cov_type="clustered", cluster_entity=True)
print(f"\nMethod 4 (TWFE country + year, clustered):")
print(f"  log_gdp coef: {twfe.params['log_gdp']:.3f} "
      f"(SE = {twfe.std_errors['log_gdp']:.3f})")

# ============================================================
# Method 5: Random Effects via linearmodels.RandomEffects
# ============================================================
mod_re = RandomEffects.from_formula("lifeExp ~ log_gdp", gm_p)
re = mod_re.fit()
print(f"\nMethod 5 (Random Effects):")
print(f"  log_gdp coef: {re.params['log_gdp']:.3f} "
      f"(SE = {re.std_errors['log_gdp']:.3f})")
