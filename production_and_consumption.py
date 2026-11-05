"""
Applied Econometrics - Project 1
Part A: Cobb-Douglas production function for the Greek economy (1961-2003)
Part B: Household consumption function and heteroskedasticity (326 households, 2004)

Python re-implementation (statsmodels) of an R project.
Author: Georgios Douvlekas - MSc Business Mathematics (NKUA & AUEB)

Data files expected in ./data :
  cobb_douglas_data.txt                 columns: Q (GDP), K (capital), L (labour)
  consumption_data_cross_sectional.txt  column 5: Res, column 6: Y (income), column 7: C (consumption)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

os.makedirs("images", exist_ok=True)


def white_test(resid, Z):
    """White test: n * R^2 of the auxiliary regression of squared residuals on Z."""
    aux = sm.OLS(resid ** 2, Z).fit()
    lm = len(resid) * aux.rsquared
    df = Z.shape[1] - 1
    return lm, stats.chi2.sf(lm, df)


def one_sided_t(estimate, se, h0, df, alternative):
    t = (estimate - h0) / se
    p = stats.t.cdf(t, df) if alternative == "less" else stats.t.sf(t, df)
    return t, p


# =============================================================================
# PART A - Cobb-Douglas production function:  Q = A * K^b1 * L^b2 * e^u
#          log-linearised:  ln Q = b0 + b1 ln K + b2 ln L + u
# =============================================================================
cd = pd.read_csv("data/cobb_douglas_data.txt", sep=r"\s+", header=None, names=["Q", "K", "L"])
cd.index = range(1961, 1961 + len(cd))
cd["lnQ"], cd["lnK"], cd["lnL"] = np.log(cd["Q"]), np.log(cd["K"]), np.log(cd["L"])

X = sm.add_constant(cd[["lnK", "lnL"]])
m1 = sm.OLS(cd["lnQ"], X).fit()
print(m1.summary())

# (a) 90% confidence intervals for the elasticities
print("\n90% confidence intervals:\n", m1.conf_int(alpha=0.10))

# (b) Individual t-tests and overall F-test are reported in the summary above
print(f"F = {m1.fvalue:.1f}, p-value = {m1.f_pvalue:.2e}")

# (c) Hypothesis tests on the elasticities (alpha = 10%)
df_res = m1.df_resid
b, V = m1.params, m1.cov_params()
print("\nH0: b1 = 1 vs H1: b1 < 1  ->  t = %.3f, p = %.4f"
      % one_sided_t(b["lnK"], np.sqrt(V.loc["lnK", "lnK"]), 1, df_res, "less"))
print("H0: b2 = 1 vs H1: b2 > 1  ->  t = %.3f, p = %.4f"
      % one_sided_t(b["lnL"], np.sqrt(V.loc["lnL", "lnL"]), 1, df_res, "greater"))

# Returns to scale: b1 + b2 = 1
crs = m1.t_test("lnK + lnL = 1")
print("H0: b1 + b2 = 1 (two-sided):", crs)
t_crs = float(np.squeeze(crs.tvalue))
print("H0: b1 + b2 = 1 vs H1: b1 + b2 > 1 -> p = %.2e" % stats.t.sf(t_crs, df_res))

# (d) Point forecasts and 90% prediction intervals
points = pd.DataFrame({
    "const": 1.0,
    "lnK": [np.log(5_000_000), np.log(cd["K"].mean())],
    "lnL": [np.log(3_500_000), np.log(cd["L"].mean())],
}, index=["K=5,000,000 / L=3,500,000", "K, L at sample means"])
pred = m1.get_prediction(points).summary_frame(alpha=0.10)
pred_levels = np.exp(pred[["mean", "obs_ci_lower", "obs_ci_upper"]])
print("\nForecasts (log scale):\n", pred[["mean", "obs_ci_lower", "obs_ci_upper"]])
print("Forecasts (GDP level):\n", pred_levels.round(0))

# (e) Model selection: Adjusted R^2, AIC, BIC
candidates = {
    "M1: K, L": ["lnK", "lnL"],
    "M2: L only": ["lnL"],
    "M3: K only": ["lnK"],
    "M4: constant only": [],
}
rows = []
for name, cols in candidates.items():
    Xm = sm.add_constant(cd[cols]) if cols else np.ones((len(cd), 1))
    fit = sm.OLS(cd["lnQ"], Xm).fit()
    rows.append([name, fit.rsquared_adj if cols else 0.0, fit.aic, fit.bic])
print("\n", pd.DataFrame(rows, columns=["Model", "Adj. R2", "AIC", "BIC"]).round(3))

# (f) Residual diagnostics
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].scatter(m1.fittedvalues, m1.resid, color="steelblue")
ax[0].axhline(0, color="red", ls="--")
ax[0].set(title="Residuals vs Fitted", xlabel="Fitted ln(Q)", ylabel="Residual")
stats.probplot(m1.resid, dist="norm", plot=ax[1])
ax[1].set_title("Normal Q-Q Plot")
ax[2].plot(cd.index, m1.resid, "o-", color="forestgreen")
ax[2].axhline(0, color="red", ls="--")
ax[2].set(title="Residuals over Time (1961-2003)", xlabel="Year", ylabel="Residual")
plt.tight_layout()
plt.savefig("images/cobb_douglas_residuals.png", dpi=150)
plt.close()

# =============================================================================
# PART B - Consumption function  C_i = b0 + b1 Y_i + e_i  (326 households)
# =============================================================================
cons = pd.read_csv("data/consumption_data_cross_sectional.txt", sep=r"\s+", header=None)
df = pd.DataFrame({"Res": cons[4], "Y": cons[5], "C": cons[6]})
XB = sm.add_constant(df[["Y"]])

# (a) OLS with 95% confidence intervals
ols_b = sm.OLS(df["C"], XB).fit()
print("\nOLS consumption function:\n", ols_b.params.round(3), "\n", ols_b.conf_int(0.05).round(3))

# (b) Heteroskedasticity tests
bp_lm, bp_p, _, _ = het_breuschpagan(ols_b.resid, XB)
w_lm, w_p = white_test(ols_b.resid, np.column_stack([np.ones(len(df)), df["Y"], df["Y"] ** 2]))
print(f"Breusch-Pagan: LM = {bp_lm:.3f}, p = {bp_p:.2e}")
print(f"White:         LM = {w_lm:.3f}, p = {w_p:.2e}")

# (c) WLS assuming Var(e_i) = a * Y_i  ->  weights 1 / Y_i
wls = sm.WLS(df["C"], XB, weights=1 / df["Y"]).fit()
print("\nWLS:\n", wls.params.round(3), "\n", wls.conf_int(0.05).round(3))

# (d) Feasible GLS assuming Var(e_i) = a0 + a1 * Y_i
aux = sm.OLS(ols_b.resid ** 2, XB).fit()
sigma2_hat = aux.fittedvalues
assert (sigma2_hat > 0).all(), "Estimated variances must be positive"
fgls = sm.WLS(df["C"], XB, weights=1 / sigma2_hat).fit()
print("\nFGLS:\n", fgls.params.round(3), "\n", fgls.conf_int(0.05).round(3))

# (e) Place of residence: urban (base), semi-urban (D1), rural (D2) + income interactions
df["D1"] = (df["Res"] == 1).astype(float)
df["D2"] = (df["Res"] == 2).astype(float)
df["D1_Y"], df["D2_Y"] = df["D1"] * df["Y"], df["D2"] * df["Y"]
XF = sm.add_constant(df[["Y", "D1", "D2", "D1_Y", "D2_Y"]])
full = sm.OLS(df["C"], XF).fit()
print(full.summary())
print("Joint test, intercept shifts (D1 = D2 = 0):", full.f_test("D1 = 0, D2 = 0"))
print("Joint test, slope shifts (D1_Y = D2_Y = 0):", full.f_test("D1_Y = 0, D2_Y = 0"))

# (f) Heteroskedasticity tests and FGLS for the full model
bp_lm, bp_p, _, _ = het_breuschpagan(full.resid, XB)
Zw = np.column_stack([np.ones(len(df)), df["Y"], df["D1"], df["D2"], df["D1_Y"], df["D2_Y"],
                      df["Y"] ** 2, df["D1"] * df["Y"] ** 2, df["D2"] * df["Y"] ** 2])
w_lm, w_p = white_test(full.resid, Zw)
print(f"Full model - Breusch-Pagan: LM = {bp_lm:.2f}, p = {bp_p:.2e}; White: LM = {w_lm:.2f}, p = {w_p:.4f}")

aux = sm.OLS(full.resid ** 2, XB).fit()
fgls_full = sm.WLS(df["C"], XF, weights=1 / aux.fittedvalues).fit()
print("\nFGLS full model:\n", fgls_full.params.round(3), "\n", fgls_full.conf_int(0.05).round(3))

# Chart: OLS vs WLS vs FGLS
grid = np.linspace(df["Y"].min(), df["Y"].max(), 100)
plt.figure(figsize=(9, 6))
plt.scatter(df["Y"], df["C"], s=12, alpha=0.5, color="grey", label="Households")
for fit, label, col in [(ols_b, "OLS", "tab:blue"), (wls, "WLS", "tab:orange"), (fgls, "FGLS", "tab:green")]:
    plt.plot(grid, fit.params["const"] + fit.params["Y"] * grid, lw=2, color=col,
             label=f"{label}: MPC = {fit.params['Y']:.3f}")
plt.title("Consumption vs Income: OLS, WLS and FGLS")
plt.xlabel("Disposable income (EUR)")
plt.ylabel("Consumption (EUR)")
plt.legend()
plt.tight_layout()
plt.savefig("images/consumption_ols_wls_fgls.png", dpi=150)
plt.close()
