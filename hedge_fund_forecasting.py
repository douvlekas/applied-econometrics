"""
Applied Econometrics - Project 2
Forecasting monthly hedge fund returns: ARMA vs multi-factor regression

Series analysed: Y1 = HFRI Fund Weighted Composite, Y5 = Event Driven (ED)
Estimation sample: 4/1990 - 12/2004   |   Out-of-sample forecast period: 1/2005 - 12/2005

Python re-implementation (statsmodels) of an R project.
Author: Georgios Douvlekas - MSc Business Mathematics (NKUA & AUEB)

Data file expected in ./data : data-assignment2.txt
  columns 1-9:  Y1..Y9 hedge fund index returns
  columns 10-24: X1..X15 market/macro factors (time t), used to predict Y at t+1
"""

import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)

Y_NAMES = ["HFRI", "EH", "M", "RVA", "ED", "CA", "DS", "EMN", "MA"]
X_NAMES = ["RUS-Rf", "RUS(-1)-Rf(-1)", "MXUS-Rf", "MEM-Rf", "SMB", "HML", "MOM", "SBGC-Rf",
           "SBWG-Rf", "LHY-Rf", "DEFSPR", "FRBI-Rf", "GSCI-Rf", "VIX", "Rf"]

data = pd.read_csv("data/data-assignment2.txt", sep=r"\s+", header=None, names=Y_NAMES + X_NAMES)
data.index = pd.period_range("1990-04", periods=len(data), freq="M")

EST_END = pd.Period("2004-12", freq="M")
est = data.loc[:EST_END]
test = data.loc[pd.Period("2005-01", freq="M"):]
H = len(test)  # 12 months


def msfe(actual, forecast):
    return np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)


def hit_ratio(actual, forecast):
    return np.mean(np.sign(np.asarray(actual)) == np.sign(np.asarray(forecast)))


def select_arma(y, exog=None, max_p=2, max_q=2, criterion="bic"):
    """Grid search over ARMA(p, q) with a constant, choosing by information criterion."""
    results = []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            fit = ARIMA(y, exog=exog, order=(p, 0, q), trend="c").fit()
            results.append({"p": p, "q": q, "AIC": fit.aic, "BIC": fit.bic})
        except Exception:
            continue
    table = pd.DataFrame(results).sort_values(criterion.upper())
    return table


summary_rows = []
forecast_store = {}

for series in ["HFRI", "ED"]:
    print("=" * 70, f"\nSERIES: {series}\n" + "=" * 70)
    y = est[series]

    # ------------------------------------------------------------------
    # (1) Univariate ARMA model (Box-Jenkins, order chosen by AIC/BIC)
    # ------------------------------------------------------------------
    grid = select_arma(y)
    print("ARMA order selection (lowest BIC first):\n", grid.round(2).head())
    p, q = int(grid.iloc[0]["p"]), int(grid.iloc[0]["q"])
    arma = ARIMA(y, order=(p, 0, q), trend="c").fit()
    print(arma.summary())

    # ------------------------------------------------------------------
    # (2) Predictive multi-factor regression: Y(t+1) = a + sum b_i X_i(t) + e(t+1)
    # ------------------------------------------------------------------
    y_next = est[series].iloc[1:].reset_index(drop=True)          # Y at t+1
    X_t_df = est[X_NAMES].iloc[:-1].reset_index(drop=True)        # X at t
    ols = sm.OLS(y_next, sm.add_constant(X_t_df)).fit()
    print(ols.summary())

    # ------------------------------------------------------------------
    # (3) Residual autocorrelation diagnostics + regression with ARMA errors
    # ------------------------------------------------------------------
    dw = durbin_watson(ols.resid)
    bg_lm, bg_p, _, _ = acorr_breusch_godfrey(ols, nlags=1)
    print(f"Durbin-Watson = {dw:.4f} | Breusch-Godfrey(1) p-value = {bg_p:.4f}")

    grid_err = select_arma(y_next, exog=X_t_df)
    pe, qe = int(grid_err.iloc[0]["p"]), int(grid_err.iloc[0]["q"])
    reg_arma = ARIMA(y_next, exog=X_t_df, order=(pe, 0, qe), trend="c").fit()
    print(f"Regression with ARMA({pe},{qe}) errors: AIC = {reg_arma.aic:.2f}, BIC = {reg_arma.bic:.2f}")

    # ------------------------------------------------------------------
    # (4)-(5) Out-of-sample forecasts for 2005
    # ------------------------------------------------------------------
    actual = test[series].values
    fc_arma = arma.forecast(steps=H).values                       # dynamic multi-step forecast
    X_fc = sm.add_constant(data[X_NAMES].loc[EST_END:].iloc[:H].reset_index(drop=True),
                           has_constant="add")
    fc_ols = ols.predict(X_fc).values                             # Y(t+1) from X(t)
    forecast_store[series] = (actual, fc_arma, fc_ols, f"ARMA({p},{q})")

    summary_rows += [
        [series, f"ARMA({p},{q})", arma.aic, arma.bic, msfe(actual, fc_arma), hit_ratio(actual, fc_arma)],
        [series, f"Factor regression, ARMA({pe},{qe}) errors", reg_arma.aic, reg_arma.bic,
         msfe(actual, fc_ols), hit_ratio(actual, fc_ols)],
    ]

summary = pd.DataFrame(summary_rows, columns=["Series", "Model", "AIC", "BIC", "MSFE 2005", "Hit ratio 2005"])
print("\n", summary.round(6).to_string(index=False))

# Chart: 2005 forecasts vs actual returns
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
months = test.index.strftime("%b")
for ax, (series, (actual, fa, fo, label)) in zip(axes, forecast_store.items()):
    ax.plot(months, actual, "o-", color="black", label="Actual")
    ax.plot(months, fa, "s--", color="tab:blue", label=label)
    ax.plot(months, fo, "^--", color="tab:orange", label="Factor regression")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"{series}: out-of-sample forecasts, 2005")
    ax.set_ylabel("Monthly return")
    ax.legend()
plt.tight_layout()
plt.savefig("images/forecasts_2005.png", dpi=150)
plt.close()
