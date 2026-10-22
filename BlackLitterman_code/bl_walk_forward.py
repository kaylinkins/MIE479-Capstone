# ============================================================
# CAPSTONE PROJECT: WALK-FORWARD TEST WITH BLACK-LITTERMAN
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# What this file does:
#   The same monthly walk-forward as walk_forward.py (report Section 2.5.4,
#   Table 5), with Black-Litterman expected returns added. At the start of
#   every validation month it:
#     1. builds overlapping month-length historical scenarios from the five
#        years strictly before that month (rolling window),
#     2. estimates the scenario mean and covariance, and the BL inputs
#        (pi, Q = scenario mean, Omega, mu_BL) from those same scenarios,
#     3. fits each model and holds the weights for the month (they drift
#        with prices; no trading inside the month),
#     4. records the actual daily returns of that month.
#   The historical-mean rows reproduce Table 5.
#
#   Models fitted with historical means (identical to walk_forward.py):
#     MVO (max Sharpe), MVO (min variance), risk parity, min CVaR,
#     CVaR (matched return), equal weight, 60/40.
#   Models fitted with BL means:
#     MVO (max Sharpe) [BL], CVaR (matched return) [BL]
#   Min variance, risk parity and min CVaR do not use expected returns, so
#   BL does not change them and they are not repeated.
#
# Run:      python bl_walk_forward.py
# Outputs (in bl_outputs_v2/):
#   walk_forward_bl_results.csv   performance of each model
#   walk_forward_bl_weights.csv   weights chosen at the start of each month
#   walk_forward_bl_returns.csv   stitched daily out-of-sample returns
#
# Notes:
#   - The first month fitted is the first validation month, so the first
#     fit uses the training data only. The test set is NOT used. Keep
#     INCLUDE_TEST = False until the models are finalized.
#   - Trading costs work as in walk_forward.py (COST_BPS, 0 = none).
# ============================================================

import numpy as np
import pandas as pd

import bl_core as bl

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

INCLUDE_TEST = False  # keep False until the models are finalized
ROLLING_YEARS = 5     # length of the rolling estimation window
PRINT_WEIGHTS = True  # print the weights chosen at every monthly rebalance
COST_BPS = {"Equity": 0.0, "Bond": 0.0, "BTC": 0.0}

out_dir = bl.HERE / "bl_outputs_v2"
out_dir.mkdir(exist_ok=True)


# STEP 1: LOAD DATA

train, validation = bl.load("train"), bl.load("validation")
frames = [train, validation]
if INCLUDE_TEST:
    frames.append(bl.load("test"))
assets = list(train.columns)
n = len(assets)
returns = pd.concat(frames)[assets].sort_index()
cost_rate = np.array([COST_BPS.get(a, 0.0) for a in assets]) / 10_000
failures = {"SLSQP": 0, "LP": []}
BL_MODELS = ("MVO (max Sharpe)", "CVaR (matched return)")


# STEP 2: FIT EVERY MODEL ON THE HISTORY BEFORE A MONTH

def fit_all(R, month, previous, holding_days):
    """R: month-length scenarios. BL is fitted on the same scenarios."""
    fit = bl.black_litterman(R, assets)
    models = {}
    for label, mu in (("Historical", R.mean(axis=0)), ("BL", fit["mu_bl"])):
        try:
            fitted, bad = bl.fit_models(R, assets, mu, fit["cov"], holding_days)
            failures["SLSQP"] += bad
        except RuntimeError as error:   # keep last month's weights if the LP fails
            failures["LP"].append((month, label, str(error)))
            suffix = " [BL]" if label == "BL" else ""
            fitted = {k: previous[k + suffix] for k in BL_MODELS if k + suffix in previous}
        for name, w in fitted.items():
            if label == "Historical":
                models[name] = w
            elif name in BL_MODELS:
                models[name + " [BL]"] = w
    models.update(bl.benchmarks(assets))
    return models


# STEP 3: WALK FORWARD (expanding window, monthly rebalancing)

oos_start = validation.index.min()
months = returns.index.to_period("M")
oos_months = sorted(set(months[returns.index >= oos_start]))
if returns.index[months == oos_months[0]][0] < oos_start:
    raise ValueError("The validation set must start at the beginning of a month.")

daily, turnover, costs, drifted = {}, {}, {}, {}
weight_rows, last_weights = [], {}

for m in oos_months:
    days = returns[months == m]
    date = days.index[0]
    # Only observations in the five years strictly before the rebalance.
    history = returns.loc[(returns.index < date) &
                          (returns.index >= date - pd.DateOffset(years=ROLLING_YEARS))]
    assert history.index.max() < days.index.min(), "Look-ahead detected."

    scenarios = bl.historical_scenarios(history, len(days))
    weights = fit_all(scenarios, str(m), last_weights, len(days))
    last_weights = weights
    growth = (1 + days.values).cumprod(axis=0)

    for name, w in weights.items():
        if name in drifted:
            turnover.setdefault(name, []).append(0.5 * np.abs(w - drifted[name]).sum())
        cost = float(cost_rate @ np.abs(w - drifted.get(name, np.zeros(n))))
        costs.setdefault(name, []).append(cost)
        wealth = (growth @ w) * (1.0 - cost)
        previous = np.concatenate([[1.0], wealth[:-1]])
        daily.setdefault(name, []).append(pd.Series(wealth / previous - 1.0, index=days.index))
        end_value = w * growth[-1]
        drifted[name] = end_value / end_value.sum()
        weight_rows.append({"Month": str(m), "Portfolio": name,
                            **{a: w[i] for i, a in enumerate(assets)}})

oos_returns = pd.DataFrame({name: pd.concat(parts) for name, parts in daily.items()})
weights_df = pd.DataFrame(weight_rows)


# STEP 4: PERFORMANCE ON THE STITCHED OUT-OF-SAMPLE RETURNS

results = pd.DataFrame({name: bl.performance(oos_returns[name])
                        for name in oos_returns.columns}).T
results["Avg Monthly Turnover"] = pd.Series({k: np.mean(v) for k, v in turnover.items()})
results["Total Costs Paid"] = pd.Series({k: np.sum(v) for k, v in costs.items()})
if "BTC" in assets:
    results["Avg BTC Weight"] = weights_df.groupby("Portfolio")["BTC"].mean()

results.to_csv(out_dir / "walk_forward_bl_results.csv")
weights_df.to_csv(out_dir / "walk_forward_bl_weights.csv", index=False)
oos_returns.to_csv(out_dir / "walk_forward_bl_returns.csv")


# STEP 5: PRINT

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print(f"Out-of-sample period: {oos_returns.index.min().date()} to "
      f"{oos_returns.index.max().date()} ({len(oos_returns)} days, "
      f"{len(oos_months)} monthly rebalances)")
print(f"Window: rolling {ROLLING_YEARS} years | Scenarios: month-length historical returns | Rebalancing: monthly")
print(f"BTC cap (MVO and CVaR models): {bl.BTC_CAP} | BL: lambda = {bl.RISK_AVERSION}, "
      f"tau = {bl.TAU}, Q = scenario mean, scenario covariance")
print("Trading costs (bps per amount traded): " +
      ", ".join(f"{a} {COST_BPS.get(a, 0.0):g}" for a in assets) +
      ("  [no costs charged]" if not cost_rate.any() else ""))
if failures["SLSQP"]:
    print(f"WARNING: the MVO solver reported {failures['SLSQP']} non-converged fits.")
for month, label, message in failures["LP"]:
    print(f"WARNING: CVaR LP failed {month} | {label} | {message} (kept last month's weights)")

print("\nWALK-FORWARD RESULTS (% except Sharpe; VaR/CVaR are daily losses):")
print(bl.show_pct(results).to_string())

print("\nAVERAGE WEIGHTS CHOSEN (%):")
print((weights_df.groupby("Portfolio")[assets].mean() * 100).round(1).to_string())

if PRINT_WEIGHTS:
    rebalance_date = {str(m): returns.index[months == m][0].date() for m in oos_months}
    for name in BL_MODELS + tuple(k + " [BL]" for k in BL_MODELS):
        table = (weights_df[weights_df["Portfolio"] == name]
                 .set_index("Month")[assets] * 100).round(1)
        table.index = [rebalance_date[m] for m in table.index]
        table.index.name = "Rebalance date"
        print(f"\nTARGET WEIGHTS AT EACH REBALANCE (%): {name}")
        print(table.to_string())

print(f"\nSaved to {out_dir.name}/: walk_forward_bl_results.csv, "
      "walk_forward_bl_weights.csv, walk_forward_bl_returns.csv")
