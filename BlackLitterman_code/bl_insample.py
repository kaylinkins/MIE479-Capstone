# ============================================================
# CAPSTONE PROJECT: BLACK-LITTERMAN, IN-SAMPLE COMPARISON
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# What this file does:
#   Repeats the in-sample baseline analysis (baseline_models.py and
#   cvar_optimization.py, report Section 2.5.3) with Black-Litterman
#   expected returns added, so the two can be compared side by side.
#     1. Fits Black-Litterman on the TRAINING set and prints its inputs
#        (benchmark weights, Q, pi, Omega) and posterior expected returns.
#     2. Fits each baseline model with historical means and with BL means:
#          MVO (max Sharpe), MVO (min variance), risk parity, min CVaR,
#          CVaR (matched return), equal weight, 60/40.
#        Min variance, risk parity and min CVaR do not use expected returns,
#        so they are the same under both inputs and are shown once.
#     3. Traces the CVaR-return frontier with historical targets (as in
#        cvar_optimization.py) and with BL targets.
#   Weights are fitted once on training and held fixed (daily rebalancing),
#   as in baseline_models.py. Validation figures are a static check only;
#   the out-of-sample comparison is bl_walk_forward.py.
#
# Run:      python bl_insample.py
# Outputs (in bl_outputs_v2/):
#   insample_bl_estimates.csv   prior, views, posterior and historical means
#   insample_portfolios.csv     weights and train/validation metrics
#   insample_cvar_frontier.csv  CVaR frontiers, historical vs BL targets
# ============================================================

import numpy as np
import pandas as pd

import bl_core as bl

N_FRONTIER = 6        # points on each CVaR-return frontier (as cvar_optimization.py)

out_dir = bl.HERE / "bl_outputs_v2"
out_dir.mkdir(exist_ok=True)
T = bl.TRADING_DAYS


# STEP 1: LOAD DATA

train, validation = bl.load("train"), bl.load("validation")
assets = list(train.columns)
if train.index.max() >= validation.index.min():
    raise ValueError("All training dates must precede all validation dates.")
R_train, R_val = train.values, validation[assets].values
upper = bl.upper_bounds(assets)


# STEP 2: BLACK-LITTERMAN ON THE TRAINING SET

fit = bl.black_litterman(R_train, assets)
mu_hist = R_train.mean(axis=0)
mu_bl = fit["mu_bl"]
cov = fit["cov"]                           # sample covariance, used by both

estimates = pd.DataFrame({
    "Benchmark weight": fit["w_benchmark"],
    "Equilibrium prior (pi)": fit["pi"] * T,
    "View Q (historical mean)": fit["q"] * T,
    "View confidence": [bl.VIEW_CONFIDENCE[a] for a in assets],
    "BL posterior": mu_bl * T,
    "Volatility": np.sqrt(np.diag(cov) * T),
}, index=pd.Index(assets, name="Asset"))
estimates.to_csv(out_dir / "insample_bl_estimates.csv")


# STEP 3: BASELINE MODELS, HISTORICAL vs BL MEANS

hist_models, bad_h = bl.fit_models(R_train, assets, mu_hist, cov)
bl_models, bad_b = bl.fit_models(R_train, assets, mu_bl, cov)

portfolios = {}       # name -> (weights, expected daily return vector used)
for name, w in hist_models.items():
    tag = "" if name in ("MVO (min variance)", "Risk parity", "Min CVaR") else " [Historical]"
    portfolios[name + tag] = (w, mu_hist)
for name in ("MVO (max Sharpe)", "CVaR (matched return)"):
    portfolios[name + " [BL]"] = (bl_models[name], mu_bl)
for name, w in bl.benchmarks(assets).items():
    portfolios[name] = (w, mu_hist)

rows = []
for name, (w, mu) in portfolios.items():
    row = {"Portfolio": name, **{a: w[i] for i, a in enumerate(assets)},
           "Expected Ann. Return (own mean)": float(mu @ w) * T,
           "BL Expected Ann. Return": float(mu_bl @ w) * T}
    for label, R in (("Train", R_train), ("Val", R_val)):
        perf = bl.performance(R @ w)
        perf.pop("Total Return")
        row.update({f"{label} {k}": v for k, v in perf.items()})
    rows.append(row)
port_table = pd.DataFrame(rows).set_index("Portfolio")
port_table.to_csv(out_dir / "insample_portfolios.csv")


# STEP 4: CVaR FRONTIERS (historical targets vs BL targets)

def frontier(mu, label):
    """Min CVaR for N_FRONTIER-1 increasing targets measured with mu (daily)."""
    w_min = bl.min_cvar(R_train, upper)
    low = float(mu @ w_min)
    # Highest return attainable under the caps: fill the best assets first.
    order, left, max_ret = np.argsort(-mu), 1.0, 0.0
    for i in order:
        take = min(upper[i], left)
        max_ret += take * mu[i]
        left -= take
    high = low + 0.98 * max(0.0, max_ret - low)       # stay just inside feasibility
    out = {f"{label} | Min CVaR": w_min}
    for target in np.linspace(low, high, N_FRONTIER)[1:]:
        out[f"{label} | return >= {target * T:.1%}/yr"] = bl.min_cvar(
            R_train, upper, mu=mu, min_return=target)
    return out


rows = []
for label, mu in (("Historical", mu_hist), ("BL", mu_bl)):
    for name, w in frontier(mu, label).items():
        row = {"Portfolio": name, **{a: w[i] for i, a in enumerate(assets)},
               "Target-basis Ann. Return": float(mu @ w) * T}
        for period, R in (("Train", R_train), ("Val", R_val)):
            perf = bl.performance(R @ w)
            row.update({f"{period} Ann. Return": perf["Ann. Return"],
                        f"{period} Sharpe": perf["Sharpe"],
                        f"{period} 95% VaR (daily)": perf["95% VaR (daily)"],
                        f"{period} 95% CVaR (daily)": perf["95% CVaR (daily)"]})
        rows.append(row)
front = pd.DataFrame(rows).set_index("Portfolio")
front.to_csv(out_dir / "insample_cvar_frontier.csv")


# STEP 5: PRINT

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print(f"Train: {len(R_train)} days ({train.index.min().date()} to {train.index.max().date()}) | "
      f"Validation: {len(R_val)} days | BTC cap (MVO, CVaR): {bl.BTC_CAP}")
print(f"BL settings: lambda = {bl.RISK_AVERSION}, tau = {bl.TAU}, "
      f"risk-free = {bl.RISK_FREE:.0%}, covariance = sample")
if bad_h + bad_b:
    print(f"WARNING: {bad_h + bad_b} MVO fit(s) did not converge.")

print("\nBLACK-LITTERMAN ESTIMATES (annualized %, P = identity):")
print(bl.show_pct(estimates).to_string())

print("\nPORTFOLIOS: WEIGHTS AND IN-SAMPLE (TRAIN) PERFORMANCE "
      "(% except Sharpe; VaR/CVaR are daily losses):")
cols = assets + [c for c in port_table.columns if c.startswith("Train")] + ["BL Expected Ann. Return"]
print(bl.show_pct(port_table[cols], keep=("Train Sharpe",)).to_string())

print("\nSTATIC VALIDATION CHECK (weights fixed from training; see bl_walk_forward.py "
      "for the proper out-of-sample test):")
cols = [c for c in port_table.columns if c.startswith("Val")]
print(bl.show_pct(port_table[cols], keep=("Val Sharpe",)).to_string())

print("\nCVaR-RETURN FRONTIERS (targets measured with historical vs BL means):")
print(bl.show_pct(front, keep=("Train Sharpe", "Val Sharpe")).to_string())

print(f"\nSaved to {out_dir.name}/: insample_bl_estimates.csv, insample_portfolios.csv, "
      "insample_cvar_frontier.csv")
