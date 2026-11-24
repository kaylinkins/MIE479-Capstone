# Rotation table in the teammates' format: Forecasts | Scenarios | Regime (Overall/Bull/Bear).
# HMM and Jump tuned, 3 assets (80% Equity + 20% BTC / Bond), 4% rf, validation only.
# Reads saved CSVs only. Does not refit and does not modify any existing file.
# Save in Simple_Regime_Heuristic/ and run.
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "outputs"
OUT = OUTPUT
OUT.mkdir(parents=True, exist_ok=True)

RF_ANNUAL, TD, COST_RATE, BTC_W = 0.04, 252, 0.001, 0.20
ASSETS = ["Equity", "Bond", "BTC"]
SOURCES = {"HMM": "hmm", "Jump tuned": "jump_tuned"}     # label -> folder / metadata key
SCENARIOS_LABEL = "None (rule-based)"
FAVORABLE = np.array([1 - BTC_W, 0.0, BTC_W])
BOND = np.array([0.0, 1.0, 0.0])


def find_validation_csv():
    for folder in [ROOT / "HMM_regime_forecasts", ROOT / "Data" / "datasets", ROOT / "Data", HERE, *HERE.parents]:
        if (folder / "portfolio_validation.csv").exists():
            return folder / "portfolio_validation.csv"
    found = sorted(ROOT.rglob("portfolio_validation.csv"))
    if found:
        return found[0]
    raise FileNotFoundError("portfolio_validation.csv not found.")


def frozen_tau(key):
    for p in (OUTPUT / "misc" / "run_metadata.json", OUTPUT / "run_metadata.json"):
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                return float(meta["sources"][key]["selected_tau"]["step2"]), str(p)
            except (KeyError, ValueError):
                pass
    print(f"WARNING: frozen tau for {key} not found in run_metadata.json; using 0.5.")
    return 0.5, "default 0.5 (metadata not found)"


def reconstruct(window, targets):
    """Same accounting as the rotation script: all-cash start, cost = 0.001 * (buys + sells)."""
    R = window[ASSETS].to_numpy(float)
    holdings, value = np.zeros(3), 1.0
    daily, cost_frac = [], {}
    for t, date in enumerate(window.index):
        prev = value
        if date in targets:
            desired = targets[date] * value
            cost = COST_RATE * np.abs(desired - holdings).sum()
            value -= cost
            holdings = targets[date] * value
            cost_frac[date] = cost / prev
        holdings = holdings * (1 + R[t])
        value = float(holdings.sum())
        daily.append(value / prev - 1)
    return pd.Series(daily, index=window.index), pd.Series(cost_frac)


def metrics(r, cost_sum):
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ["Net CAGR (%)", "Annual volatility (%)", "Annual Sharpe",
                                    "Daily 95% VaR (%)", "Monthly 95% VaR (%)",
                                    "Max drawdown (%)", "Annualized transaction costs (%)"]}
    a = r.to_numpy(float)
    years = n / TD
    wealth = np.cumprod(1 + a)
    peak = np.maximum.accumulate(np.r_[1., wealth])[1:]
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    sd = a.std(ddof=1)
    return {
        "Net CAGR (%)": (wealth[-1] ** (1 / years) - 1) * 100,
        "Annual volatility (%)": sd * np.sqrt(TD) * 100,
        "Annual Sharpe": (a.mean() - RF_ANNUAL / TD) / sd * np.sqrt(TD),
        "Daily 95% VaR (%)": np.quantile(-a, .95) * 100,
        "Monthly 95% VaR (%)": np.quantile(-monthly.to_numpy(), .95) * 100,
        "Max drawdown (%)": -(1 - wealth / peak).max() * 100,      # negative, as in the teammates' table
        "Annualized transaction costs (%)": cost_sum / years * 100,
    }


validation = pd.read_csv(find_validation_csv(), index_col=0, parse_dates=True).sort_index()
month_starts = validation.index[~validation.index.to_period("M").duplicated()]

rows, checks = [], []
for label, key in SOURCES.items():
    folder = OUTPUT / key
    sig = pd.read_csv(folder / "validation_signals.csv", parse_dates=["rebalance_date", "information_date"])
    sig = sig.sort_values("rebalance_date").set_index("rebalance_date")
    if not pd.DatetimeIndex(sig.index).equals(month_starts):
        raise ValueError(f"{label}: signal dates do not match validation month starts")
    tau, tau_src = frozen_tau(key)
    bear = sig["p_bear_today"] >= tau                      # True = higher-vol state = defensive (Bond)
    targets = {d: (BOND if bear[d] else FAVORABLE) for d in sig.index}

    rec_daily, cost_frac = reconstruct(validation, targets)
    saved = pd.read_csv(folder / "step2_validation_daily_net_returns.csv",
                        index_col=0, parse_dates=True).sort_index().iloc[:, 0]
    if len(saved) != len(rec_daily) or not saved.index.equals(rec_daily.index):
        raise ValueError(f"{label}: saved daily returns do not line up with the validation window")
    checks.append((label, tau, tau_src, float((rec_daily - saved).abs().max())))

    r = saved                                              # saved net returns are the authoritative series
    state_by_month = pd.Series(bear.to_numpy(), index=sig.index.to_period("M"))
    day_bear = r.index.to_period("M").map(state_by_month).to_numpy(dtype=bool)
    cost_bear = pd.Series(bear.to_numpy(), index=sig.index)
    for regime, mask, cmask in (("Overall", np.ones(len(r), bool), np.ones(len(sig), bool)),
                                ("Bull", ~day_bear, ~bear.to_numpy()),
                                ("Bear", day_bear, bear.to_numpy())):
        cost_sum = float(cost_frac[sig.index[cmask]].sum())
        m = metrics(r[mask], cost_sum)
        rows.append({"Forecasts": label, "Scenarios": SCENARIOS_LABEL, "Regime": regime, **m,
                     "Months": int(cmask.sum()), "Days": int(mask.sum())})

table = pd.DataFrame(rows)
table.to_csv(OUT / "rotation_overall_bull_bear.csv", index=False)

cols = ["Net CAGR (%)", "Annual volatility (%)", "Annual Sharpe", "Daily 95% VaR (%)",
        "Monthly 95% VaR (%)", "Max drawdown (%)", "Annualized transaction costs (%)", "Months", "Days"]
print("| Forecasts | Scenarios | Regime | " + " | ".join(cols) + " |")
print("|---|---|---|" + "---:|" * len(cols))
for _, row in table.iterrows():
    cells = []
    for c in cols:
        v = row[c]
        cells.append(f"{int(v)}" if c in ("Months", "Days") else
                     (f"{v:.3f}" if c == "Annual Sharpe" else f"{v:.2f}"))
    print(f"| {row['Forecasts']} | {row['Scenarios']} | {row['Regime']} | " + " | ".join(cells) + " |")

print("\nCHECK (re-simulated vs saved daily net returns; max abs difference should be ~1e-12):")
for label, tau, src, diff in checks:
    print(f"  {label:12s} tau={tau}  max diff={diff:.2e}  [{src}]")
print(f"\nSaved to {OUT}")
