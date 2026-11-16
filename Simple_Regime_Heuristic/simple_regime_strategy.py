"""Work Package 3.1: simple HMM regime-rotation heuristic.

Step 1 (two assets): lower-volatility state -> 100% Equity,
                     higher-volatility state -> 100% Bond.
Step 2 (three assets): lower-volatility state -> fixed Equity + BTC mix,
                       higher-volatility state -> 100% Bond.

Signal: p_bear_today = filtered probability of the higher-Equity-variance
state using returns strictly before the rebalance date. If p_bear_today >= tau,
hold the defensive allocation.

Threshold tau is chosen on TRAINING data only:
  * the HMM is refit at monthly origins inside the training period
    (expanding window, only data before each origin, 36-month burn-in);
  * tau is chosen from a preset grid by training net Sharpe after costs;
  * tau is then frozen and applied to the validation signals.
Validation HMM probabilities are read from the teammate's
HMM_regime_forecasts/hmm_outputs/regime_probabilities.csv.

Training benchmarks (buy-and-hold, 50/50, equal weight) are reported over the
same training origins. The rotation's tau was selected on that same period, so
its training results are in-sample for tau and favor the rotation.

Only portfolio_train.csv and portfolio_validation.csv are read, through the
teammate's load_returns(). The test split is never opened.
Validation has been examined repeatedly: results are exploratory.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- settings --------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HMM_DIR = ROOT / "HMM_regime_forecasts"
HMM_SCRIPT = HMM_DIR / "hmm_regime_model.py"
HMM_VALIDATION_PROBS = HMM_DIR / "hmm_outputs" / "regime_probabilities.csv"
DATA_DIR = None            # set to a Path if portfolio_train/validation.csv live elsewhere
OUTPUT = HERE / "outputs"

ASSETS = ("Equity", "Bond", "BTC")
COST_RATE = 0.001          # charged on traded notional (buys + sells)
BURN_IN_MONTHS = 36        # minimum history before the first training origin
THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
BTC_WEIGHT_FAVORABLE = 0.20  # Step 2: fixed in advance (the BTC cap); not tuned
TRADING_DAYS = 252
FORCE_REFIT = False        # True = ignore the cached training HMM signals


# ------------------------- loading and signals --------------------------
def import_hmm_module():
    if not HMM_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find {HMM_SCRIPT}")
    spec = importlib.util.spec_from_file_location("hmm_regime_model", HMM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hmm_regime_model"] = module
    spec.loader.exec_module(module)  # main() is guarded by __name__, so it does not run
    return module


def find_data_dir():
    candidates = [HMM_DIR, ROOT, HERE, ROOT / "Data Work"]
    for folder in candidates:
        if (folder / "portfolio_train.csv").exists() and (folder / "portfolio_validation.csv").exists():
            return folder
    raise FileNotFoundError(
        "Could not find portfolio_train.csv and portfolio_validation.csv in: "
        + ", ".join(str(c) for c in candidates)
        + ". Set DATA_DIR at the top of this script."
    )


def month_start_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First observed trading day of each calendar month."""
    return index[~index.to_period("M").duplicated()]


def training_signals(hmm, train: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Walk-forward HMM at training origins: fit only on training data before each origin."""
    cache = OUTPUT / "training_hmm_signals.csv"
    if cache.exists() and not FORCE_REFIT:
        cached = pd.read_csv(cache, parse_dates=["rebalance_date", "information_date"])
        if pd.DatetimeIndex(cached["rebalance_date"]).equals(origins):
            print(f"Using cached training HMM signals: {cache}")
            return cached
        print("Cached training signals do not match the current origins; refitting.")
    rows = []
    for i, date in enumerate(origins):
        history = train.loc[train.index < date]  # strictly prior, training data only
        model = hmm.fit_hmm(history.to_numpy(), seed=i)
        p = hmm.filtered_state_probabilities(model, history.to_numpy())[-1]
        rows.append(dict(
            rebalance_date=date,
            information_date=history.index[-1],
            p_bear_today=float(p[1]),
            history_days=len(history),
            fit_seed=i,
        ))
        print(f"Training HMM {i + 1}/{len(origins)} {date.date()}: p_bear_today={p[1]:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def validation_signals(origins: pd.DatetimeIndex) -> pd.DataFrame:
    if not HMM_VALIDATION_PROBS.exists():
        raise FileNotFoundError(f"Cannot find {HMM_VALIDATION_PROBS}")
    probs = pd.read_csv(HMM_VALIDATION_PROBS, parse_dates=["rebalance_date", "information_date"])
    probs = probs[["rebalance_date", "information_date", "p_bear_today"]]
    probs = probs.sort_values("rebalance_date").reset_index(drop=True)
    if not pd.DatetimeIndex(probs["rebalance_date"]).equals(origins):
        raise ValueError("Validation HMM origins do not match the validation month starts")
    if (probs["information_date"] >= probs["rebalance_date"]).any():
        raise ValueError("A validation signal uses information on or after its rebalance date")
    return probs


# --------------------------- portfolio simulation -----------------------
def simulate(window: pd.DataFrame, origins: pd.DatetimeIndex, targets: dict):
    """Daily simulation with drifting weights and proportional trading costs.

    targets: {rebalance_date: target weights}. Dates not in targets are not traded,
    so a buy-and-hold portfolio has a single entry on its first date.
    New weights earn returns from the rebalance date onward (the signal uses
    only data before that date). The start is all cash, so the initial purchase
    is charged once like any other trade: cost = COST_RATE * (buys + sells).
    """
    missing = [d for d in targets if d not in window.index]
    if missing:
        raise ValueError(f"Target dates not in the return window: {missing[:3]}")
    R = window[list(ASSETS)].to_numpy()
    month_starts = set(origins)
    holdings, value = np.zeros(len(ASSETS)), 1.0
    daily, records = [], []
    for t, date in enumerate(window.index):
        prev_value = value
        trade_cost, traded = 0.0, 0.0
        if date in targets:
            desired = np.asarray(targets[date], dtype=float) * value
            traded = float(np.abs(desired - holdings).sum())
            trade_cost = COST_RATE * traded
            value -= trade_cost
            holdings = np.asarray(targets[date], dtype=float) * value
        if date in month_starts:
            weights = holdings / holdings.sum()
            records.append(dict(date=date, traded_pct=100 * traded / prev_value,
                                cost=trade_cost, **{f"w_{a}": 100 * w for a, w in zip(ASSETS, weights)}))
        holdings = holdings * (1.0 + R[t])
        value = float(holdings.sum())
        daily.append(value / prev_value - 1.0)
    return pd.Series(daily, index=window.index), pd.DataFrame(records)


def performance(name: str, daily: pd.Series, rec: pd.DataFrame, extra: dict | None = None) -> dict:
    n = len(daily)
    wealth = (1.0 + daily).cumprod()
    total = float(wealth.iloc[-1] - 1.0)
    std = float(daily.std(ddof=1))
    q = float(np.quantile(daily, 0.05))
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    out = {
        "Strategy": name,
        "Total return (%)": 100 * total,
        "Ann. return (%)": 100 * ((1 + total) ** (TRADING_DAYS / n) - 1),
        "Ann. volatility (%)": 100 * std * np.sqrt(TRADING_DAYS),
        "Sharpe": daily.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        "95% VaR loss, daily (%)": -100 * q,
        "95% CVaR loss, daily (%)": -100 * float(daily[daily <= q].mean()),
        "Max drawdown (%)": -100 * float((wealth.to_numpy() / peak - 1.0).min()),
        "Avg monthly turnover (%)": float(rec["traded_pct"].iloc[1:].mean()),  # excludes initial purchase
        "Total costs (% of initial capital)": 100 * float(rec["cost"].sum()),  # includes initial purchase
        "Months": len(rec),
        "Daily observations": n,
        "Defensive months (%)": np.nan,
        "Switches": np.nan,
        "Avg weight Equity (%)": float(rec["w_Equity"].mean()),
        "Avg weight Bond (%)": float(rec["w_Bond"].mean()),
        "Avg weight BTC (%)": float(rec["w_BTC"].mean()),
    }
    if extra:
        out.update(extra)
    return out


# ------------------------------ strategies ------------------------------
def strategy_targets(step: int, origins: pd.DatetimeIndex, p: pd.Series, tau: float):
    """Return [(name, targets, is_rotation)] for the HMM rule and its benchmarks."""
    b = BTC_WEIGHT_FAVORABLE
    first = origins[0]
    bond = np.array([0.0, 1.0, 0.0])
    if step == 1:
        favorable = np.array([1.0, 0.0, 0.0])
        rotation_name = "HMM rotation: Equity / Bond"
    else:
        favorable = np.array([1.0 - b, 0.0, b])
        rotation_name = f"HMM rotation: {100 * (1 - b):.0f}% Equity + {100 * b:.0f}% BTC / Bond"
    rotation = {d: (bond if p[d] >= tau else favorable) for d in origins}
    items = [(rotation_name, rotation, True)]
    if step == 1:
        half = np.array([0.5, 0.5, 0.0])
        items += [
            ("50/50 monthly rebalanced", {d: half for d in origins}, False),
            ("50/50 buy-and-hold", {first: half}, False),
            ("100% Equity buy-and-hold", {first: favorable}, False),
            ("100% Bond buy-and-hold", {first: bond}, False),
        ]
    else:
        third = np.ones(3) / 3
        items += [
            ("Equal weight (1/3 each) monthly rebalanced", {d: third for d in origins}, False),
            ("Equal weight (1/3 each) buy-and-hold", {first: third}, False),
            (f"Always {100 * (1 - b):.0f}% Equity + {100 * b:.0f}% BTC buy-and-hold", {first: favorable}, False),
            ("100% Bond buy-and-hold", {first: bond}, False),
        ]
    return items


def run_step(step: int, window: pd.DataFrame, origins: pd.DatetimeIndex,
             probs: pd.DataFrame, tau: float, only_rotation: bool = False):
    p = probs.set_index("rebalance_date")["p_bear_today"]
    rows, daily_all = [], {}
    for name, targets, is_rotation in strategy_targets(step, origins, p, tau):
        if only_rotation and not is_rotation:
            continue
        daily, rec = simulate(window, origins, targets)
        extra = None
        if is_rotation:
            defensive = (p.loc[origins] >= tau).astype(int)
            extra = {"Defensive months (%)": 100 * float(defensive.mean()),
                     "Switches": int(defensive.diff().abs().sum())}
        rows.append(performance(name, daily, rec, extra))
        daily_all[name] = daily
    return pd.DataFrame(rows), pd.DataFrame(daily_all)


def threshold_grid(step: int, window, origins, probs) -> pd.DataFrame:
    rows = []
    for tau in THRESHOLD_GRID:
        table, _ = run_step(step, window, origins, probs, tau, only_rotation=True)
        rows.append({"tau": tau, **table.iloc[0].drop("Strategy").to_dict()})
    return pd.DataFrame(rows)


def select_tau(grid: pd.DataFrame) -> float:
    """Highest training net Sharpe; ties go to the tau closest to 0.5, then the lower tau."""
    ranked = grid.assign(_sharpe=grid["Sharpe"].round(6), _dist=(grid["tau"] - 0.5).abs())
    ranked = ranked.sort_values(["_sharpe", "_dist", "tau"], ascending=[False, True, True])
    return float(ranked.iloc[0]["tau"])


def signal_diagnostics(window: pd.DataFrame, probs: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Realized asset behavior in months the signal put in each state (descriptive only)."""
    p = probs.set_index("rebalance_date")["p_bear_today"]
    months = window.index.to_period("M")
    monthly = (1.0 + window[list(ASSETS)]).groupby(months).prod() - 1.0
    state = pd.Series(np.where(p.to_numpy() >= tau, "Higher-vol state (p >= tau)",
                               "Lower-vol state (p < tau)"), index=p.index.to_period("M"))
    if not monthly.index.equals(state.index):
        raise ValueError("Signal months do not line up with return months")
    rows = []
    groups = [("All months", state.index)] + [(s, state.index[state == s]) for s in state.unique()]
    for label, sel in groups:
        eq_daily = window["Equity"][months.isin(sel)]
        rows.append({
            "Signal state": label,
            "Months": len(sel),
            "Avg Equity monthly return (%)": 100 * monthly.loc[sel, "Equity"].mean(),
            "Avg Bond monthly return (%)": 100 * monthly.loc[sel, "Bond"].mean(),
            "Avg BTC monthly return (%)": 100 * monthly.loc[sel, "BTC"].mean(),
            "Realized Equity ann. volatility (%)": 100 * eq_daily.std(ddof=1) * np.sqrt(TRADING_DAYS),
        })
    return pd.DataFrame(rows)


# --------------------------------- main ---------------------------------
def show(title: str, table: pd.DataFrame, digits: int = 3):
    print(f"\n{title}")
    print(table.round(digits).to_string(index=False))


RESULT_COLUMNS = ["Strategy", "Total return (%)", "Ann. return (%)", "Ann. volatility (%)", "Sharpe",
                  "Max drawdown (%)", "Avg monthly turnover (%)", "Total costs (% of initial capital)",
                  "Defensive months (%)", "Switches"]
WEIGHT_COLUMNS = ["Strategy", "Avg weight Equity (%)", "Avg weight Bond (%)", "Avg weight BTC (%)",
                  "95% VaR loss, daily (%)", "95% CVaR loss, daily (%)"]


def main():
    OUTPUT.mkdir(exist_ok=True)
    hmm = import_hmm_module()
    data_dir = Path(DATA_DIR) if DATA_DIR else find_data_dir()
    print(f"Reading portfolio_train.csv and portfolio_validation.csv from {data_dir}")
    train, _, validation, _ = hmm.load_returns(data_dir)  # test split is never opened

    train_origins = month_start_dates(train.index)[BURN_IN_MONTHS + 1:]
    val_origins = month_start_dates(validation.index)
    train_window = train.loc[train.index >= train_origins[0]]
    print(f"Training signal origins: {len(train_origins)} ({train_origins[0].date()} to {train_origins[-1].date()})")
    print(f"Validation origins: {len(val_origins)} ({val_origins[0].date()} to {val_origins[-1].date()})")

    train_probs = training_signals(hmm, train, train_origins)
    val_probs = validation_signals(val_origins)
    val_probs.to_csv(OUTPUT / "validation_signals.csv", index=False)

    metadata = dict(
        created=datetime.now().isoformat(timespec="seconds"),
        data_dir=str(data_dir), cost_rate=COST_RATE, burn_in_months=BURN_IN_MONTHS,
        threshold_grid=list(THRESHOLD_GRID), btc_weight_favorable=BTC_WEIGHT_FAVORABLE,
        training_origins=len(train_origins), validation_origins=len(val_origins),
        training_window=[str(train_window.index[0].date()), str(train_window.index[-1].date())],
        validation_window=[str(validation.index[0].date()), str(validation.index[-1].date())],
        selection_criterion="training net Sharpe (daily, zero risk-free rate, sqrt(252))",
        selected_tau={},
    )

    for step in (1, 2):
        label = "Step 1: Equity/Bond" if step == 1 else "Step 2: Equity/Bond/BTC"
        print("\n" + "=" * 78 + f"\n{label}\n" + "=" * 78)

        # ---- training: select tau, then show benchmarks over the same period ----
        grid = threshold_grid(step, train_window, train_origins, train_probs)
        tau = select_tau(grid)
        metadata["selected_tau"][f"step{step}"] = tau
        grid.assign(selected=(grid["tau"] == tau)).to_csv(
            OUTPUT / f"step{step}_training_threshold_grid.csv", index=False)
        show(f"{label} - TRAINING threshold grid (net of costs). Selected tau = {tau}",
             grid[["tau", "Sharpe", "Ann. return (%)", "Ann. volatility (%)", "Max drawdown (%)",
                   "Avg monthly turnover (%)", "Defensive months (%)", "Switches"]])

        train_table, train_daily = run_step(step, train_window, train_origins, train_probs, tau)
        train_table.to_csv(OUTPUT / f"step{step}_training_benchmarks.csv", index=False)
        train_daily.to_csv(OUTPUT / f"step{step}_training_daily_net_returns.csv")
        show(f"{label} - TRAINING benchmarks, same {len(train_origins)} origins "
             f"(tau = {tau} was selected on this period, so the rotation row is in-sample for tau)",
             train_table[RESULT_COLUMNS], 2)
        show(f"{label} - TRAINING average weights (Equity / Bond / BTC, % of portfolio)",
             train_table[WEIGHT_COLUMNS], 2)

        train_diag = signal_diagnostics(train_window, train_probs, tau)
        train_diag.to_csv(OUTPUT / f"step{step}_training_signal_diagnostics.csv", index=False)
        show(f"{label} - TRAINING: realized behavior by signaled state (descriptive)", train_diag, 2)

        # ---- validation: frozen tau ----
        table, daily = run_step(step, validation, val_origins, val_probs, tau)
        table.to_csv(OUTPUT / f"step{step}_validation_results.csv", index=False)
        daily.to_csv(OUTPUT / f"step{step}_validation_daily_net_returns.csv")
        show(f"{label} - VALIDATION results with frozen tau = {tau} (exploratory)",
             table[RESULT_COLUMNS], 2)
        show(f"{label} - VALIDATION average weights (Equity / Bond / BTC, % of portfolio)",
             table[WEIGHT_COLUMNS], 2)

        diag = signal_diagnostics(validation, val_probs, tau)
        diag.to_csv(OUTPUT / f"step{step}_validation_signal_diagnostics.csv", index=False)
        show(f"{label} - VALIDATION: realized behavior by signaled state (descriptive)", diag, 2)

        sens = pd.concat(
            [run_step(step, validation, val_origins, val_probs, t, only_rotation=True)[0].assign(tau=t)
             for t in THRESHOLD_GRID],
            ignore_index=True,
        )
        sens.to_csv(OUTPUT / f"step{step}_validation_threshold_sensitivity_EXPLORATORY.csv", index=False)
        show(f"{label} - VALIDATION threshold sensitivity (EXPLORATORY; NOT used to select tau)",
             sens[["tau", "Sharpe", "Ann. return (%)", "Max drawdown (%)", "Defensive months (%)", "Switches"]])

    with open(OUTPUT / "run_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nAll outputs written to {OUTPUT}")


if __name__ == "__main__":
    main()