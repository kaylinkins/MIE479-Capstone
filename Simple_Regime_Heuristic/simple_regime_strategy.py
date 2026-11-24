"""Work Package 3.1: simple regime-rotation heuristic, three signal sources.

Signal sources (same rule, same accounting, same benchmarks for each):
  * HMM         HMM_regime_forecasts/hmm_outputs/regime_probabilities.csv
  * Jump        HMM_regime_forecasts/jump_outputs/regime_probabilities.csv
  * Jump tuned  HMM_regime_forecasts/jump_tuned_outputs/regime_probabilities.csv

Step 1 (two assets): lower-volatility state -> 100% Equity,
                     higher-volatility state -> 100% Bond.
Step 2 (three assets): lower-volatility state -> fixed Equity + BTC mix,
                       higher-volatility state -> 100% Bond.

Signal: p_bear_today = estimate of the higher-Equity-variance state using
returns strictly before the rebalance date. If p_bear_today >= tau, hold the
defensive allocation.
  * HMM: filtered probability in [0, 1].
  * Jump models: HARD 0/1 labels (online regime assignment, not a probability).
    Every tau in (0, 1] then gives identical decisions, so tau is not
    identifiable for them; 0.5 is reported as a nominal value only.

Threshold tau is chosen on TRAINING data only:
  * each model is refit at monthly origins inside the training period
    (expanding window, only data before each origin, 36-month burn-in);
  * tau is chosen from a preset grid by training net Sharpe after costs;
  * tau is then frozen and applied to the validation signals, which are read
    from the teammate's regime_probabilities.csv for each source.

Tuned jump penalty: taken from TUNED_JUMP_PENALTY if set, else from a cached
selected_settings.json, else (if the jump module supports it) from the
teammate's select_settings(), else INFERRED by finding which candidate penalty
reproduces the teammate's tuned validation CSV. Inference identifies what the
teammate used; it does not choose a setting from validation performance. The
teammate's tuning used the last 24 training months, which overlap the later
training origins, so tuned training results are in-sample for that setting.

A reproduction check refits each model at the validation origins with the same
code path used for training and compares to the teammate's CSV.

Sharpe ratio: daily net returns, 4% annual risk-free rate (converted to
0.04/252 per day), annualized with sqrt(252).

Training benchmarks are reported over the same training origins. The rotation's
tau was selected on that period, so its training results are in-sample.

Only portfolio_train.csv and portfolio_validation.csv are read, through the
teammate's load_returns(). The test split is never opened.
Validation has been examined repeatedly: results are exploratory.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# ----------------------------- settings --------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HMM_DIR = ROOT / "HMM_regime_forecasts"
HMM_SCRIPT = HMM_DIR / "hmm_regime_model.py"
DATA_DIR = None            # set to a Path if portfolio_train/validation.csv live elsewhere
OUTPUT = HERE / "outputs"

# folder = per-source results; cache = cached training signals.
# The HMM reuses the cache written by the earlier version of this script.
SOURCES = {
    "hmm": dict(label="HMM",
                probs=HMM_DIR / "hmm_outputs" / "regime_probabilities.csv",
                folder=OUTPUT / "hmm",
                cache=OUTPUT / "hmm" / "training_signals.csv"),
    "jump": dict(label="Jump",
                 probs=HMM_DIR / "jump_outputs" / "regime_probabilities.csv",
                 folder=OUTPUT / "jump",
                 cache=OUTPUT / "jump" / "training_signals.csv"),
    "jump_tuned": dict(label="Jump tuned",
                       probs=HMM_DIR / "jump_tuned_outputs" / "regime_probabilities.csv",
                       folder=OUTPUT / "jump_tuned",
                       cache=OUTPUT / "jump_tuned" / "training_signals.csv"),
}
RUN_SOURCES = ("hmm", "jump", "jump_tuned")

# Tuned jump penalty: the value the teammate's tuning run printed
# ("Selected penalty=..."). Leave None to use a cached value or to infer it.
TUNED_JUMP_PENALTY = None
TUNED_PENALTY_CANDIDATES = (10.0, 50.0, 150.0)   # the teammate's PENALTIES grid

ASSETS = ("Equity", "Bond", "BTC")
COST_RATE = 0.001          # charged on traded notional (buys + sells)
BURN_IN_MONTHS = 36        # minimum history before the first training origin
THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
BTC_WEIGHT_FAVORABLE = 0.20  # Step 2: fixed in advance (the BTC cap); not tuned
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04      # annual; Sharpe uses RISK_FREE_RATE / TRADING_DAYS per day
FORCE_REFIT = False        # True = ignore all cached training signals / checks / settings
VERIFY_VALIDATION_REFIT = True  # refit at validation origins and compare to the CSVs
PRINT_DETAIL = True        # False = only threshold grids and comparison tables
SIGNAL_TOLERANCE = 1e-6


# ------------------------- loading and signals --------------------------
def import_hmm_module():
    if not HMM_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find {HMM_SCRIPT}")
    spec = importlib.util.spec_from_file_location("hmm_regime_model", HMM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hmm_regime_model"] = module
    spec.loader.exec_module(module)  # main() is guarded by __name__, so it does not run
    return module


def import_jump_module():
    """Import the teammate's jump module (needs the `jumpmodels` package)."""
    sys.path.insert(0, str(HMM_DIR))
    try:
        import jump_regime_model as jump
    except ImportError as err:
        raise ImportError(f"Could not import jump_regime_model from {HMM_DIR}: {err}") from err
    return jump


def jump_fit_at(jump, history: pd.DataFrame, seed: int, penalty: float | None = None) -> float:
    """Jump-model p_bear_today (hard 0/1).

    penalty=None uses the module's default. Newer jump_regime_model versions take
    a jump_penalty argument; the version that only reads the module-level
    JUMP_PENALTY is handled by setting it temporarily (no file is modified).
    """
    if penalty is None:
        return float(jump.fit_jump(history, seed=seed)[1][1])
    if "jump_penalty" in inspect.signature(jump.fit_jump).parameters:
        return float(jump.fit_jump(history, seed=seed, jump_penalty=float(penalty))[1][1])
    if not hasattr(jump, "JUMP_PENALTY"):
        raise RuntimeError("jump_regime_model has neither a jump_penalty argument nor a JUMP_PENALTY "
                           "global, so the penalty cannot be varied.")
    original = jump.JUMP_PENALTY
    jump.JUMP_PENALTY = float(penalty)
    try:
        return float(jump.fit_jump(history, seed=seed)[1][1])
    finally:
        jump.JUMP_PENALTY = original


def infer_tuned_penalty(jump, full: pd.DataFrame, origins: pd.DatetimeIndex,
                        probs: pd.DataFrame) -> list:
    """Candidate penalties whose refit reproduces the teammate's tuned validation CSV exactly."""
    target = probs["p_bear_today"].to_numpy(dtype=float)
    matches = []
    for penalty in TUNED_PENALTY_CANDIDATES:
        ok = True
        for i, date in enumerate(origins):
            history = full.loc[full.index < date]
            if abs(jump_fit_at(jump, history, i, penalty) - target[i]) > SIGNAL_TOLERANCE:
                ok = False
                print(f"Penalty {penalty:g}: first mismatch at validation origin {i + 1}/{len(origins)} "
                      f"({date.date()})", flush=True)
                break
        if ok:
            print(f"Penalty {penalty:g}: reproduces all {len(origins)} validation origins.", flush=True)
            matches.append(penalty)
    return matches


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


def build_training_signals(label: str, fit, train: pd.DataFrame, origins: pd.DatetimeIndex,
                           cache: Path, tag: str | None = None) -> pd.DataFrame:
    """Walk-forward refits at training origins: fit only on training data before each origin.

    fit(history, seed) -> p_bear_today (float).
    """
    if cache.exists() and not FORCE_REFIT:
        cached = pd.read_csv(cache, parse_dates=["rebalance_date", "information_date"])
        same_origins = pd.DatetimeIndex(cached["rebalance_date"]).equals(origins)
        same_tag = tag is None or ("setting" in cached and (cached["setting"] == tag).all())
        if same_origins and same_tag:
            print(f"Using cached training signals ({label}): {cache}")
            return cached
        print(f"Cached training signals ({label}) do not match the current origins/settings; refitting.")
    rows = []
    for i, date in enumerate(origins):
        history = train.loc[train.index < date]  # strictly prior, training data only
        p = float(fit(history, i))
        rows.append(dict(
            rebalance_date=date,
            information_date=history.index[-1],
            p_bear_today=p,
            history_days=len(history),
            fit_seed=i,
            setting=tag or "",
        ))
        print(f"Training {label} {i + 1}/{len(origins)} {date.date()}: p_bear_today={p:.4f}", flush=True)
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def read_validation_signals(path: Path, origins: pd.DatetimeIndex, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    probs = pd.read_csv(path, parse_dates=["rebalance_date", "information_date"])
    return validate_signal_table(probs, origins, label)


def validate_signal_table(probs: pd.DataFrame, origins: pd.DatetimeIndex, label: str) -> pd.DataFrame:
    probs = probs[["rebalance_date", "information_date", "p_bear_today"]]
    probs = probs.sort_values("rebalance_date").reset_index(drop=True)
    if not pd.DatetimeIndex(probs["rebalance_date"]).equals(origins):
        raise ValueError(f"Signal origins for {label} do not match the requested month starts")
    if not (probs["information_date"] < probs["rebalance_date"]).all():
        raise ValueError(f"A {label} validation signal uses information on or after its rebalance date")
    if not probs["p_bear_today"].between(0, 1).all():
        raise ValueError(f"Bear signals for {label} must be between 0 and 1")
    return probs


def verify_validation_refit(label: str, fit, full: pd.DataFrame, origins: pd.DatetimeIndex,
                            probs: pd.DataFrame, path: Path, tag: str | None) -> dict:
    """Refit at validation origins with the training code path; compare to the teammate's CSV."""
    result = None
    if path.exists() and not FORCE_REFIT:
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("setting") == (tag or ""):
            result = cached
            print(f"Using cached reproduction check ({label}): {path}")
    if result is None:
        refit = []
        for i, date in enumerate(origins):
            history = full.loc[full.index < date]
            refit.append(float(fit(history, i)))
            print(f"Reproduction check {label} {i + 1}/{len(origins)} {date.date()}", flush=True)
        diff = np.abs(np.array(refit) - probs["p_bear_today"].to_numpy())
        result = dict(setting=tag or "", origins=len(origins), max_abs_difference=float(diff.max()),
                      mismatched_origins=int((diff > SIGNAL_TOLERANCE).sum()))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
    if result["mismatched_origins"] > 0:
        print(f"WARNING ({label}): my refit differs from the teammate's CSV at "
              f"{result['mismatched_origins']}/{result['origins']} validation origins "
              f"(max abs difference {result['max_abs_difference']:.6f}). "
              "Training signals may not be built the same way as validation signals.")
    else:
        print(f"Reproduction check ({label}): refit matches the teammate's validation CSV "
              f"at all {result['origins']} origins.")
    return result


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
    sharpe = ((daily.mean() - RISK_FREE_RATE / TRADING_DAYS) / std * np.sqrt(TRADING_DAYS)
              if std > 0 else np.nan)
    out = {
        "Strategy": name,
        "Total return (%)": 100 * total,
        "Ann. return (%)": 100 * ((1 + total) ** (TRADING_DAYS / n) - 1),
        "Ann. volatility (%)": 100 * std * np.sqrt(TRADING_DAYS),
        "Sharpe": float(sharpe),  # daily net returns, 4% annual risk-free rate
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
def strategy_targets(step: int, origins: pd.DatetimeIndex, p: pd.Series, tau: float, label: str):
    """Return [(name, targets, is_rotation)] for the rotation rule and its benchmarks."""
    b = BTC_WEIGHT_FAVORABLE
    first = origins[0]
    bond = np.array([0.0, 1.0, 0.0])
    if step == 1:
        favorable = np.array([1.0, 0.0, 0.0])
        rotation_name = f"{label} rotation: Equity / Bond"
    else:
        favorable = np.array([1.0 - b, 0.0, b])
        rotation_name = f"{label} rotation: {100 * (1 - b):.0f}% Equity + {100 * b:.0f}% BTC / Bond"
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
             probs: pd.DataFrame, tau: float, label: str, only_rotation: bool = False):
    p = probs.set_index("rebalance_date")["p_bear_today"]
    rows, daily_all = [], {}
    for name, targets, is_rotation in strategy_targets(step, origins, p, tau, label):
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


def threshold_grid(step: int, window, origins, probs, label: str) -> pd.DataFrame:
    rows = []
    for tau in THRESHOLD_GRID:
        table, _ = run_step(step, window, origins, probs, tau, label, only_rotation=True)
        rows.append({"tau": tau, **table.iloc[0].drop("Strategy").to_dict()})
    return pd.DataFrame(rows)


def select_tau(grid: pd.DataFrame) -> float:
    """Highest training net Sharpe; ties go to the tau closest to 0.5, then the lower tau."""
    ranked = grid.assign(_sharpe=grid["Sharpe"].round(6), _dist=(grid["tau"] - 0.5).abs())
    ranked = ranked.sort_values(["_sharpe", "_dist", "tau"], ascending=[False, True, True])
    return float(ranked.iloc[0]["tau"])


def comparison_rotation_paths(train: pd.DataFrame, validation: pd.DataFrame, signals: pd.DataFrame):
    """Evaluate Step 2 using cached HMM training signals and the paired validation forecasts."""
    training_origins = month_start_dates(train.index)[BURN_IN_MONTHS + 1:]
    training_cache = SOURCES["hmm"]["cache"]
    if not training_cache.is_file():
        training_cache = OUTPUT / "training_hmm_signals.csv"
    training_signals = read_validation_signals(training_cache, training_origins, "HMM training")
    training_window = train.loc[training_origins[0]:]
    hmm_tau = select_tau(threshold_grid(2, training_window, training_origins, training_signals, "HMM"))

    origins = month_start_dates(validation.index)
    paths = {}
    for forecast, tau in (("HMM", hmm_tau), ("Jump Model", 0.5)):
        rows = signals.loc[signals.source.eq(forecast) & signals.rebalance_date.isin(origins)]
        probs = validate_signal_table(rows, origins, forecast)
        if forecast == "Jump Model" and not signal_is_binary(probs):
            raise ValueError("The tuned Jump Model comparison requires binary as-of regime labels")
        bear = probs.set_index("rebalance_date").p_bear_today
        _, targets, _ = strategy_targets(2, origins, bear, tau, forecast)[0]
        daily, _ = simulate(validation, origins, targets)
        weights = pd.DataFrame.from_dict(targets, orient="index", columns=ASSETS)
        paths[forecast] = daily, weights, bear, tau
    return paths


def signal_is_binary(*prob_tables: pd.DataFrame) -> bool:
    values = np.concatenate([t["p_bear_today"].to_numpy(dtype=float) for t in prob_tables])
    return bool(np.all(np.isclose(values, 0.0) | np.isclose(values, 1.0)))


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


def agreement_table(states: dict) -> pd.DataFrame:
    """Percent of months in which two sources make the same defensive/favorable decision."""
    names = list(states)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            out.loc[a, b] = 100 * float((states[a].to_numpy() == states[b].to_numpy()).mean())
    return out


# --------------------------------- output -------------------------------
def show(title: str, table: pd.DataFrame, digits: int = 3):
    print(f"\n{title}")
    print(table.round(digits).to_string(index=False))


RESULT_COLUMNS = ["Strategy", "Total return (%)", "Ann. return (%)", "Ann. volatility (%)", "Sharpe",
                  "Max drawdown (%)", "Avg monthly turnover (%)", "Total costs (% of initial capital)",
                  "Defensive months (%)", "Switches"]
WEIGHT_COLUMNS = ["Strategy", "Avg weight Equity (%)", "Avg weight Bond (%)", "Avg weight BTC (%)",
                  "95% VaR loss, daily (%)", "95% CVaR loss, daily (%)"]


def comparison(step_store: dict, which: str) -> pd.DataFrame:
    """Each source's rotation row, then the (source-independent) benchmarks once."""
    keys = list(step_store)
    rotations = [step_store[k][which].iloc[[0]].assign(**{"Frozen tau": step_store[k]["tau"]}) for k in keys]
    benchmarks = step_store[keys[0]][which].iloc[1:].assign(**{"Frozen tau": np.nan})
    return pd.concat(rotations + [benchmarks], ignore_index=True)


def publish_step2_validation_results():
    """Move all three completed Step 2 validation tables into the output root."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    moves = [
        (SOURCES[key]["folder"] / "step2_validation_results.csv",
         OUTPUT / f"{key}_step2_validation_results.csv")
        for key in RUN_SOURCES
    ]
    missing = [str(source) for source, _ in moves if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot publish Step 2 validation results; missing source file(s): "
            + ", ".join(missing)
        )
    invalid_destinations = [
        str(destination) for _, destination in moves
        if destination.exists() and not destination.is_file()
    ]
    if invalid_destinations:
        raise IsADirectoryError(
            "Cannot publish Step 2 validation results; destination(s) are not files: "
            + ", ".join(invalid_destinations)
        )

    for source, destination in moves:
        source.replace(destination)
        print(f"Moved {source.name} to {destination}")


def run_source(key: str, spec: dict, fit, tag: str | None, ctx) -> dict:
    label, folder = spec["label"], spec["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 78 + f"\nSIGNAL SOURCE: {label}\n" + "#" * 78)

    train_probs = build_training_signals(label, fit, ctx.train, ctx.train_origins, spec["cache"], tag)
    val_probs = read_validation_signals(spec["probs"], ctx.val_origins, label)
    val_probs.to_csv(folder / "validation_signals.csv", index=False)
    check = None
    if VERIFY_VALIDATION_REFIT:
        check = verify_validation_refit(label, fit, ctx.full, ctx.val_origins, val_probs,
                                        folder / "validation_reproduction_check.json", tag)
    binary = signal_is_binary(train_probs, val_probs)
    if binary:
        print(f"NOTE ({label}): the signal is hard 0/1, so every tau in the grid gives identical "
              "decisions. tau is not identifiable; 0.5 is reported as a nominal value only.")

    out = dict(label=label, binary=binary, check=check, train_probs=train_probs, val_probs=val_probs,
               steps={}, tau={})
    for step in (1, 2):
        step_label = "Step 1: Equity/Bond" if step == 1 else "Step 2: Equity/Bond/BTC"
        print("\n" + "=" * 78 + f"\n{label} | {step_label}\n" + "=" * 78)

        # ---- training: select tau, then show benchmarks over the same period ----
        grid = threshold_grid(step, ctx.train_window, ctx.train_origins, train_probs, label)
        tau = select_tau(grid)
        out["tau"][step] = tau
        grid.assign(selected=(grid["tau"] == tau)).to_csv(
            folder / f"step{step}_training_threshold_grid.csv", index=False)
        tau_text = f"Nominal tau = {tau} (binary signal; tau has no effect)" if binary else f"Selected tau = {tau}"
        show(f"{label} | {step_label} - TRAINING threshold grid (net of costs, 4% rf Sharpe). {tau_text}",
             grid[["tau", "Sharpe", "Ann. return (%)", "Ann. volatility (%)", "Max drawdown (%)",
                   "Avg monthly turnover (%)", "Defensive months (%)", "Switches"]])

        train_table, train_daily = run_step(step, ctx.train_window, ctx.train_origins, train_probs, tau, label)
        train_table.to_csv(folder / f"step{step}_training_benchmarks.csv", index=False)
        train_daily.to_csv(folder / f"step{step}_training_daily_net_returns.csv")
        train_diag = signal_diagnostics(ctx.train_window, train_probs, tau)
        train_diag.to_csv(folder / f"step{step}_training_signal_diagnostics.csv", index=False)
        if PRINT_DETAIL:
            show(f"{label} | {step_label} - TRAINING benchmarks, same {len(ctx.train_origins)} origins "
                 f"(tau = {tau} was set on this period, so the rotation row is in-sample)",
                 train_table[RESULT_COLUMNS], 2)
            show(f"{label} | {step_label} - TRAINING average weights (% of portfolio)",
                 train_table[WEIGHT_COLUMNS], 2)
            show(f"{label} | {step_label} - TRAINING: realized behavior by signaled state (descriptive)",
                 train_diag, 2)

        # ---- validation: frozen tau ----
        table, daily = run_step(step, ctx.validation, ctx.val_origins, val_probs, tau, label)
        table.to_csv(folder / f"step{step}_validation_results.csv", index=False)
        daily.to_csv(folder / f"step{step}_validation_daily_net_returns.csv")
        diag = signal_diagnostics(ctx.validation, val_probs, tau)
        diag.to_csv(folder / f"step{step}_validation_signal_diagnostics.csv", index=False)
        sens = pd.concat(
            [run_step(step, ctx.validation, ctx.val_origins, val_probs, t, label, only_rotation=True)[0].assign(tau=t)
             for t in THRESHOLD_GRID], ignore_index=True)
        sens.to_csv(folder / f"step{step}_validation_threshold_sensitivity_EXPLORATORY.csv", index=False)
        if PRINT_DETAIL:
            show(f"{label} | {step_label} - VALIDATION results with frozen tau = {tau} (exploratory)",
                 table[RESULT_COLUMNS], 2)
            show(f"{label} | {step_label} - VALIDATION average weights (% of portfolio)",
                 table[WEIGHT_COLUMNS], 2)
            show(f"{label} | {step_label} - VALIDATION: realized behavior by signaled state (descriptive)",
                 diag, 2)
            if binary:
                print(f"\n{label} | {step_label} - validation threshold sensitivity skipped: "
                      "binary signal, all taus identical.")
            else:
                show(f"{label} | {step_label} - VALIDATION threshold sensitivity (EXPLORATORY; NOT used to select tau)",
                     sens[["tau", "Sharpe", "Ann. return (%)", "Max drawdown (%)",
                           "Defensive months (%)", "Switches"]])
        out["steps"][step] = dict(train=train_table, val=table, tau=tau)
    return out


def resolve_tuned_settings(jump, ctx, hmm, train_probs_unused=None) -> dict:
    """Decide which jump penalty the tuned source used (see module docstring)."""
    folder = SOURCES["jump_tuned"]["folder"]
    settings_path = folder / "selected_settings.json"
    folder.mkdir(parents=True, exist_ok=True)

    if TUNED_JUMP_PENALTY is not None:
        settings = dict(penalty=float(TUNED_JUMP_PENALTY), source="set manually (TUNED_JUMP_PENALTY)")
    elif settings_path.exists() and not FORCE_REFIT:
        with open(settings_path) as handle:
            settings = json.load(handle)
        print(f"Using cached tuned-jump settings: {settings_path}")
        return settings
    elif "jump_penalty" in inspect.signature(jump.fit_jump).parameters:
        # The jump module supports the tuned script's interface: use its own selection.
        import jump_regime_model_tuned as jump_tuned
        calendar = hmm.make_calendar(ctx.full, ctx.validation)
        penalty, pooling = jump_tuned.select_settings(ctx.train, calendar)  # training data only
        settings = dict(penalty=float(penalty), pooling=float(pooling),
                        source="teammate's select_settings (last 24 training months)")
    else:
        print("jump_regime_model.fit_jump has no jump_penalty argument, so the tuned script's "
              "select_settings() cannot run here.\nInferring the penalty by finding which candidate "
              "reproduces the teammate's tuned validation CSV.")
        probs = read_validation_signals(SOURCES["jump_tuned"]["probs"], ctx.val_origins, "Jump tuned")
        matches = infer_tuned_penalty(jump, ctx.full, ctx.val_origins, probs)
        if not matches:
            raise RuntimeError(
                "No candidate penalty reproduces the tuned validation CSV with the jump_regime_model.py "
                "on disk. The tuned outputs were produced by a different version. Ask the teammate "
                "for the jump_regime_model.py used, or set TUNED_JUMP_PENALTY.")
        if len(matches) > 1:
            raise RuntimeError(
                f"Penalties {matches} all reproduce the tuned validation CSV, so the penalty cannot be "
                "identified from it. Ask the teammate which penalty was selected and set "
                "TUNED_JUMP_PENALTY.")
        settings = dict(penalty=float(matches[0]),
                        source="inferred: the only candidate penalty reproducing the tuned validation CSV")
        if matches[0] == 50.0:
            print("NOTE: the inferred penalty (50) equals the untuned default, so Jump tuned will "
                  "be identical to Jump.")
    with open(settings_path, "w") as handle:
        json.dump(settings, handle, indent=2)
    return settings


# --------------------------------- main ---------------------------------
def main():
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "misc").mkdir(parents=True, exist_ok=True)
    hmm = import_hmm_module()
    data_dir = Path(DATA_DIR) if DATA_DIR else find_data_dir()
    print(f"Reading portfolio_train.csv and portfolio_validation.csv from {data_dir}")
    print(f"Sharpe uses a {100 * RISK_FREE_RATE:.1f}% annual risk-free rate "
          f"({RISK_FREE_RATE / TRADING_DAYS:.6f} per day).")
    train, full, validation, _ = hmm.load_returns(data_dir)  # test split is never opened
    if not validation.index.isin(full.index).all() or full.index.min() != train.index.min():
        raise ValueError("load_returns() second output is not train + validation history as assumed")

    train_origins = month_start_dates(train.index)[BURN_IN_MONTHS + 1:]
    val_origins = month_start_dates(validation.index)
    train_window = train.loc[train.index >= train_origins[0]]
    print(f"Training signal origins: {len(train_origins)} ({train_origins[0].date()} to {train_origins[-1].date()})")
    print(f"Validation origins: {len(val_origins)} ({val_origins[0].date()} to {val_origins[-1].date()})")
    ctx = SimpleNamespace(train=train, full=full, validation=validation, train_origins=train_origins,
                          val_origins=val_origins, train_window=train_window)

    # ---- fit functions: history DataFrame, seed -> p_bear_today ----
    fits, tags, tuned_settings = {}, {}, None

    def hmm_fit(history, seed):
        model = hmm.fit_hmm(history.to_numpy(), seed=seed)
        return hmm.filtered_state_probabilities(model, history.to_numpy())[-1][1]

    fits["hmm"], tags["hmm"] = hmm_fit, None

    if "jump" in RUN_SOURCES or "jump_tuned" in RUN_SOURCES:
        jump = import_jump_module()

        def jump_fit(history, seed):
            return jump_fit_at(jump, history, seed, None)

        fits["jump"], tags["jump"] = jump_fit, "jump_penalty=default"

        if "jump_tuned" in RUN_SOURCES:
            tuned_settings = resolve_tuned_settings(jump, ctx, hmm)
            penalty = tuned_settings["penalty"]
            print(f"Tuned jump penalty used for training signals: {penalty:g} ({tuned_settings['source']})")

            def tuned_fit(history, seed, _p=penalty):
                return jump_fit_at(jump, history, seed, _p)

            fits["jump_tuned"], tags["jump_tuned"] = tuned_fit, f"jump_penalty={penalty:g}"

    metadata = dict(
        created=datetime.now().isoformat(timespec="seconds"),
        data_dir=str(data_dir), cost_rate=COST_RATE, burn_in_months=BURN_IN_MONTHS,
        threshold_grid=list(THRESHOLD_GRID), btc_weight_favorable=BTC_WEIGHT_FAVORABLE,
        risk_free_rate_annual=RISK_FREE_RATE,
        training_origins=len(train_origins), validation_origins=len(val_origins),
        training_window=[str(train_window.index[0].date()), str(train_window.index[-1].date())],
        validation_window=[str(validation.index[0].date()), str(validation.index[-1].date())],
        selection_criterion=("training net Sharpe (daily net returns, 4% annual risk-free rate "
                             "converted to 0.04/252 per day, sqrt(252) annualization); "
                             "ties go to the tau closest to 0.5"),
        tuned_jump_settings=tuned_settings,
        sources={},
    )

    results = {}
    for key in RUN_SOURCES:
        results[key] = run_source(key, SOURCES[key], fits[key], tags[key], ctx)
        r = results[key]
        metadata["sources"][key] = dict(
            label=r["label"], signal_is_binary=r["binary"],
            selected_tau={f"step{s}": r["tau"][s] for s in (1, 2)},
            validation_reproduction_check=r["check"])

    # ---- combined comparison: same benchmarks, three signal sources ----
    print("\n" + "#" * 78 + "\nCOMPARISON ACROSS SIGNAL SOURCES\n" + "#" * 78)
    for step in (1, 2):
        step_label = "Step 1: Equity/Bond" if step == 1 else "Step 2: Equity/Bond/BTC"
        step_store = {k: results[k]["steps"][step] for k in RUN_SOURCES}
        cols = RESULT_COLUMNS + ["Frozen tau"]
        train_cmp = comparison(step_store, "train")
        val_cmp = comparison(step_store, "val")
        train_cmp.to_csv(OUTPUT / "misc" / f"comparison_step{step}_training.csv", index=False)
        val_cmp.to_csv(OUTPUT / "misc" / f"comparison_step{step}_validation.csv", index=False)
        show(f"{step_label} - TRAINING comparison (tau chosen on this period for HMM; jump signals are binary)",
             train_cmp[cols], 2)
        show(f"{step_label} - VALIDATION comparison, frozen tau (exploratory; benchmarks identical across sources)",
             val_cmp[cols], 2)

        for which, window_name in (("training", "train_probs"), ("validation", "val_probs")):
            states = {}
            for k in RUN_SOURCES:
                p = results[k][window_name].set_index("rebalance_date")["p_bear_today"]
                states[results[k]["label"]] = p >= results[k]["tau"][step]
            print(f"\n{step_label} - {which.upper()} decision agreement between sources "
                  "(% of months with the same defensive/favorable choice)")
            print(agreement_table(states).round(1).to_string())

    with open(OUTPUT / "misc" / "run_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    publish_step2_validation_results()
    print(f"\nAll outputs written to {OUTPUT}")
    print("Validation is exploratory, not a fresh independent holdout. The project test split was not read.")


if __name__ == "__main__":
    main()
