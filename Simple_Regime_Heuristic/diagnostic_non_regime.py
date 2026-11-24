# ============================================================
# NON-REGIME MPO BASELINE TABLE (validation only)
# Returns : Return_Forecasting/outputs/walk_forward_validation_expected_returns_h1_h3.csv
#           (training-selected, non-regime forecasts)
# Covariance: Covariance_Estimation/outputs/covariances_ewma_lambda_0_93_walk_forward_validation.csv
#           (plain EWMA, non-regime)
# Risk-free rate 4% annual: used in the Sharpe ratio AND inside the Max Sharpe targets
#           (monthly rate = 0.04/252 * trading days in the stage).
# Does not read portfolio_test.csv. Does not modify any existing script.
# Put this file in Simple_Regime_Heuristic/.
# ============================================================
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT = HERE / "outputs" / "non_regime_mpo_rf0_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- settings ----------------
RF_ANNUAL = 0.04
TRADING_DAYS = 252
HORIZONS = (1, 2, 3)
COST_RATE = 0.001
BTC_CAP = 0.20
CVAR_RISK_AVERSION = 0.5
CVAR_ALPHA = 0.95
N_SCENARIOS = 4000
SEED = 20240601
INCLUDE_RISK_PARITY = True
INCLUDE_CVAR = True
INCLUDE_FIXED_REBALANCE = True      # simple monthly-rebalanced 60/40 and equal weight (no optimizer)
INCLUDE_TRACKING_BASELINES = True   # the module's 60/40 and equal-weight tracking MPO rows

RETURNS_FILE = PROJECT_ROOT / "Return_Forecasting" / "outputs" / "walk_forward_validation_expected_returns_h1_h3.csv"
COV_FILE = PROJECT_ROOT / "Covariance_Estimation" / "outputs" / "covariances_ewma_lambda_0_93_walk_forward_validation.csv"


# ---------------- locate modules and data ----------------
def find_optimizer_dir():
    env = os.environ.get("MPO_OPTIMIZER_DIR")
    candidates = [Path(env)] if env else []
    candidates += [PROJECT_ROOT / "CVaR MPO and Scenarios", PROJECT_ROOT / "Portfolio Optimization"]
    for folder in candidates:
        if (folder / "Multi_Period_Baselines.py").exists():
            return folder
    found = sorted(PROJECT_ROOT.rglob("Multi_Period_Baselines.py"))
    if found:
        return found[0].parent
    raise FileNotFoundError("Multi_Period_Baselines.py not found; set MPO_OPTIMIZER_DIR.")


OPT_DIR = find_optimizer_dir()
sys.path.insert(0, str(OPT_DIR))
import Multi_Period_Baselines as mb
import cvar_mpo as mc

ASSETS = list(mc.ASSETS)


def find_validation_csv():
    for folder in [PROJECT_ROOT / "HMM_regime_forecasts", PROJECT_ROOT / "Data" / "datasets",
                   PROJECT_ROOT / "Data", HERE, *HERE.parents]:
        if (folder / "portfolio_validation.csv").exists():
            return folder / "portfolio_validation.csv"
    found = sorted(PROJECT_ROOT.rglob("portfolio_validation.csv"))
    if found:
        return found[0]
    raise FileNotFoundError("portfolio_validation.csv not found.")


# ---------------- load handoffs ----------------
def load_origins(validation):
    ret = pd.read_csv(RETURNS_FILE, parse_dates=["rebalance_date", "information_date"])
    cov = pd.read_csv(COV_FILE, parse_dates=["rebalance_date"])
    for col in ASSETS:
        if col not in ret.columns:
            raise ValueError(f"Return handoff missing column {col}")
    if not set(ret.rebalance_date.unique()) == set(cov.rebalance_date.unique()):
        raise ValueError("Return and covariance handoffs have different origins")
    months = validation.index.to_period("M")
    origins = []
    for reb in sorted(ret.rebalance_date.unique()):
        r = ret[(ret.rebalance_date == reb) & (ret.month_ahead.isin(HORIZONS))].sort_values("month_ahead")
        if list(r.month_ahead) != list(HORIZONS):
            raise ValueError(f"Origin {reb.date()} lacks month_ahead {HORIZONS} in returns")
        means = r[ASSETS].to_numpy(float)
        covs = []
        for k, fm in zip(HORIZONS, r.forecast_month.astype(str)):
            c = cov[(cov.rebalance_date == reb) & (cov.month_ahead == k)]
            if len(c) != 3 or set(c.forecast_month.astype(str)) != {fm}:
                raise ValueError(f"Covariance handoff mismatch at {reb.date()} month_ahead {k}")
            covs.append(c.set_index("asset").loc[ASSETS, ASSETS].to_numpy(float))
        covs = np.array(covs)
        period = reb.to_period("M")
        days = validation[months == period]
        if days.empty or days.index[0] < reb:
            raise ValueError(f"No validation days for origin {reb.date()}")
        if not (r.information_date < days.index[0]).all():
            raise ValueError(f"Information date not before first trading day at {reb.date()}")
        stage_days = []
        for k, fm in enumerate(r.forecast_month.astype(str)):
            if k == 0:
                stage_days.append(len(days))
            else:
                start = pd.Period(fm).start_time.date()
                end = (pd.Period(fm) + 1).start_time.date()
                stage_days.append(int(np.busday_count(start, end)))
        stage_days = np.array(stage_days, dtype=int)
        rf = 0.0 / TRADING_DAYS * stage_days        # rf=0 inside targets; Sharpe reporting still uses RF_ANNUAL
        origins.append({"date": reb, "days": days, "means": means, "covs": covs,
                        "stage_days": stage_days, "rf": rf, "methods": r[[c for c in r.columns if c.startswith("method_")]].iloc[0].to_dict()})
    return origins, ret


# ---------------- Gaussian scenarios, moment-matched ----------------
def gaussian_scenarios(means, covs, seed):
    H = len(means)
    for attempt in range(20):
        rng = np.random.default_rng(seed + 1_000_003 * attempt)
        out = np.empty((N_SCENARIOS, H, 3))
        for k in range(H):
            z = rng.standard_normal((N_SCENARIOS, 3))
            zc = z - z.mean(axis=0)
            Lz = np.linalg.cholesky(np.cov(zc, rowvar=False))
            white = np.linalg.solve(Lz, zc.T).T           # sample covariance = identity
            vals, vecs = np.linalg.eigh((covs[k] + covs[k].T) / 2)
            A = vecs * np.sqrt(np.clip(vals, 0., None))    # A A' = covariance
            out[:, k] = means[k] + white @ A.T
        if (out > -1).all():
            return out
    raise RuntimeError("Could not draw scenarios with all returns above -100%")


# ---------------- planners ----------------
MV_NAMES = [n for n in mb.BASELINES if
            (INCLUDE_RISK_PARITY or not n.startswith("Risk parity")) and
            (INCLUDE_TRACKING_BASELINES or n not in ("60/40 (tracking)", "Equal weight (tracking)"))]
CVAR_MATCHED, CVAR_MIN = "CVaR (matched return)", "CVaR (minimum)"
MODELS = MV_NAMES + ([CVAR_MATCHED, CVAR_MIN] if INCLUDE_CVAR else [])


def plan_mv(name, o, H, w_prev):
    means, covs, rf = o["means"][:H], o["covs"][:H], o["rf"][:H]
    capped, builder = mb.BASELINES[name]
    caps = np.array([1., 1., BTC_CAP if capped else 1.])
    if name in (mb.MATCHED_SHARPE_BASELINE, "Max Sharpe (tracking)"):
        tangency = np.array([mb.max_sharpe_weights(m, S, caps, risk_free=r)
                             for m, S, r in zip(means, covs, rf)])
        if name == mb.MATCHED_SHARPE_BASELINE:
            settings = {"return_reward": 0., "min_return": (means * tangency).sum(axis=1) - 1e-9}
        else:
            settings = {"return_reward": 0., "target_weights": tangency}
    else:
        settings = builder(means, covs, caps)
    return mb.solve_mv_plan(covs, w_prev, expected_returns=means,
                            risk_aversion=mb.DEFAULT_RISK_AVERSION, cost_rate=COST_RATE,
                            upper=caps, **settings)


def plan_cvar(name, o, H, w_prev):
    scen = o["scenarios"][:, :H]
    upper = (1., 1., BTC_CAP)
    if name == CVAR_MATCHED:
        return mc.optimize_matched_cvar(scen, w_prev, o["stage_days"][:H], upper=upper,
                                        risk_free=o["rf"][:H], risk_aversion=CVAR_RISK_AVERSION,
                                        cost_rate=COST_RATE, alpha=CVAR_ALPHA)
    return mc.solve_cvar_plan(scen, w_prev, return_reward=0., risk_aversion=CVAR_RISK_AVERSION,
                              cost_rate=COST_RATE, alpha=CVAR_ALPHA, upper=upper)


# ---------------- metrics ----------------
def metrics(r, fee_total):
    a = r.to_numpy(float)
    n = len(a)
    years = n / TRADING_DAYS
    wealth = np.cumprod(1 + a)
    peak = np.maximum.accumulate(np.r_[1., wealth])[1:]
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    sd = a.std(ddof=1)
    return {
        "Net CAGR (%)": (wealth[-1] ** (1 / years) - 1) * 100,
        "Annual volatility (%)": sd * np.sqrt(TRADING_DAYS) * 100,
        "Annual Sharpe": (a.mean() - RF_ANNUAL / TRADING_DAYS) / sd * np.sqrt(TRADING_DAYS),
        "Daily 95% VaR (%)": mc.loss_var_cvar(a, .95)[0] * 100,
        "Monthly 95% VaR (%)": mc.loss_var_cvar(monthly.to_numpy(), .95)[0] * 100,
        "Max drawdown (%)": (1 - wealth / peak).max() * 100,
        "Annual transaction costs (%)": fee_total / years * 100,
        "Total return (%)": (wealth[-1] - 1) * 100,
        "Daily observations": n,
    }


# ---------------- strategy runners ----------------
def run_sequence(origins, plan_fn):
    w_prev, parts, fees, turn, wrows = np.zeros(3), [], [], [], []
    for o in origins:
        plan = plan_fn(o, w_prev)
        ex = mb.execute_first_period(plan, o["days"].to_numpy(float))
        if w_prev.sum() > 0:
            turn.append(0.5 * np.abs(plan.first_weights - w_prev).sum())
        fees.append(ex.fee_fraction)
        parts.append(pd.Series(ex.net_daily_returns, index=o["days"].index))
        wrows.append(plan.first_weights)
        w_prev = ex.ending_weights
    return pd.concat(parts), float(np.sum(fees)), (float(np.mean(turn)) if turn else 0.), np.array(wrows)


def fixed_plan(target):
    t = np.asarray(target, float)
    return lambda o, w_prev: SimpleNamespace(first_weights=t, pretrade_weights=w_prev,
                                             cost_rates=np.full(3, COST_RATE))


def main():
    t0 = time.perf_counter()
    val_path = find_validation_csv()
    validation = pd.read_csv(val_path, index_col=0, parse_dates=True).sort_index()[ASSETS]
    origins, ret_raw = load_origins(validation)
    for i, o in enumerate(origins):
        if INCLUDE_CVAR:
            o["scenarios"] = gaussian_scenarios(o["means"], o["covs"], SEED + i)
    print(f"Optimizer dir: {OPT_DIR}\nValidation: {val_path}\nOrigins: {len(origins)} "
          f"({origins[0]['date'].date()} to {origins[-1]['date'].date()}) | "
          f"daily obs {sum(len(o['days']) for o in origins)}", flush=True)

    rows, daily, weight_rows = [], {}, []

    def record(label, H, r, fees, turn, w):
        m = metrics(r, fees)
        avg_w = None if w is None else w.mean(axis=0) * 100
        m.update({"Strategy": label, "H": H, "Avg monthly turnover (%)": turn * 100,
                  "Avg Equity (%)": np.nan if avg_w is None else avg_w[0],
                  "Avg Bond (%)": np.nan if avg_w is None else avg_w[1],
                  "Avg BTC (%)": np.nan if avg_w is None else avg_w[2]})
        rows.append(m)
        daily[f"{label} | H={H}" if H else label] = r

    # Simple rows (no optimizer)
    if INCLUDE_FIXED_REBALANCE:
        for label, tgt in (("60/40 (monthly rebalanced)", [.6, .4, 0.]),
                           ("Equal weight (monthly rebalanced)", [1 / 3] * 3)):
            r, f, t, w = run_sequence(origins, lambda o, wp, tgt=tgt: fixed_plan(tgt)(o, wp))
            record(label, 0, r, f, t, w)
    all_days = validation.loc[validation.index >= origins[0]["days"].index[0]]
    all_days = all_days[all_days.index <= origins[-1]["days"].index[-1]]
    bh_plan = fixed_plan([1 / 3] * 3)(None, np.zeros(3))
    bh = mb.execute_first_period(bh_plan, all_days.to_numpy(float))
    record("Buy-and-hold (1/3 each)", 0, pd.Series(bh.net_daily_returns, index=all_days.index),
           bh.fee_fraction, 0., None)

    # Optimizer rows
    for H in HORIZONS:
        for name in MODELS:
            t1 = time.perf_counter()
            fn = (lambda o, wp, n=name, h=H: plan_cvar(n, o, h, wp)) if name in (CVAR_MATCHED, CVAR_MIN) \
                else (lambda o, wp, n=name, h=H: plan_mv(n, o, h, wp))
            try:
                r, f, t, w = run_sequence(origins, fn)
            except (RuntimeError, ValueError, AssertionError) as error:
                raise RuntimeError(f"{name}, H={H}: {error}") from error
            record(name, H, r, f, t, w)
            for o, wv in zip(origins, w):
                weight_rows.append({"Strategy": name, "H": H, "Origin": str(o["date"].date()),
                                    "Equity": wv[0], "Bond": wv[1], "BTC": wv[2]})
            print(f"H={H} {name}: {time.perf_counter() - t1:.1f}s", flush=True)

    by_h = pd.DataFrame(rows)
    order = list(dict.fromkeys(by_h.Strategy))
    cols = ["Net CAGR (%)", "Annual volatility (%)", "Annual Sharpe", "Daily 95% VaR (%)",
            "Monthly 95% VaR (%)", "Max drawdown (%)", "Annual transaction costs (%)"]
    extra = ["Total return (%)", "Avg monthly turnover (%)", "Avg Equity (%)", "Avg Bond (%)", "Avg BTC (%)"]
    summary = by_h.groupby("Strategy", sort=False)[cols + extra].mean().reindex(order)

    by_h.to_csv(OUT / "non_regime_mpo_by_horizon.csv", index=False)
    summary.to_csv(OUT / "non_regime_mpo_summary.csv")
    pd.DataFrame(weight_rows).to_csv(OUT / "non_regime_mpo_weights.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "non_regime_mpo_daily_returns.csv")
    meta = {"returns_file": str(RETURNS_FILE), "covariance_file": str(COV_FILE),
            "risk_free_annual": RF_ANNUAL, "risk_free_inside_max_sharpe_targets": False,
            "horizons": list(HORIZONS), "cost_rate": COST_RATE, "btc_cap": BTC_CAP,
            "mv_risk_aversion": mb.DEFAULT_RISK_AVERSION, "cvar_risk_aversion": CVAR_RISK_AVERSION,
            "cvar_scenarios": f"{N_SCENARIOS} Gaussian, whitened to match forecast mean/covariance, fixed seed {SEED}+origin index",
            "return_methods_first_origin": origins[0]["methods"],
            "transaction_cost_definition": "sum of self-financing fee fractions incl. initial purchase / (daily obs/252)",
            "monthly_var": "interpolated 95% loss VaR of compounded calendar-month net returns",
            "test_set_used": False}
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\nNON-REGIME MPO TABLE (validation; MPO rows averaged over H=1-3; Sharpe at 4% rf)")
    print(summary[cols + ["Avg monthly turnover (%)"]].round(2).to_string())
    print("\nAverage executed weights (%):")
    print(summary[["Avg Equity (%)", "Avg Bond (%)", "Avg BTC (%)"]].round(1).to_string())
    print(f"\nSaved to {OUT}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()