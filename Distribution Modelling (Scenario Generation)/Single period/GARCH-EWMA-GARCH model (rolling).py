"""Monthly rolling walk-forward GARCH/FHS validation for the capstone.

At the FIRST available trading day of each VALIDATION calendar month:
  1. Fit each asset's candidate volatility model using ONLY the prior 5 years.
  2. Rebuild the aligned empirical joint-shock pool from that same window.
  3. Generate 21-trading-day joint return scenarios for later CVaR optimization.
  4. Forecast realized risk over that calendar month's available trading days.
  5. Update daily variance forecasts using daily returns as they become known;
     fitted parameters are updated only at the NEXT monthly rebalance.

Past VALIDATION returns may enter LATER estimation windows. No future
validation returns, and no TEST returns, enter any validation forecast.

INPUTS: portfolio_train.csv, portfolio_validation.csv (date index; aligned
         asset columns; SIMPLE daily returns as decimals).

OUTPUTS: rolling_model_diagnostics.csv, rolling_model_comparison.csv,
         rolling_daily_var.csv, rolling_monthly_var.csv,
         rolling_scenarios_21day.npz, optionally rolling_scenario_paths_21day.npy,
         and, separately, scenarios_horizon_returns.csv / scenario_paths.npy
         refitted as of the END of validation for a possible next period.

Run: python "Multiple GARCH models (rolling).py"
Requires: numpy, pandas, scipy, arch, statsmodels
"""

# Support direct execution after organizing the source folders.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_MODELLING_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents
                      if (p / "project_paths.py").is_file())
if str(_MODELLING_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_MODELLING_ROOT))
from project_paths import (PROJECT_ROOT, MODELLING_ROOT, source_path,
                           cache_source_bytes, cache_source_name)

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# ====================== USER SETTINGS ============================
HORIZON = 21                 # fixed-horizon CVaR scenarios at EACH rebalance
ROLLING_YEARS = 5            # each fit uses past five calendar years
MIN_WINDOW_OBS = 750         # at least ~3 years of observations required
N_SCENARIOS = 10_000         # selected-model scenarios saved for CVaR
N_BACKTEST = 3_000          # candidate-model scenarios for each realized month
SEED = 42
TRADING_DAYS = 252
VAR_TAIL_PROB = 0.05
EWMA_LAMBDA = 0.94
VOL_CAP_MULTIPLE = 1.5       # None disables the simulated-volatility cap
DIAGNOSTIC_LAGS = 10

# Start with ("GARCH",) if you want only the original model to run.
# Leave all enabled for like-for-like EWMA/GARCH/GJR/AR(1)-GARCH comparison.
CANDIDATE_MODELS = ("GARCH", "EWMA")

SELECTED_MODEL_BY_ASSET = {
    "Equity": "GARCH",
    "Bond": "EWMA",
    "BTC": "GARCH"
}
CHECK_WEIGHTS = None          # None: equal weight; can provide a list
SAVE_ROLLING_DAILY_PATHS = True  # ~70 MB for 28 months, 10k paths, 3 assets

# Must be chosen using information KNOWN at each rebalance if changed.
# None = the trailing-window historical average for each asset.
EXPECTED_RETURN_ANNUAL = None  # e.g. {"Equity": .07, "Bond": .03, "BTC": .15}
PROJECT_ROOT = MODELLING_ROOT.parent
DATA_DIR = PROJECT_ROOT / "Data" / "datasets"
OUTPUT = PROJECT_ROOT / "outputs" / "volatility"
OUTPUT.mkdir(parents=True, exist_ok=True)
print(f"Output filenames use the prefix: {Path(__file__).stem}_")
# =================================================================


def load_split(name):
    path = DATA_DIR / f"portfolio_{name}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty or df.isna().any().any():
        raise ValueError(f"{path.name} is empty or has missing data")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} needs unique, ascending dates")
    if not np.isfinite(df.to_numpy(float)).all() or (df.to_numpy(float) <= -1).any():
        raise ValueError(f"{path.name} has invalid simple daily returns")
    return df.astype(float)


train = load_split("train")
validation = load_split("validation")
assets = list(train.columns)
n = len(assets)
if len(assets) < 2:
    raise ValueError("Provide at least two aligned asset return series")
if set(validation.columns) != set(assets):
    raise ValueError("Training and validation columns must match")
validation = validation[assets]
if train.index[-1] >= validation.index[0]:
    raise ValueError("Training and validation dates overlap")
if EXPECTED_RETURN_ANNUAL is not None and set(EXPECTED_RETURN_ANNUAL) != set(assets):
    raise ValueError("EXPECTED_RETURN_ANNUAL must include all asset columns")
selected_kind = [SELECTED_MODEL_BY_ASSET.get(a, "GARCH") for a in assets]
if not all(kind in CANDIDATE_MODELS for kind in selected_kind):
    raise ValueError("Selected model must be included in CANDIDATE_MODELS")
weights = (np.full(n, 1 / n) if CHECK_WEIGHTS is None else
           np.asarray(CHECK_WEIGHTS, dtype=float))
if len(weights) != n or np.any(~np.isfinite(weights)) or not np.isclose(weights.sum(), 1):
    raise ValueError("CHECK_WEIGHTS must have one finite weight per asset and sum to 1")
if (weights < 0).any():
    warnings.warn("Portfolio weights include shorts; ensure the return calculation suits your constraints")

# Data may include future validation values in this array, but slices below
# are ALWAYS selected with the strict condition date < forecast date.
all_returns = pd.concat([train, validation])
if all_returns.index.has_duplicates:
    raise ValueError("Duplicate dates across concatenated datasets")


def training_window(rebalance_date):
    """Strictly historical trailing five *calendar* years."""
    cutoff = rebalance_date - pd.DateOffset(years=ROLLING_YEARS)
    data = all_returns.loc[(all_returns.index >= cutoff) &
                           (all_returns.index < rebalance_date)]
    if len(data) < MIN_WINDOW_OBS:
        raise ValueError(f"Only {len(data)} observations preceding {rebalance_date.date()}")
    assert data.index.max() < rebalance_date  # look-ahead guard
    return data


def fit_asset(returns_pct, kind, asset):
    """Fit one model on one rolling window (returns are in PERCENT)."""
    series = returns_pct[asset]
    values = series.to_numpy(float)
    sample_mean = float(series.mean())
    if kind == "EWMA":
        residuals = values - sample_mean
        variance = np.empty(len(values))
        variance[0] = max(float(np.var(values[:min(60, len(values))], ddof=1)), 1e-8)
        for t in range(1, len(values)):
            variance[t] = (EWMA_LAMBDA * variance[t - 1] +
                           (1 - EWMA_LAMBDA) * residuals[t - 1] ** 2)
        return dict(kind=kind, asset=asset, mu=sample_mean, phi=0.,
                    sample_mean=sample_mean, omega=0., alpha=1 - EWMA_LAMBDA,
                    beta=EWMA_LAMBDA, gamma=0., nu=np.nan, skew=np.nan,
                    convergence=0, aic=np.nan, bic=np.nan,
                    resid=pd.Series(residuals, index=series.index),
                    variance=pd.Series(variance, index=series.index),
                    std_resid=pd.Series(residuals / np.sqrt(variance), index=series.index))

    if kind not in ("GARCH", "GJR-GARCH", "AR1-GARCH"):
        raise ValueError(f"Unknown volatility model: {kind}")
    is_ar = kind == "AR1-GARCH"
    model = arch_model(series, mean="AR" if is_ar else "Constant",
                       lags=1 if is_ar else 0,
                       vol="GARCH", p=1, o=1 if kind == "GJR-GARCH" else 0,
                       q=1, dist="skewt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp="off")
    p = result.params
    if is_ar:
        lag_keys = [key for key in p.index if key.endswith("[1]") and
                    key not in ("alpha[1]", "beta[1]", "gamma[1]")]
        if len(lag_keys) != 1:
            raise RuntimeError(f"Unexpected AR parameters for {asset}: {p.index.tolist()}")
        phi = float(p[lag_keys[0]])
    else:
        phi = 0.
    if result.convergence_flag != 0:
        warnings.warn(f"{kind} {asset} did not fully converge on window ending {series.index[-1].date()}")
    return dict(kind=kind, asset=asset,
                mu=float(p["Const"] if is_ar else p["mu"]), phi=phi,
                sample_mean=sample_mean, omega=float(p["omega"]),
                alpha=float(p["alpha[1]"]), beta=float(p["beta[1]"]),
                gamma=float(p["gamma[1]"]) if kind == "GJR-GARCH" else 0.,
                nu=float(p["eta"]), skew=float(p["lambda"]),
                convergence=int(result.convergence_flag),
                aic=float(result.aic), bic=float(result.bic),
                resid=result.resid, variance=result.conditional_volatility ** 2,
                std_resid=result.std_resid)


def fitted_mean(state, last_return_pct):
    """Mean used to FILTER actual returns and update the conditional variance."""
    return state["mu"] + state["phi"] * last_return_pct


def simulated_mean(state, last_return_pct):
    """Rolling sample-mean baseline, with AR(1) adjustment where applicable."""
    if state["kind"] == "AR1-GARCH":
        if abs(state["phi"]) < 0.999:
            fitted_unconditional_mean = state["mu"] / (1 - state["phi"])
        else:
            fitted_unconditional_mean = state["sample_mean"]
        base = fitted_mean(state, last_return_pct) + (state["sample_mean"] - fitted_unconditional_mean)
    else:
        base = state["sample_mean"]
    if EXPECTED_RETURN_ANNUAL is not None:
        desired_mean_pct = EXPECTED_RETURN_ANNUAL[state["asset"]] / TRADING_DAYS * 100
        base = base + (desired_mean_pct - state["sample_mean"])
    return base


def next_variance(state, shock_pct, var_pct2):
    if state["kind"] == "EWMA":
        return (EWMA_LAMBDA * var_pct2 + (1 - EWMA_LAMBDA) * shock_pct ** 2)
    return (state["omega"] +
            (state["alpha"] + state["gamma"] * (shock_pct < 0)) * shock_pct ** 2 +
            state["beta"] * var_pct2)


def start_variance(state):
    eps = float(state["resid"].dropna().iloc[-1])
    v = float(state["variance"].dropna().iloc[-1])
    answer = float(next_variance(state, eps, v))
    if not np.isfinite(answer) or answer <= 0:
        raise ValueError(f"Nonpositive/invalid starting forecast variance: {state['asset']}, {state['kind']}")
    return answer


def aligned_shock_pool(states):
    z = pd.concat([s["std_resid"].rename(s["asset"]) for s in states], axis=1)
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 100:
        raise ValueError("Too few aligned standardized residual days")
    z = (z - z.mean()) / z.std(ddof=1)
    return z.to_numpy(float)


def variance_caps(states):
    if VOL_CAP_MULTIPLE is None:
        return np.full(len(states), np.inf)
    return np.array([(VOL_CAP_MULTIPLE * np.sqrt(s["variance"].max())) ** 2
                     for s in states])


def simulate(states, pool, starting_variances, preceding_returns, paths_count, horizon, rng):
    """Simulate daily simple returns, resampling JOINT historical shocks."""
    n_assets = len(states)
    var = np.broadcast_to(np.asarray(starting_variances, float),
                          (paths_count, n_assets)).copy()
    previous = np.broadcast_to(np.asarray(preceding_returns, float),
                               (paths_count, n_assets)).copy()
    caps = variance_caps(states)
    paths = np.empty((paths_count, horizon, n_assets), dtype=np.float32)
    cap_hits, clips = 0, 0
    for t in range(horizon):
        z = pool[rng.integers(0, len(pool), size=paths_count)]
        eps = np.sqrt(var) * z
        for j, s in enumerate(states):
            simulated_return_pct = simulated_mean(s, previous[:, j]) + eps[:, j]
            clips += int(np.sum(simulated_return_pct < -99.))
            simulated_return_pct = np.maximum(simulated_return_pct, -99.)
            paths[:, t, j] = simulated_return_pct / 100
            previous[:, j] = simulated_return_pct
            var[:, j] = next_variance(s, eps[:, j], var[:, j])
        cap_hits += int((var > caps).sum())
        var = np.minimum(var, caps)
    return paths, cap_hits / (paths_count * horizon * n_assets), clips


def compounded(paths):
    return np.prod(1. + paths.astype(float), axis=1) - 1.


def risk_details(simulated_returns, realized_returns, label, date, days):
    """Input shapes: scenarios x assets, real assets; include fixed-weight portfolio."""
    joint = np.column_stack([simulated_returns, simulated_returns @ weights])
    actual = np.append(realized_returns, realized_returns @ weights)
    rows = []
    for j, asset in enumerate(assets + ["Portfolio"]):
        q = float(np.quantile(joint[:, j], VAR_TAIL_PROB))
        es = float(joint[joint[:, j] <= q, j].mean())
        rows.append(dict(Model=label, Date=date, Asset=asset,
                         **{"Holding days":days, "Realized return":float(actual[j]),
                            "VaR 5% return threshold":q, "Predicted ES return":es,
                            "VaR exceedance":int(actual[j] < q)}))
    return rows


def ljung_box_p(x, squared=False):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 * DIAGNOSTIC_LAGS:
        return np.nan
    if squared:
        x = x ** 2
    return float(acorr_ljungbox(x, lags=[DIAGNOSTIC_LAGS],
                               return_df=True)["lb_pvalue"].iloc[0])


def arch_lm_p(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(het_arch(x, nlags=DIAGNOSTIC_LAGS)[1]) if len(x) > 2 * DIAGNOSTIC_LAGS else np.nan


def model_diagnostic(state, date, window):
    z = state["std_resid"].dropna()
    return {"Rebalance date":date, "Training start":window.index[0],
            "Training end":window.index[-1], "Training observations":len(window),
            "Model":state["kind"], "Asset":state["asset"],
            "Convergence flag":state["convergence"],
            "omega":state["omega"], "alpha":state["alpha"],
            "beta":state["beta"], "gamma":state["gamma"],
            "Persistence (approx.)":state["alpha"] + state["beta"] + state["gamma"] / 2,
            "AR1 phi":state["phi"], "Tail df (nu)":state["nu"],
            "Skew (lambda)":state["skew"], "AIC":state["aic"], "BIC":state["bic"],
            "Std residuals LB p":ljung_box_p(z),
            "Squared std residuals LB p":ljung_box_p(z, squared=True),
            "Std residual ARCH-LM p":arch_lm_p(z)}


def daily_backtest(states, pool, window_returns, date_index, label, date):
    """Predict each day in this month, updating variance AFTER each observed return."""
    variances = np.array([start_variance(s) for s in states])
    previous = (window_returns.iloc[-1].to_numpy() * 100).copy()
    rows, errors = [], []
    for t, now in enumerate(date_index):
        # Forecast today's distribution BEFORE consuming today's outcome.
        means = np.array([simulated_mean(s, previous[j]) for j, s in enumerate(states)])
        sims = np.maximum((means[None, :] + np.sqrt(variances)[None, :] * pool) / 100, -0.99)
        observed = (validation.loc[now, assets] * 100).to_numpy(float)
        rows.extend(risk_details(sims, observed / 100, label, now, 1))
        for j, s in enumerate(states):
            # For fair volatility scoring and variance updating, residuals
            # are based on the FITTED conditional mean, not scenario drift.
            eps = observed[j] - fitted_mean(s, previous[j])
            var = max(float(variances[j]), 1e-10)
            errors.append({"Model":label, "Date":now, "Asset":assets[j],
                           "QLIKE":float(np.log(var) + eps ** 2 / var),
                           "Variance MSE":float((eps ** 2 - var) ** 2)})
            variances[j] = float(next_variance(s, eps, var))
            if not np.isfinite(variances[j]) or variances[j] <= 0:
                raise ValueError(f"Invalid filtered variance for {label} {assets[j]} on {now}")
        # Avoid hiding a volatility regime shift by CAPPING filtered actual
        # volatility. Caps affect simulated future paths only.
        previous = observed
    return rows, errors


def kupiec_p(n_obs, n_hits):
    k, N = int(n_hits), int(n_obs)
    if N == 0:
        return np.nan
    phat = k / N
    ln0 = k * np.log(VAR_TAIL_PROB) + (N-k) * np.log(1-VAR_TAIL_PROB)
    ln1 = ((k * np.log(phat)) if k else 0.) + (
        (N-k) * np.log(1-phat) if k < N else 0.)
    return float(chi2.sf(max(0., 2 * (ln1-ln0)), 1))


# Calendar-month starts (rather than skipping forward 21 observations).
month_groups = [(period, group.index) for period, group in
                validation.groupby(validation.index.to_period("M"), sort=True)]
if not month_groups:
    raise ValueError("No validation months found")

print(f"Rolling {ROLLING_YEARS}-year refits; {len(month_groups)} calendar months; "
      f"{len(validation)} validation days")
print("Each rebalance uses only dates BEFORE its first trading day.")
print("Candidates:", CANDIDATE_MODELS, "| CVaR selection:", dict(zip(assets, selected_kind)))

all_diag, all_daily, all_monthly, all_errors = [], [], [], []
selected_rebalance_dates = []
rolling_scenarios = []
rolling_paths = []
selected_cap_records = []

for i, (period, days) in enumerate(month_groups):
    date = days[0]
    window = training_window(date)
    returns_pct = window * 100
    model_states = {}
    for kind in CANDIDATE_MODELS:
        states = [fit_asset(returns_pct, kind, asset) for asset in assets]
        model_states[kind] = states
        all_diag.extend(model_diagnostic(s, date, window) for s in states)

    last_returns_pct = returns_pct.iloc[-1].to_numpy()
    real_period = np.prod(1 + validation.loc[days, assets].to_numpy(), axis=0) - 1
    for j, kind in enumerate(CANDIDATE_MODELS):
        states = model_states[kind]
        pool = aligned_shock_pool(states)
        daily_rows, error_rows = daily_backtest(states, pool, window, days, kind, date)
        all_daily.extend(daily_rows)
        all_errors.extend(error_rows)
        sim, _, _ = simulate(states, pool, [start_variance(s) for s in states],
                             last_returns_pct, N_BACKTEST, len(days),
                             np.random.default_rng(SEED + 100_003*i + j))
        all_monthly.extend(risk_details(compounded(sim), real_period, kind, date, len(days)))

    # The mixed/selected model is also estimated FROM SCRATCH each month.
    selected_states = [model_states[k][j] for j, k in enumerate(selected_kind)]
    selected_pool = aligned_shock_pool(selected_states)
    selected_path, hit_rate, clipped = simulate(
        selected_states, selected_pool, [start_variance(s) for s in selected_states],
        last_returns_pct, N_SCENARIOS, HORIZON,
        np.random.default_rng(SEED + 1_000_003*i))
    selected_rebalance_dates.append(str(date.date()))
    rolling_scenarios.append(compounded(selected_path).astype(np.float32))
    if SAVE_ROLLING_DAILY_PATHS:
        rolling_paths.append(selected_path)
    selected_cap_records.append({"Rebalance date":date,
                                 "Cap hit share of asset-days":hit_rate,
                                 "Daily -99% clips":clipped})

    # Report risk over the ACTUAL calendar-month holding period. If the
    # month has 19 or 23 observed days, the simulations use that many days.
    if len(set(selected_kind)) > 1:
        sim, _, _ = simulate(selected_states, selected_pool,
                             [start_variance(s) for s in selected_states],
                             last_returns_pct, N_BACKTEST, len(days),
                             np.random.default_rng(SEED + 2_000_003*i))
        all_monthly.extend(risk_details(compounded(sim), real_period,
                                        "Selected mixed", date, len(days)))
        d, errors = daily_backtest(selected_states, selected_pool, window, days,
                                   "Selected mixed", date)
        all_daily.extend(d)
        all_errors.extend(errors)

    print(f" {i+1:2d}/{len(month_groups)} | {period} | ")
    print(f"    window {window.index[0].date()} to {window.index[-1].date()} "
          f"({len(window)} obs) -> holding {len(days)} trading days")

# Different forecast vintages must remain DISTINCT. Each row of the array
# is what was available at THAT month, before actual monthly returns.
rolling_returns = np.stack(rolling_scenarios)
np.savez_compressed(OUTPUT / f"{Path(__file__).stem}_rolling_scenarios_21day.npz",
                    rebalance_dates=np.asarray(selected_rebalance_dates),
                    assets=np.asarray(assets), returns=rolling_returns)
if SAVE_ROLLING_DAILY_PATHS:
    np.save(OUTPUT / f"{Path(__file__).stem}_rolling_scenario_paths_21day.npy", np.stack(rolling_paths))

rolling_diagnostics = pd.DataFrame(all_diag)
rolling_daily = pd.DataFrame(all_daily)
rolling_monthly = pd.DataFrame(all_monthly)
rolling_errors = pd.DataFrame(all_errors)
rolling_diagnostics.to_csv(OUTPUT / f"{Path(__file__).stem}_rolling_model_diagnostics.csv", index=False)
rolling_daily.to_csv(OUTPUT / f"{Path(__file__).stem}_rolling_daily_var.csv", index=False)
rolling_monthly.to_csv(OUTPUT / f"{Path(__file__).stem}_rolling_monthly_var.csv", index=False)
rolling_errors.to_csv(OUTPUT / f"{Path(__file__).stem}_rolling_volatility_errors.csv", index=False)
pd.DataFrame(selected_cap_records).to_csv(OUTPUT / f"{Path(__file__).stem}_rolling_cap_hits.csv", index=False)

# Summarize forecasts across all validation days and calendar months.
comparison_rows = []
for (kind, asset), daily_df in rolling_daily.groupby(["Model", "Asset"], sort=False):
    monthly_df = rolling_monthly.loc[(rolling_monthly.Model == kind) &
                                    (rolling_monthly.Asset == asset)]
    daily_hits = int(daily_df["VaR exceedance"].sum())
    monthly_hits = int(monthly_df["VaR exceedance"].sum())
    df_errors = rolling_errors.loc[(rolling_errors.Model == kind) &
                                   (rolling_errors.Asset == asset)]
    comparison_rows.append({
        "Model":kind, "Asset":asset,
        "1d windows":len(daily_df), "1d VaR hits":daily_hits,
        "1d exceedance rate":daily_hits/len(daily_df),
        "1d Kupiec p":kupiec_p(len(daily_df), daily_hits),
        "1d QLIKE (lower better)":df_errors.QLIKE.mean() if len(df_errors) else np.nan,
        "1d variance MSE (lower better)":df_errors["Variance MSE"].mean() if len(df_errors) else np.nan,
        "Monthly windows":len(monthly_df), "Monthly VaR hits":monthly_hits,
        "Monthly exceedance rate":monthly_hits / len(monthly_df),
        "Monthly Kupiec p (LOW POWER)":kupiec_p(len(monthly_df), monthly_hits),
        "Avg predicted monthly ES (loss %)":-100*monthly_df["Predicted ES return"].mean(),
        "Realized monthly loss on exceptions (%)":(
            -100*monthly_df.loc[monthly_df["VaR exceedance"] == 1,"Realized return"].mean()
            if monthly_hits else np.nan),
        "Predicted monthly ES on exception dates (%)":(
            -100*monthly_df.loc[monthly_df["VaR exceedance"] == 1,"Predicted ES return"].mean()
            if monthly_hits else np.nan),
    })
comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(OUTPUT / f"{Path(__file__).stem}_rolling_model_comparison.csv", index=False)

# Optional visual reports relevant to the bond and Bitcoin investigations.
# These are NOT used in fitting or forecast selection.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "Bond" in assets:
        bond_vol = all_returns["Bond"].rolling(21).std() * np.sqrt(TRADING_DAYS) * 100
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(bond_vol.index, bond_vol, lw=1)
        ax.axvline(validation.index[0], color="tab:red", ls="--", label="Validation begins")
        ax.set(title="Bond 21-day rolling realized volatility", ylabel="Annualized volatility (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{Path(__file__).stem}_rolling_bond_volatility.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(n, 1, figsize=(11, max(3*n, 4)), sharex=True, squeeze=False)
    for j, asset in enumerate(assets):
        ax = axes[j, 0]
        for kind in ("GARCH", "GJR-GARCH", "AR1-GARCH"):
            subset = rolling_diagnostics.loc[
                (rolling_diagnostics["Asset"] == asset) &
                (rolling_diagnostics["Model"] == kind)]
            if not subset.empty:
                ax.plot(subset["Rebalance date"], subset["Persistence (approx.)"],
                        label=kind, marker=".", ms=4)
        ax.axhline(1.0, color="tab:red", ls="--", lw=1)
        ax.set(ylabel=f"{asset} persistence", title=f"{asset}: persistence across rolling refits")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{Path(__file__).stem}_rolling_garch_persistence.png", dpi=150)
    plt.close(fig)
except ImportError:
    print("matplotlib not installed: skipped optional diagnostic plots")

# For compatibility with the ORIGINAL optimizer, also generate one 21-day
# forecast AS OF THE END OF VALIDATION. No test-set observations are used.
# Important: this is a *new forecast for the period after validation*, not a
# substitute for the historical rolling scenarios used in validation.
after_validation = validation.index[-1] + pd.offsets.BDay(1)
final_window = training_window(after_validation)
final_state_map = {}
for kind in set(selected_kind):
    final_state_map[kind] = [fit_asset(final_window*100, kind, a) for a in assets]
final_states = [final_state_map[k][j] for j,k in enumerate(selected_kind)]
final_pool = aligned_shock_pool(final_states)
final_paths, _, _ = simulate(final_states, final_pool,
    [start_variance(s) for s in final_states],
    (final_window.iloc[-1]*100).to_numpy(), N_SCENARIOS, HORIZON,
    np.random.default_rng(SEED + 9_999_999))
final_scenarios = pd.DataFrame(compounded(final_paths), columns=assets)
final_scenarios.index.name = "Scenario"
final_scenarios.to_csv(OUTPUT / f"{Path(__file__).stem}_scenarios_horizon_returns.csv")
np.save(OUTPUT / f"{Path(__file__).stem}_scenario_paths.npy", final_paths)

final_params = pd.DataFrame([{
    "Asset":s["asset"], "Model":s["kind"], "alpha":s["alpha"],
    "beta":s["beta"], "gamma":s["gamma"],
    "Persistence (approx.)":s["alpha"]+s["beta"]+s["gamma"]/2,
    "omega":s["omega"], "AR1 phi":s["phi"],
    "Estimation end":final_window.index[-1], "Convergence flag":s["convergence"]
} for s in final_states])
final_params.to_csv(OUTPUT / f"{Path(__file__).stem}_garch_parameters.csv", index=False)

print("\nROLLING VALIDATION MODEL COMPARISON (daily & actual calendar-month holding periods):")
cols = ["Model", "Asset", "1d VaR hits", "1d exceedance rate",
        "1d QLIKE (lower better)", "Monthly VaR hits", "Monthly exceedance rate"]
print(comparison[cols].round(4).to_string(index=False))
print("\nNear-unit persistence check (fraction of monthly GARCH fits >= 0.995):")
for (kind, asset), g in rolling_diagnostics.groupby(["Model", "Asset"]):
    if kind != "EWMA":
        print(f" {kind} {asset}: {(g['Persistence (approx.)'] >= 0.995).mean():.1%}; "
              f"range {g['Persistence (approx.)'].min():.4f}–{g['Persistence (approx.)'].max():.4f}")
print("\nREMEMBER: MONTHLY TESTS HAVE LOW POWER. Daily portfolio tests use a fixed-weight benchmark; ")
print("portfolio optimization itself must be run separately at EACH rebalance using that month's saved scenarios.")
print("Final single-period scenarios were refit as of", final_window.index[-1].date(),
      "using no test-set data.")
print("\nSaved: rolling_model_diagnostics.csv, rolling_model_comparison.csv, "
      "rolling_daily_var.csv, rolling_monthly_var.csv, rolling_volatility_errors.csv, "
      "rolling_cap_hits.csv, rolling_scenarios_21day.npz, "
      + ("rolling_scenario_paths_21day.npy, " if SAVE_ROLLING_DAILY_PATHS else "") +
      "scenarios_horizon_returns.csv, scenario_paths.npy, garch_parameters.csv")
