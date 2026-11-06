# ============================================================
# CAPSTONE PROJECT: JOINT DISTRIBUTION AND SCENARIO GENERATION
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# What this file does:
#   Builds a simulator for how the three assets move together, then uses it
#   to generate thousands of possible future periods ("scenarios") for the
#   CVaR optimizer.
#
# The method (filtered historical simulation with GARCH):
#   1. GARCH, one per asset: models how volatile each asset is right now
#      (calm or wild) and how that changes after big moves. Skewed-t
#      innovations give fat, asymmetric tails (no normal distribution).
#   2. Dividing each day's return by that day's fitted volatility leaves
#      "standardized shocks": was the day better or worse than expected,
#      given how wild things were.
#   3. Dependence: whole days of shocks (all three assets together) are
#      resampled from history. A crash day keeps its equity, bond and BTC
#      shocks together, so the assets crash together in the simulation
#      exactly as often as they did in the data.
#   4. Simulation: starting from today's volatility, draw a random day of
#      shocks, scale by the current volatility, update the volatility, and
#      repeat for HORIZON days. Compound to one return per asset. Do this
#      N_SCENARIOS times. Volatility is capped at VOL_CAP_MULTIPLE times the
#      highest level seen in the training data, so a run of bad draws cannot
#      send it to unrealistic levels.
#
# Inputs (from "Splitting data.py"):
#   portfolio_train.csv       models are fitted on this ONLY
#   portfolio_validation.csv  used only to check the simulator
#
# Outputs:
#   scenarios_horizon_returns.csv   N_SCENARIOS x 3 simulated HORIZON-day returns
#   scenario_paths.npy              N_SCENARIOS x HORIZON x 3 daily paths (for multi-period/MPC)
#   garch_parameters.csv            fitted model parameters
#
# How to use the scenarios in the CVaR optimizer:
#   Replace the historical daily returns with scenarios_horizon_returns.csv.
#   The CVaR is then a HORIZON-day number (about one month for 21 days), so
#   compare it only against other models measured over the same horizon, and
#   state any minimum-return target as a HORIZON-day return too.
#
# Where Black-Litterman plugs in:
#   Set EXPECTED_RETURN_ANNUAL below. The simulation then uses those returns
#   as each asset's average instead of the training-sample average (which is
#   what the baseline models use). The volatility, tails and dependence are
#   unchanged.
#
# Limits to keep in mind:
#   - Bitcoin trades every day but the other two do not, so each Monday's
#     return covers the weekend. GARCH treats all days as equally spaced.
#   - Only the training set is used to fit. In the real walk-forward the
#     models would be refit each month; here parameters are fixed.
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

HORIZON = 21           # trading days per scenario (about one month)
N_SCENARIOS = 10_000   # number of simulated scenarios
N_BACKTEST = 2_000     # scenarios per window in the validation check
SEED = 42              # makes the simulation repeatable
TRADING_DAYS = 252

# False = plain symmetric GARCH(1,1) (default). On the project's training data
#         it matched the historical equity tail better than the option below.
# True  = GJR-GARCH (bad days raise volatility more than good days). It
#         tends to give heavier equity tails here. Worth comparing both.
ASYMMETRIC = False

# Volatility in the simulation is not allowed to exceed this multiple of the
# highest volatility fitted on the training data.
VOL_CAP_MULTIPLE = 1.5

# Annual expected returns to use as the average (e.g. from Black-Litterman).
# None = use each asset's average daily return in the training data, the same
# sample average the baseline models use. Example:
# EXPECTED_RETURN_ANNUAL = {"Equity": 0.07, "Bond": 0.03, "BTC": 0.15}
EXPECTED_RETURN_ANNUAL = None

# Portfolio used for the validation check (buy-and-hold over each window).
# None = equal weight.
CHECK_WEIGHTS = None

output = Path(__file__).resolve().parent


# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------

def load(name):
    return pd.read_csv(output / f"portfolio_{name}.csv",
                       index_col=0, parse_dates=True).sort_index()


train, validation = load("train"), load("validation")
assets = list(train.columns)
n = len(assets)
validation = validation[assets]

# GARCH is fitted on returns in percent (better numerical behaviour).
R_train = train * 100
R_val = validation * 100

w_check = np.full(n, 1.0 / n) if CHECK_WEIGHTS is None else np.asarray(CHECK_WEIGHTS, float)


# ------------------------------------------------------------
# STEP 2: FIT A GJR-GARCH MODEL TO EACH ASSET (training data only)
# ------------------------------------------------------------
# Volatility update:
#   tomorrow's variance = omega + (alpha + gamma * [today's shock < 0]) * shock^2
#                         + beta * today's variance
# gamma is 0 for plain GARCH. With ASYMMETRIC = True (GJR-GARCH), gamma > 0
# means bad days raise volatility more than good days.

fits = {}
for asset in assets:
    model = arch_model(R_train[asset], mean="Constant", vol="GARCH",
                       p=1, o=1 if ASYMMETRIC else 0, q=1, dist="skewt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits[asset] = model.fit(disp="off")
    if fits[asset].convergence_flag != 0:
        print(f"WARNING: the GARCH fit for {asset} did not fully converge.")


def param(asset, key):
    return float(fits[asset].params[key])


par = {name: np.array([param(a, key) for a in assets]) for name, key in
       {"mu": "mu", "omega": "omega", "alpha": "alpha[1]",
        "gamma": "gamma[1]", "beta": "beta[1]"}.items()
       if ASYMMETRIC or name != "gamma"}
if not ASYMMETRIC:
    par["gamma"] = np.zeros(n)
persistence = par["alpha"] + par["beta"] + par["gamma"] / 2   # < 1 means volatility settles back

params_table = pd.DataFrame({
    "Daily mean (%)": par["mu"], "omega": par["omega"], "alpha": par["alpha"],
    "gamma (bad-day extra)": par["gamma"], "beta": par["beta"],
    "Persistence": persistence,
    "Tail df (nu)": [param(a, "eta") for a in assets],
    "Skew (lambda)": [param(a, "lambda") for a in assets],
}, index=assets)
params_table.to_csv(output / "garch_parameters.csv")

# Average daily return (in %) in the training data, used as the simulation drift.
sample_mean_pct = R_train.mean().values

# Ceiling on simulated variance (see VOL_CAP_MULTIPLE).
max_vol = np.array([fits[a].conditional_volatility.max() for a in assets])
variance_cap = (VOL_CAP_MULTIPLE * max_vol) ** 2
cap_hits = [0, 0]      # [days where the cap was applied, total simulated days]

if (persistence >= 1).any():
    print("WARNING: persistence is 1 or more for at least one asset; "
          "its volatility never settles back, so long simulations may explode.")


# ------------------------------------------------------------
# STEP 3: STANDARDIZED SHOCKS AND DIAGNOSTICS
# ------------------------------------------------------------
# Rows are days, columns are assets. Each row keeps the three assets'
# shocks for that day together; this is the dependence structure.

Z = pd.DataFrame({a: fits[a].std_resid for a in assets}).dropna()
# Re-center to mean 0 and variance 1 so the drift in the simulation is
# controlled only by the chosen average return.
Z = (Z - Z.mean()) / Z.std()
Zv = Z.values


def ljung_box_p(x, lags=10):
    return float(acorr_ljungbox(np.asarray(x), lags=[lags], return_df=True)["lb_pvalue"].iloc[0])


# A p-value above 0.05 means no leftover pattern. Squared returns should
# show clustering before GARCH (small p) and none after (large p).
diagnostics = pd.DataFrame({a: {
    "Volatility clustering before GARCH (squared returns)": ljung_box_p((R_train[a] - R_train[a].mean()) ** 2),
    "Volatility clustering after GARCH (squared shocks)": ljung_box_p(Z[a] ** 2),
    "Pattern in shocks after GARCH": ljung_box_p(Z[a]),
} for a in assets})


# ------------------------------------------------------------
# STEP 4: THE SIMULATOR
# ------------------------------------------------------------

def next_variance(shock, variance):
    """GJR-GARCH: variance for the next day, given today's shock and variance (all in % units)."""
    return (par["omega"] + (par["alpha"] + par["gamma"] * (shock < 0)) * shock ** 2
            + par["beta"] * variance)


def simulate(sigma2_start, n_paths, rng, drift_pct=None, horizon=HORIZON):
    """
    Simulate daily simple returns, shape (n_paths, horizon, n assets).

    sigma2_start: each asset's variance (in %^2) for the first simulated day.
    drift_pct:    each asset's average daily return in %. None = training average.
    """
    mean = sample_mean_pct if drift_pct is None else drift_pct
    variance = np.tile(sigma2_start, (n_paths, 1))
    paths = np.empty((n_paths, horizon, n))
    for day in range(horizon):
        shocks = Zv[rng.integers(0, len(Zv), n_paths)]   # whole rows: keeps dependence
        shock = np.sqrt(variance) * shocks                # today's surprise, in %
        paths[:, day, :] = (mean + shock) / 100
        variance = next_variance(shock, variance)
        cap_hits[0] += int((variance > variance_cap).sum())
        cap_hits[1] += variance.size
        variance = np.minimum(variance, variance_cap)
    return np.clip(paths, -0.99, None)                    # a loss cannot exceed 100%


def horizon_returns(paths):
    """Compound daily returns into one return per path and asset."""
    return np.prod(1 + paths, axis=1) - 1


# Where volatility stands at the end of the training data.
e_last = np.array([fits[a].resid.iloc[-1] for a in assets])
s2_last = np.array([fits[a].conditional_volatility.iloc[-1] ** 2 for a in assets])
sigma2_now = next_variance(e_last, s2_last)

# Sanity check against the arch library's own one-day-ahead forecast.
for i, a in enumerate(assets):
    arch_forecast = float(fits[a].forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    if abs(arch_forecast - sigma2_now[i]) > 1e-6 * max(1.0, arch_forecast):
        raise RuntimeError(f"Variance recursion does not match arch for {a}.")

# Variance levels for reference: the model's long-run level, and the average
# variance in the training data (used as the start for the like-for-like check).
sample_var = R_train.var().values
long_run = np.where(persistence < 0.999, par["omega"] / (1 - persistence), sample_var)

drift = None
if EXPECTED_RETURN_ANNUAL is not None:
    drift = np.array([EXPECTED_RETURN_ANNUAL[a] for a in assets]) / TRADING_DAYS * 100


# ------------------------------------------------------------
# STEP 5: GENERATE THE SCENARIOS
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)
cap_hits[:] = [0, 0]
paths = simulate(sigma2_now, N_SCENARIOS, rng, drift)
cap_share = cap_hits[0] / cap_hits[1]
scenarios = pd.DataFrame(horizon_returns(paths), columns=assets)
scenarios.index.name = "Scenario"

scenarios.to_csv(output / "scenarios_horizon_returns.csv")
np.save(output / "scenario_paths.npy", paths)


# ------------------------------------------------------------
# STEP 6: CHECK 1 - DO THE SIMULATED PERIODS LOOK LIKE HISTORY?
# ------------------------------------------------------------
# Simulated from the average volatility level in the training data (not
# today's), so the comparison with training history is like for like.

rng_check = np.random.default_rng(SEED + 1)
paths_check = simulate(sample_var, N_SCENARIOS, rng_check)
sim_h = horizon_returns(paths_check)

hist_h = (np.exp(np.log1p(train).rolling(HORIZON).sum()) - 1).dropna().values


def tail_stats(x):
    var = np.quantile(x, 0.05, axis=0)
    cvar = np.array([x[x[:, j] <= var[j], j].mean() for j in range(x.shape[1])])
    return pd.DataFrame({"Mean": x.mean(axis=0), "Std dev": x.std(axis=0, ddof=1),
                         "5% VaR (loss)": -var, "5% CVaR (loss)": -cvar,
                         "Worst": x.min(axis=0)}, index=assets) * 100


compare = pd.concat({"History (overlapping windows)": tail_stats(hist_h),
                     "Simulated": tail_stats(sim_h)}, axis=1)

corr_hist = train.corr().values
corr_sim = np.corrcoef(paths_check.reshape(-1, n), rowvar=False)
corr_table = pd.DataFrame(
    {"History": [corr_hist[i, j] for i in range(n) for j in range(i + 1, n)],
     "Simulated": [corr_sim[i, j] for i in range(n) for j in range(i + 1, n)]},
    index=[f"{assets[i]} - {assets[j]}" for i in range(n) for j in range(i + 1, n)])


# ------------------------------------------------------------
# STEP 7: CHECK 2 - VALIDATION BACKTEST
# ------------------------------------------------------------
# Walk through the validation set in non-overlapping windows. For each:
#   - update volatility using the validation data seen so far (parameters
#     stay fixed),
#   - simulate the next HORIZON days,
#   - see whether the REAL outcome fell below the simulated 5% level.
# If the simulator is well calibrated, that happens in about 5% of windows.

Rv = R_val.values
V = np.empty((len(Rv) + 1, n))     # V[k] = variance for validation day k
V[0] = sigma2_now
for k in range(len(Rv)):
    V[k + 1] = next_variance(Rv[k] - par["mu"], V[k])

rng_bt = np.random.default_rng(SEED + 2)
names = assets + ["Portfolio"]
hits = np.zeros(n + 1)
windows = 0
for start in range(0, len(Rv) - HORIZON + 1, HORIZON):
    sim = horizon_returns(simulate(V[start], N_BACKTEST, rng_bt))
    sim_all = np.column_stack([sim, sim @ w_check])
    real = np.prod(1 + validation.values[start:start + HORIZON], axis=0) - 1
    real_all = np.append(real, real @ w_check)
    hits += real_all < np.quantile(sim_all, 0.05, axis=0)
    windows += 1

backtest = pd.DataFrame({"Windows": windows, "Expected misses": 0.05 * windows,
                         "Actual misses": hits.astype(int),
                         "Miss rate": hits / windows}, index=names)


# ------------------------------------------------------------
# STEP 8: PRINT
# ------------------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print(f"Fitted on {len(train)} training days | {N_SCENARIOS:,} scenarios of {HORIZON} days each")
print(("GJR-GARCH" if ASYMMETRIC else "GARCH(1,1)") + " with skewed-t shocks per asset; "
      "whole days of shocks resampled to keep the assets' dependence")
print("Average return used:", "training-sample averages" if drift is None
      else "EXPECTED_RETURN_ANNUAL " + str(EXPECTED_RETURN_ANNUAL))
print(f"Volatility cap: {VOL_CAP_MULTIPLE}x the highest fitted volatility "
      f"(applied on {cap_share:.3%} of simulated asset-days in the main scenarios)")

print("\nFITTED PARAMETERS (returns in %):")
print(params_table.round(4).to_string())

print("\nDIAGNOSTICS (Ljung-Box p-values, 10 lags; above 0.05 = no leftover pattern):")
print(diagnostics.round(4).to_string())

print("\nVOLATILITY (annualized %):")
print(pd.DataFrame({"End of training": np.sqrt(sigma2_now * TRADING_DAYS),
                    "Training average": np.sqrt(sample_var * TRADING_DAYS),
                    "Model long-run level": np.sqrt(long_run * TRADING_DAYS)},
                   index=assets).round(1).to_string())

print(f"\nCHECK 1: SIMULATED VS HISTORICAL {HORIZON}-DAY RETURNS (%, simulated from the training-average volatility):")
print(compare.round(2).to_string())

print("\nCHECK 1: DAILY CORRELATIONS")
print(corr_table.round(3).to_string())

print(f"\nCHECK 2: VALIDATION BACKTEST (real {HORIZON}-day outcome below simulated 5% level; "
      "expect about 5% misses):")
print(backtest.round(3).to_string())
print("Note: only a few windows fit in the validation set, so this is a rough sanity check, "
      "not a precise test.")

print(f"\nSCENARIOS GENERATED (starting from today's volatility), {HORIZON}-day returns (%):")
print(tail_stats(scenarios.values).round(2).to_string())
print("\nScenario correlations:")
print(scenarios.corr().round(3).to_string())

n_clipped = int((paths <= -0.99 + 1e-12).sum())
if n_clipped:
    print(f"\nNote: {n_clipped} simulated daily returns were limited to -99%.")
print("\nSaved: scenarios_horizon_returns.csv, scenario_paths.npy, garch_parameters.csv")
