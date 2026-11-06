# Matched-return CVaR multi-period optimization

Upload **this entire folder** to GitHub, preserving its internal structure. It is
self-contained: no other project directories, old scenario caches, experiment
reports, manifests or archived source files are needed to run it.

There are exactly four distribution models:

| Model | Marginal volatility | Dependence / innovation sampling |
|---|---|---|
| **FHS GARCH** | GARCH for Equity, Bond and BTC | Resample aligned standardized residual rows |
| **FHS Hybrid** | Equity/BTC GARCH; Bond EWMA | Resample aligned hybrid residual rows |
| **DCC GARCH** | GARCH for all three assets | Resample whitened empirical rows and recolor with evolving DCC correlations |
| **DCC Hybrid** | Equity/BTC GARCH; Bond EWMA | The same DCC approach fitted to hybrid residuals |

Historical scenarios and the weighted DCC challenger are not extra models in
this package. The original research implementations remain preserved elsewhere.

## Files

```text
CVaR_MPO/
    cvar_mpo.py                 # Optimizer, matched targets, costs and execution
    scenario_generation.py      # All four forecasting/scenario methods
    run_cvar_mpo.py              # Monthly validation walk-forward runner
    requirements.txt
    README.md
    SOURCE_PROVENANCE.json       # Informational only; not loaded at runtime
    .gitignore
    data/
        portfolio_train.csv
        portfolio_validation.csv
```

The CSVs contain **daily simple returns as decimals**, not prices or percentages.
The first column is the date index, followed by `Equity`, `Bond` and `BTC`.
For example, `0.01` means a 1% daily return. The supplied data reproduce the
February 2022–May 2024 validation comparison. Testing data are not included.

## Install and run

Tested using Python **3.10.7** and the direct scientific-package versions pinned
in `requirements.txt`. Use Python 3.10 for the verified setup. No CVXPY, OSQP, network
data download or credentials are required.

In Windows PowerShell, open this folder and run:

```powershell
py -3.10 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" run_cvar_mpo.py
```

On macOS/Linux:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_cvar_mpo.py
```

The default run evaluates **all four models at H=1 and H=3**, producing eight
portfolio paths. Progress prints after each monthly rebalance. Forecast fitting
and solving require some time; no previous completed run is shipped in this folder.

From the original project's root, its existing environment can also run:

```powershell
& ".\.venv\Scripts\python.exe" ".\CVaR_MPO\run_cvar_mpo.py"
```

To select a subset, quote each model name containing spaces:

```powershell
& ".\.venv\Scripts\python.exe" run_cvar_mpo.py --models "DCC GARCH" "DCC Hybrid" --horizons 1 3
```

Use a different output directory for a different configuration:

```powershell
& ".\.venv\Scripts\python.exe" run_cvar_mpo.py --output results_second_run
```

An unchanged completed run in the same output directory is checksum-verified
and reused. Source, data, configuration or core package-version changes require
a new output directory. `--solver full` uses the full scenario LP rather than the
default exact constraint-generation backend. Both solve the same CVaR problem
and retain every scenario. The default backend falls back to the full LP if its
optimality gap cannot be certified.

## Preserved forecasting specifications

- Refit at each monthly origin using only the preceding **five calendar years**;
  require at least 750 prior aligned observations.
- GARCH(1,1) marginals retain the original constant-mean/skewed-t likelihood.
  Scenario draws are empirical residual vectors, not fresh skewed-t draws.
- Bond EWMA decay is **0.94** for the hybrid methods. Bond simulation drift uses
  its original **five-year arithmetic daily mean**, not the later prior-252-day
  override. Equity/BTC simulation drifts also use original five-year daily means.
- DCC uses the original Gaussian quasi-likelihood estimator and empirical
  whitened innovations. Gaussian estimation does not impose Gaussian scenarios.
- Generate **10,000 paths**, with original seed `42 + 1_000_003 * origin_index`.
  Draw successive months separately so H=1 and H=3 share the first-month paths.
- Continue volatility and DCC states across the three monthly stages. Reset only
  each month's incremental return accumulator; future stages do not use future fits.
- Retain the original volatility caps, daily downside clipping, and float32
  monthly-cache precision, then convert scenarios to float64 for optimization.

Scenario arrays have shape `(scenario, period, asset)`, with asset order
`Equity, Bond, BTC`. Each period contains **incremental monthly simple returns**,
not cumulative three-month returns. The optimizer uses the **native scenario
mean for each model and stage**, without shifting scenarios to a common mean.

## Matched-return optimization and costs

For each model and planned month, the original maximum-Sharpe routine estimates
a reference portfolio using that stage's native mean/covariance, zero risk-free
return, full investment, long-only bounds and the same BTC cap. The gross floor is:

```text
floor[k] = native_mean[k] @ maximum_sharpe_weights[k] - 0.000001 * holding_days[k]
```

The portfolio objective is:

```text
minimize 0.5 * sum(monthly CVaR95) + estimated proportional trading costs
```

Expected return enters through the floor, not an extra return reward. Defaults
are **lambda 0.5**, **CVaR95**, **20% BTC cap**, and **0.1% cost per dollar bought or
sold**. Floors are not relaxed beyond the original stated daily allowance.

The plan is open-loop: all scenarios share the planned allocations. Future
planning trades approximate differences between successive target weights and
ignore future return drift. This is a **sum of stage CVaRs**, not CVaR of compounded
terminal wealth, nested CVaR, or a scenario-tree policy.

Reoptimize monthly and **implement only the first month's allocation**. Execution
uses exact self-financing buy/sell fees and realized daily holding drift. Start
from cash, charge the initial purchase fee and impose no terminal liquidation.
The cap applies at rebalance; weights can drift above it during the month.

The original single-period optimizer did not plan around costs. Therefore H=1
with the retained cost planner can differ slightly from the original cost-unaware
results even though its forecasts and return-target specifications are restored.

## Information cutoff and calendar

Forecasting receives training plus strictly elapsed validation observations.
Every plan/reference is saved before current-month return values enter execution.
No testing file or combined all-data file is loaded automatically.

The built-in calendar contains the original published 2022–2024 NYSE sessions,
including full planning months beyond the validation boundary. It is independent
of future return rows. For another date range, provide `--calendar calendar.csv`
with an ordered `Date` column of complete trading sessions. Its schedule must be
known at the forecast origins; unexpected closures require appropriate
announcement-time handling when constructing that calendar.

Only the final evaluation month may be partial. Its daily returns count toward
performance, but it is excluded from monthly CVaR and VaR coverage. Its forecast
still uses the complete month, so data availability cannot truncate the forecast.

## Outputs and verified reference results

Default results go to `results/` beside the three Python files:

- `sharpe_comparison.csv` — four models by H=1/H=3.
- `portfolio_summary.csv` — return, volatility, Sharpe, drawdown, CVaR, turnover,
  fees, allocations and VaR breaches.
- `portfolio_monthly.csv`, `portfolio_daily.csv`, `planned_allocations.csv`.
- `forecasts/`, `decisions/`, `protocol.json`, `run_manifest.json` — forecast
  checkpoints, pre-execution decisions, settings, environment and checksums.

The full isolated-copy validation run reproduced the restored research results:

| Model | H=1 net Sharpe | H=3 net Sharpe |
|---|---:|---:|
| FHS GARCH | 0.897017 | 0.895929 |
| FHS Hybrid | 0.818398 | 0.818326 |
| DCC GARCH | 0.919191 | 0.920181 |
| DCC Hybrid | 0.779395 | 0.780099 |

Sharpe uses net daily returns, zero risk-free rate and 252-day annualization.
CAGR is separately compounded. Numerical fitting can vary across software
versions and platforms; these are reference results, not guaranteed future returns.
These validation observations have already been used repeatedly for development.

## Programmatic use

```python
import numpy as np
from scenario_generation import generate_scenarios
from cvar_mpo import optimize_matched_cvar, execute_first_period

# history contains only observations available strictly before origin.
# periods contains full monthly DatetimeIndex blocks from an ex-ante calendar.
forecast = generate_scenarios(history, origin, periods, origin_index)
plan = optimize_matched_cvar(
    forecast.returns["DCC GARCH"],
    pretrade_weights=np.zeros(3),  # initial cash; later use actual drifted weights
    holding_days=forecast.holding_days["DCC GARCH"],
)
# Once month-one returns are observed:
execution = execute_first_period(plan, realized_daily_returns)
# At the next origin, refit/replan using execution.ending_weights.
```

`SOURCE_PROVENANCE.json` records the original implementations consolidated into
these files. It is informational only: the original paths do not need to exist
on your teammate's computer. Existing research files and results were preserved.
