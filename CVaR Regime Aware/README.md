# Regime-aware and regime-switching portfolio backtests

## V1 runs with the tuned Jump model

The V1 runs compare conditional value-at-risk (CVaR) and mean-variance optimization (MVO)
using a Gaussian hidden Markov model (HMM) and a training-tuned Jump model. Each optimizer
plans three months at each monthly rebalance and executes the first month's allocation.
The annual risk-free rate is **4%** for the max-Sharpe return reference and reported Sharpe ratios.

### Set up from the repository root

Install the packages into the same Python environment you will use for the runs. The V1
backtests were verified with Python 3.13.

```powershell
python -m pip install -r requirements.txt
```

### Run on the validation data

```powershell
python "CVaR Regime Models/run_validation_v1.py"
```

This command runs **eight backtests**. Each combination of CVaR or MVO and regime-aware or
regime-switching uses both HMM and tuned Jump forecasts. The validation charts include the
three-asset heuristic benchmark for both forecast models.

The runner reads the training and validation return CSVs, the saved HMM means and regime
probabilities, the heuristic's training signals, and the completed 10,000-scenario DCC-GARCH
archive. It tunes the Jump penalty on training data and fits the Jump model at each validation
rebalance using returns available before that date.

### Run on the held-out test data

**CVaR and sensitivity analysis**

```powershell
python "CVaR MPO and Scenarios/run_testing_v1.py"
```

This command runs HMM regime-aware CVaR, HMM regime-switching CVaR, and tuned Jump
regime-switching CVaR. It also runs the three-asset heuristic with each forecast model.
After the baseline runs, it changes one setting at a time for each of the three CVaR models.
Bold values mark the baseline.

| Setting | Values tested |
| --- | --- |
| Transaction costs per dollar traded | 5, **10**, 15, and 20 basis points |
| Bitcoin allocation cap | 15%, **20%**, 25%, and 30% |
| CVaR confidence level | 90%, **95%**, and 99% |
| CVaR risk weight | 0.10, 0.25, **0.50**, 0.75, and 1.00 |

**MVO**

```powershell
python "MVO Regime Models/regime_mvo_backtest.py" --testing
```

This command runs HMM regime-switching MVO, tuned Jump regime-aware MVO, and tuned Jump
regime-switching MVO. The `--testing` flag selects the held-out run. Running the script without
the flag selects the historical paired validation workflow, which reads
`all_dcc_comparison_results_rf4/` and `regime_switching_all_dcc_results_rf4/`.

Both held-out commands read the training, validation, and test return CSVs. For each rebalance,
they generate HMM and tuned Jump forecasts and 10,000 DCC-GARCH scenarios using returns available
before that date. CVaR creates or reuses its forecast cache in `Testing Results v1/_forecast_inputs/`,
and MVO does the same in `Testing Results/_forecast_inputs/`.

### How the policies work

| Policy | Return floor and risk objective |
| --- | --- |
| Regime-aware CVaR | Applies the max-Sharpe return floor at every stage. The objective uses scenario CVaR and trading costs. |
| Regime-switching CVaR | Applies the floor in forecast bull stages. In forecast bear stages the plan minimizes CVaR plus trading costs without a return floor. |
| Regime-aware MVO | Applies the max-Sharpe return floor at every stage. The objective uses variance and trading costs. |
| Regime-switching MVO | Applies the floor in forecast bull stages. In forecast bear stages the plan minimizes variance plus trading costs without a return floor. |

Switching uses the forecast month-end bear probability for each planning stage. A probability below
0.5 selects bull, and a probability above 0.5 selects bear. CVaR uses the DCC-GARCH scenario returns
directly. MVO uses sample covariances from those scenarios. The baseline cap is 20% Bitcoin, and
the transaction cost is 10 basis points per dollar traded.

### Required files and their purpose

The tables below are the **50-file GitHub upload list** for the V1 commands. Keep the paths relative
to the repository root and include this README for the instructions.

#### Setup, return data, and saved validation inputs

| Path | Purpose |
| --- | --- |
| `requirements.txt` | Lists the Python packages for estimation, optimization, calendars, and figures. |
| `Data/portfolio_train.csv` | Daily equity, bond, and Bitcoin returns from January 5, 2015 through January 31, 2022. Provides the initial estimation history and training data for the Jump penalty and heuristic. |
| `Data/portfolio_validation.csv` | Daily returns from February 1, 2022 through May 31, 2024. Supplies realized validation returns and contributes history for the held-out runs. |
| `Data/portfolio_test.csv` | Daily returns from June 3, 2024 through October 8, 2026. Supplies realized returns for the held-out runs. |
| `outputs/training_hmm_signals.csv` | Saved HMM training signals used to select the HMM heuristic threshold. |
| `2-State HMM/HMM Forecasts/expected_returns.csv` | Saved HMM expected asset returns by validation rebalance and planning stage. |
| `2-State HMM/HMM Forecasts/regime_probabilities.csv` | Saved HMM bear probabilities at the validation rebalance and forecast month-end. |
| `CVaR Regime Models/forecast_comparison_results_rf4/run_manifest.json` | Records the completed validation scenario run and the checksums V1 uses to verify the scenario files. |
| `CVaR Regime Models/forecast_comparison_results_rf4/forecasts/*.npz` | **All 28 dated files**, one per validation rebalance. V1 reads the 10,000 DCC-GARCH paths for each of three monthly stages and verifies their dates and HMM means. |

#### Forecast, optimization, and reporting code

| Path | Purpose |
| --- | --- |
| `2-State HMM/HMM Forecasts/hmm_regime_model.py` | Fits the Gaussian HMM and computes regime probabilities and expected returns. Also supplies the trading calendar and forecast helpers used by both model families. |
| `2-State HMM/Jump Model Forecasts/jump_regime_model.py` | Provides the shared Jump features and base implementation imported by the tuned model. |
| `2-State HMM/Jump Model Forecasts/jump_regime_model_tuned.py` | Selects the Jump penalty on training returns and fits the Jump model used in the V1 runs. |
| `CVaR MPO and Scenarios/scenario_generation.py` | Fits the historical volatility and correlation models and simulates return paths. The held-out runs use its DCC-GARCH paths. |
| `CVaR MPO and Scenarios/cvar_mpo.py` | Provides the numerical CVaR optimizer, scenario loss calculations, and first-period execution and transaction-cost accounting. |
| `CVaR Regime Models/regime_cvar.py` | Implements both V1 CVaR policies and the scenario-based 4% max-Sharpe return reference shared with MVO. It also provides return-data and checksum helpers. |
| `MVO Regime Models/regime_aware_mvo.py` | Implements the variance and trading-cost objective and the regime-aware MVO policy. |
| `MVO Regime Models/regime_switching_mvo.py` | Uses the MVO optimizer to select the bull or bear policy at each stage. |
| `simple_regime_strategy.py` | Runs the three-asset heuristic benchmark using HMM or tuned Jump signals. |
| `CVaR Regime Models/portfolio_metrics.py` | Calculates 4% excess-return Sharpe, drawdown, VaR, turnover, allocation statistics, and other report metrics. |
| `regime_comparison_figures.py` | Draws cumulative wealth, allocation, and drawdown figures from backtest results. |

#### Entry points

| Path | Purpose |
| --- | --- |
| `CVaR Regime Models/run_validation_v1.py` | Runs both optimizers on validation and writes validation reports, charts, and the first-month DCC-GARCH figures. |
| `CVaR MPO and Scenarios/run_testing_v1.py` | Runs held-out CVaR and its sensitivity analysis. Also provides held-out forecast preparation and report functions for MVO. |
| `MVO Regime Models/regime_mvo_backtest.py` | Runs the held-out MVO backtests when called with `--testing`. |

### Where the outputs go

| Run | Output location and files |
| --- | --- |
| V1 validation | `Regime Aware CVaR Results v1/`, `Regime Switching CVaR Results v1/`, `Regime Aware MVO Results v1/`, and `Regime Switching MVO Results v1/`. Each has `table_1_performance` and `table_2_allocations` as CSV and Markdown, `portfolio_daily.csv`, `portfolio_monthly.csv`, and `forecast_and_plan_audit.csv`. The MVO validation folders also contain the wealth, allocation, and drawdown charts comparing CVaR, MVO, and the heuristic. |
| First-month validation scenario figures | `CVaR Regime Models/forecast_comparison_results_rf4/figures/dcc_first_month_correlations.png` and `dcc_first_month_loss_distributions.png`. The runner uses the first forecast month at each rebalance. |
| Held-out CVaR and heuristic | `Testing Results v1/`. Each strategy folder has performance and allocation tables, daily and monthly CSVs, and an allocation figure. CVaR folders also contain forecast and plan audits. The top-level results folder contains HMM and Jump wealth and drawdown comparisons. |
| Held-out CVaR sensitivity | `sensitivity analysis/Sensitivity Summary v1.md`. |
| Held-out MVO | `Testing Results/`. Each of the three MVO folders contains performance and allocation tables and wealth, allocation, and drawdown figures. |

The historical `run_forecast_comparison.py` producer created the saved validation scenario archive
and the optional `scenario_statistics.csv` and `scenario_correlations.csv` diagnostics. For
first-month DCC-GARCH statistics, use `source = DCC GARCH` and `stage = 1`. For the correlation
table, use `source = DCC GARCH`, `row_stage = 1`, and `column_stage = 1`.

## Historical comparison runs

Validation now defaults to **10,000 DCC scenarios per origin**, matching the
underlying generator. The HMM/Jump forecast comparison also uses 10,000 native
paths. Cached runners reject older 5,000-scenario inputs by default; regenerate
the forecasts rather than relabelling, duplicating or extending cached arrays.
Existing files in `Archive/` remain historical 5,000-scenario results.

Run the forecast producer before the cache-based comparisons:

```powershell
.venv/Scripts/python.exe "CVaR Regime Models/run_forecast_comparison.py"
.venv/Scripts/python.exe "CVaR Regime Models/run_all_dcc_comparison.py"
.venv/Scripts/python.exe "CVaR Regime Models/run_regime_switching.py"
.venv/Scripts/python.exe "CVaR Regime Models/run_switching_all_dcc.py"
```

All use 10,000 scenarios by default. A deliberate smaller smoke run must pass
the same explicit `--n-scenarios` value to the producer and consumers. Changed
code or settings require a fresh output directory; old manifests are not rewritten.

This folder compares 12 portfolios on the existing validation split (February
2022–May 2024): HMM, DCC GARCH and DCC Hybrid scenarios, each using minimum CVaR
and matched-return CVaR, with 1- and 3-month horizons. Test data is never loaded.

Run from the project root:

```powershell
.venv/Scripts/python.exe "CVaR Regime Models/run_regime_cvar.py"
```

The script prints forecast inputs and portfolio results, and saves the full
comparison under `results/`. Repeating the same command verifies checksums and
reuses a completed run. A changed configuration or code requires another output
folder (`--output "CVaR Regime Models/results_new"`). An interrupted run resumes
with its verified scenario caches and recomputes the portfolio paths from cash.

## Model and inputs

The adapter in `regime_cvar.py` uses the same `solve_cvar_plan` or full scenario
LP as the two public policies in `../CVaR MPO and Scenarios/cvar_mpo.py`. Both
minimize

```text
0.5 * sum_k CVaR95(-scenario_return_k @ weight_k)
    + 0.001 * sum_k sum_assets abs(weight_k - previous_weight_k).
```

Matched return additionally requires, separately at each stage:

```text
expected_return_k @ weight_k >=
    expected_return_k @ maximum_Sharpe_weights_k - 1e-6 * holding_days_k.
```

The Sharpe reference uses the selected explicit monthly expected returns, the
native scenario sample covariance, and the original multistart SLSQP method.
It compounds a 4% annual risk-free rate over each stage's trading days.
Its reference weights obey the same caps. The default
wrapper in `cvar_mpo.py` deliberately requires native scenario means, so this
adapter calls the underlying LP to permit separate forecast inputs. That source
file is unchanged. Native-mean equivalence is tested for both policies and both
backends. There is no expected-return reward in either existing objective:
expected returns affect matched return through the floor and are reported for
minimum CVaR.

All portfolios are long-only, fully invested, and capped at 20% BTC. They begin
in cash, reoptimize monthly, and execute only the first allocation. Future
planned trades ignore return drift exactly as in the original planner; realized
holdings use the original exact self-financing fees and daily drift. There is no
terminal liquidation. Risk is additive stage CVaR, not nested or terminal-wealth
CVaR. Although the paths retain cross-month dependence, that dependence is not
directly priced by this additive risk measure.

**HMM:** uses `../2-State HMM/HMM Forecasts/expected_returns.csv` for analytical
conditional monthly means and `cvar_scenarios.npz` for the joint paths. The
existing generator filters the last known daily regime posterior, evolves daily
Markov states continuously across months, and draws multivariate state-specific
Gaussian daily returns before monthly compounding. The existing inputs are
loaded, not refitted. Exported information dates must agree with the last return
available before each decision; period boundaries must match NYSE sessions.
The original archive has no training-data checksum, so original forecast
provenance cannot be independently established by this runner's timing checks.

**Conventional means:** expand over completed historical NYSE calendar months.
For each asset, first calculate `prod(1 + daily_return) - 1` within each month,
then take the arithmetic average across those monthly returns. Incomplete
months, including January 2015 in the supplied data, are excluded. These are
monthly forecasts, repeated across future stages without daily scaling. They
are not obtained by compounding the mean daily return. Only strictly prior
history is included at each rebalance.

**Conventional scenarios:** reuse the original `scenario_generation.py` without
changing drift, residual draws, DCC recursion, marginal volatility, volatility
caps, clipping, or float32 cache conventions. The original shared generator
internally calculates all four methods; only DCC GARCH and DCC Hybrid are used
or saved here. It fits a rolling five-year history, while the supplied HMM fits
expanding history. Both conventional methods share the same historical mean
inputs but retain their distinct, unshifted scenario distributions. Their
scenario sample means need not equal the historical forecast means.

All three scenario sources use 10,000 paths by default, matching the HMM archive.
Never apply a second regime-volatility multiplier to HMM paths: their
state-specific covariance already models this. No HMM/GARCH blend, volatility
overlay, or regime-dependent risk-aversion penalty is applied here.

## Future regression expected returns

Replace just the conventional mean provider, leaving both scenario generators
and optimization formulas unchanged:

```powershell
.venv/Scripts/python.exe "CVaR Regime Models/run_regime_cvar.py" `
  --conventional-means "path/to/regression_expected_returns.csv" `
  --output "CVaR Regime Models/results_regression"
```

The file must have one row for every decision date and each of stages 1, 2, 3:

```csv
rebalance_date,month_ahead,forecast_month,information_date,Equity,Bond,BTC
2022-02-01,1,2022-02,2022-01-31,0.01,0.002,0.03
2022-02-01,2,2022-03,2022-01-31,0.012,0.002,0.035
2022-02-01,3,2022-04,2022-01-31,0.014,0.002,0.04
```

Values are **decimal monthly simple-return forecasts**. `forecast_month` is
optional but checked if supplied. `information_date` is mandatory and must
precede `rebalance_date`; forecast authors must ensure training, features and
model selection also exclude future information. The historical run exports
`expected_returns_conventional.csv` in this exact format as an input example.
Both DCC methods consume the same supplied means. HMM keeps its HMM forecasts.

## Results and interpretation

- `report.md`: settings, performance table, HMM-minus-DCC comparisons, and inputs.
- `portfolio_summary.csv`: net returns/CAGR, annualized volatility, zero-RF
  Sharpe, drawdown, empirical daily/monthly CVaR95, turnover and calibration.
- `portfolio_monthly.csv`, `portfolio_daily.csv`, `planned_allocations.csv`:
  executed portfolios, net returns, fees, forecast risk and future planned trades.
- `expected_returns.csv`: every origin/source/stage's expected-return vector.
- `scenario_statistics.csv`: separate input and scenario means, SD, lower 5%
  return quantile, empirical loss CVaR95 and HMM mean Monte Carlo diagnostics.
- `scenario_correlations.csv`: cross-asset and cross-month sample correlations.
- `regime_inputs.csv`: information dates and bear probabilities.
- `forecasts/YYYY-MM-DD.npz`: complete raw HMM and DCC cubes, means, holding days,
  and generator diagnostics/provenance. Load with `numpy.load(..., allow_pickle=False)`;
  cube keys are `HMM`, `DCC GARCH`, `DCC Hybrid`, each `(10000, 3, 3)`, asset order
  Equity, Bond, BTC. Sample means, covariance and losses use these unshifted cubes.
- `decisions/*.json`: all plans recorded before that month's execution.
- `run_manifest.json`, `protocol.json`, `run.log`: input/source checksums,
  environment versions, protocol and printed results.

VaR forecasts and forecast CVaR include the exact fee for the executed first
allocation. VaR breaches are measured on complete months; the nominal rate is
5%. Lower VaR pinball loss is better. The signed ES calibration residual is
`forecast_CVaR - forecast_VaR - max(realized_loss - forecast_VaR, 0)/0.05`;
its average is zero in expectation under calibration and requires cautious
interpretation in a short sample. Monthly empirical CVaR95 with 28 observations
is dominated by the worst one or two months. Comparisons change both forecast
means and distributions and retain different HMM/DCC estimation windows;
they do not isolate the causal effect of regimes or establish statistical
superiority. HMM Gaussian state shocks and DCC empirical innovations also have
different tail assumptions.

Checks:

```powershell
.venv/Scripts/python.exe -m unittest discover -s "Regime Aware Models" -p "test_*.py" -v
.venv/Scripts/python.exe "CVaR Regime Models/verify_results.py"
```

The completed validation run passed six unit checks and independent verification
of all 336 plans, 24 comparisons against the full scenario LP, fee accounting,
586 realized sessions per portfolio, and saved input/source/output checksums.

A short integration check is available with `--limit-months 1 --n-scenarios 256
--output "CVaR Regime Models/smoke_results_new"`; those results are not the full
validation comparison.
