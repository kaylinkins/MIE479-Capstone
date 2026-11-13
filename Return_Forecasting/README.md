Save the following as `Return_Forecasting/README.md`.

# Expected-Return Forecasting

This folder contains tools for generating monthly expected returns for Equity, Bond, and BTC. These forecasts can be passed to the MPO optimizers as expected-return inputs.

There are two separate approaches:

1. **Static forecasts:** historical averages or regressions that do not explicitly condition returns on a market regime.
2. **Regime-dependent forecasts:** estimates from the team’s HMM, including returns conditional on its states and probability-weighted expected returns.

The static return module does **not** estimate covariance matrices or read `portfolio_test.csv`. Covariance work is outside the scope of this module.

## Quick start

Run commands from the repository root.

```bash
python -m pip install numpy pandas yfinance exchange_calendars
python Return_Forecasting/return_forecasting.py
```

The first run may need internet access to download long adjusted-price histories for Equity (`^SP500TR`) and Bond (`AGG`). The script caches those prices in `Return_Forecasting/long_equity_bond_prices.csv`. If the project’s HMM dependencies are installed, the separate regime-dependent script can also be run:

```bash
python Return_Forecasting/regime_conditional_returns.py
```

See `HMM_regime_forecasts/README.md` for dependencies and details on the team’s HMM.

Each run writes or overwrites output files under `Return_Forecasting/outputs/`. Archive copies if you need to preserve outputs from previous runs.

## Inputs and return history

The static script reads:

- `portfolio_train.csv`
- `portfolio_validation.csv`

It searches for these files in `Data Work/`, `HMM_regime_forecasts/`, or `Data/datasets/`. The CSVs should contain daily simple returns with the columns `Equity`, `Bond`, and `BTC`.

The script deliberately uses only training and validation data. It does **not** open `portfolio_test.csv`.

Histories are built separately by asset:

- Equity and Bond use longer Yahoo adjusted-price histories before 2015. From 2015 onward, they use the team’s aligned daily returns.
- BTC uses the team’s aligned returns starting in February 2015. January 2015 is excluded because the aligned sample does not include a preceding common-session close.
- Monthly returns are compounded simple returns.

## Return units and dates

All forecast returns are **monthly simple returns in decimals**, not percentages or annualized returns.

- `0.01` means a forecast return of **1% for the month**.
- A forecast row’s `forecast_month` is the calendar month being forecast.
- `month_ahead = 1` means the next month after the rebalance month’s information cutoff.
- `month_ahead = 2` and `3` are forecasts made from that same rebalance date for the following months.

For the monthly validation outputs, the rebalance date is the **first XNYS trading session of the month**. The information date is the last available trading session before that rebalance—normally the final XNYS session of the previous month.

The wide CSV handoff files contain:

- `rebalance_date`, `information_date`, `month_ahead`, and `forecast_month`
- The expected returns: `Equity`, `Bond`, and `BTC`
- The method used for each asset: `method_Equity`, `method_Bond`, and `method_BTC`

When passing a vector to an optimizer, use the named asset columns in the order expected by that optimizer. For example:

```python
import pandas as pd

forecasts = pd.read_csv(
    "Return_Forecasting/outputs/"
    "walk_forward_validation_expected_returns_h1_h3.csv"
)

row = forecasts.loc[
    (forecasts["rebalance_date"] == "2022-02-01")
    & (forecasts["month_ahead"] == 1)
].iloc[0]

mu = row[["Equity", "Bond", "BTC"]].to_numpy(dtype=float)
```

## Static forecast files: which one should I use?

| File | What it contains | Typical use |
|---|---|---|
| `selected_one_month_optimizer_expected_returns.csv` | One next-month vector for the first rebalance after the latest complete validation month | A single forward handoff or integration test |
| `selected_forward_expected_returns.csv` | Month-ahead forecasts for horizons 1–3, all made from the same latest information date | Looking at the next three months from one forecast origin |
| `walk_forward_validation_expected_returns.csv` | One-month forecasts at each validation-period rebalance date | A monthly validation backtest using the next-month vector at each rebalance |
| `walk_forward_validation_expected_returns_h1_h3.csv` | Horizons 1–3 at each validation-period rebalance date | A multi-month planning horizon, or comparison with HMM forecasts |
| `validation_selected_expected_returns.csv` | Validation forecasts using methods selected on the full validation period; includes realized returns | Diagnostics only—not a clean historical optimizer backtest |
| `validation_candidate_forecasts.csv` | Candidate-method forecasts and realized validation returns | Model diagnostics and candidate comparisons |

The three rows in `selected_forward_expected_returns.csv` are **not** three separate monthly rebalance forecasts. They all use the same information date. For monthly rebalancing, take the `month_ahead = 1` row from each rebalance date in a walk-forward file.

The static forecast files contain **forecasts**, not realized portfolio returns. To calculate portfolio performance, match forecast months with the corresponding realized returns.

## Static methods and selection

The static module compares historical averages, rolling means, and predictive regressions where there is enough history. Rolling-mean windows are 1, 3, 5, and 10 years when feasible. The regression candidates include an expanding AR(1) and a 12–1 momentum regression.

The current full-validation selections in the supplied run are:

| Asset | Selected method | Notes |
|---|---|---|
| Equity | `reg_ar1_expanding` | Expanding-sample AR(1) regression |
| Bond | `rolling_mean_10y` | Trailing 10-year monthly mean |
| BTC | `selected_shrinkage_50` | 50% `rolling_mean_1y` + 50% `expanding_mean` |

For static selection, a method is scored using the square root of the equally weighted average of its mean squared forecast errors at horizons 1, 2, and 3. A candidate must have at least 12 scored validation forecasts at each horizon and match the expanding-mean benchmark’s scored dates.

The module also reports:

- RMSE, MAE, and forecast bias
- RMSE skill versus the expanding-mean benchmark
- An out-of-sample signal-to-noise measure defined as `RMS(forecast returns) / RMSE(forecast errors)`

These are validation diagnostics, not guarantees of future performance.

### Full-validation selection vs. walk-forward selection

`selected_methods.json` records methods selected using the **full validation period**. Those are appropriate for forecasts made after that validation period—for example, the supplied June 3, 2024 forward forecast, which uses information through May 31, 2024.

The walk-forward files use a different procedure: methods are selected using **training data only** and then held fixed through validation. This prevents later validation outcomes from choosing methods for earlier historical forecasts. In the supplied walk-forward run, the selected methods were Equity `expanding_mean`, Bond `rolling_mean_10y`, and BTC `expanding_mean`. These can therefore differ from the full-validation selections; this is intentional.

Even with training-only method selection, treat the validation walk-forward results as **validation**, not as a fresh independent test. The forecasting candidates and process have been examined during model development. The untouched test set has not been used.

## Static forecast outputs

The main diagnostics and selection files include:

- `history_coverage.csv` — monthly history available for each asset.
- `validation_scores.csv` — validation performance by asset, method, and horizon.
- `combined_validation_scores.csv` — three-horizon selection scores and eligibility.
- `shrinkage_validation_comparison.csv` — comparison of expanding mean, selected method, and a 50/50 blend.
- `selected_methods.json` — full-validation method selections and selection details.
- `walk_forward_training_method_selection.json` — training-only selections used in the walk-forward files.
- `walk_forward_training_method_scores.csv` — candidate scores on the training sample.
- `walk_forward_training_shrinkage_comparison.csv` — training-only shrinkage comparison.

Mean-based forecasts repeat across horizons by construction. This is expected: an expanding or rolling mean does not change its prediction merely because the target month is farther away. The `h1_h3` file uses the same forecast origin for all three horizons; its later-horizon rows are not forecasts refreshed at later rebalance dates.

## Regime-dependent forecasts

`regime_conditional_returns.py` is a separate return-estimation script. It imports the team’s HMM code; it does not modify `hmm_regime_model.py`. It writes:

```text
Return_Forecasting/outputs/hmm_regime_conditional_returns.csv
```

The output includes HMM forecast dates and horizons, month-end state probabilities, state-conditional return estimates, and probability-weighted expected returns. The state-conditional estimates are conditional on the **month-end state**; they do not represent average state occupancy throughout the month.

For an optimizer that takes one expected-return vector, use the HMM’s **probability-weighted expected returns** for the matching rebalance date and horizon. An optimizer designed to explicitly use state-conditional inputs may instead use the conditional estimates alongside the state probabilities; confirm that this matches the optimizer’s design.

The HMM labels state 0 “bull” and state 1 “bear” based on lower versus higher Equity variance. These labels describe the estimated variance states; they do **not** guarantee that the state’s expected returns are positive or negative.

To compare static and HMM forecasts, match all three keys:

- `rebalance_date`
- `month_ahead`
- `forecast_month`

For a one-month monthly-rebalancing decision, compare the `month_ahead = 1` forecasts from the same rebalance date. An HMM `month_ahead = 2` forecast made at an earlier rebalance is not the same forecast as a new `month_ahead = 1` estimate made using updated information at the later rebalance.

## Updating the forward handoff

To produce the next monthly forecast, update the allowed train/validation inputs through the latest **completed** month and rerun the static script. The default forward rebalance date is the first XNYS trading session after the latest complete validation month. The resulting one-month file uses the latest completed month as its information cutoff.

The script does not automatically download or create updated train/validation return splits. It also does not create a new forward vector for every future month from one run. The walk-forward files cover the validation dates available in the input data.

If `REQUESTED_REBALANCE_DATES` is set explicitly, each requested date must be after the method-selection cutoff and supported by the available completed-month data. The test set should remain separate and must not be added to the inputs.

## Current supplied data snapshot

The outputs shared with the team were generated from data through **May 31, 2024**. The forward one-month handoff was for the June 3, 2024 rebalance, forecast month June 2024:

| Asset | Expected monthly return |
|---|---:|
| Equity | 0.8866% |
| Bond | 0.1125% |
| BTC | 8.2147% |

The validation walk-forward sample ran from February 2022 through May 2024, with 28 monthly rebalance dates. Its three-horizon version should contain 84 rows: three forecast horizons per rebalance date.