# Covariance Estimation

This folder contains scripts for comparing covariance estimators and exporting an EWMA covariance candidate for portfolio-optimizer experiments. It is separate from the return-forecasting scripts and does not modify the teammate’s HMM code.

## Folder contents

```text
Covariance_Estimation/
├── README.md
├── covariance_diagnostics.py
├──export_ewma_optimizer_covariance.py
└── outputs/
```

Keep the two Python scripts together: `export_ewma_optimizer_covariance.py` imports functions from `covariance_diagnostics.py`.

## Data conventions and evaluation boundaries

- Assets are **Equity**, **Bond**, and **BTC**.
- Returns are monthly simple returns in decimals. Covariance entries are therefore in monthly decimal-return squared.
- The scripts use the project’s configured training and validation inputs. Keep those input paths pointing to the existing data files; generated outputs should go to this folder’s `outputs/` directory.
- In the current data split, the first complete common month is February 2015; January 2015 is partial. The training period ends in January 2022, and the validation period runs from February 2022 through May 2024.
- **Do not read or use `portfolio_test.csv`** when selecting methods or assessing results.

## Scripts

### `covariance_diagnostics.py`

Compares covariance estimators using walk-forward forecasts and writes diagnostic CSVs. The comparisons include rolling sample covariances, Ledoit–Wolf, constant-correlation shrinkage, and EWMA candidates. HMM covariance forecasts are included as a validation comparator when the HMM covariance file is available at the path configured in the script.

EWMA lambda selection uses training data only. The current selected candidate is **λ = 0.93**, chosen from the configured lambda grid by the lowest mean multivariate Gaussian negative log-likelihood (NLL) over common training rolling-origin forecasts. The training windows overlap, so this is a reasonable candidate—not proof that 0.93 is a precisely determined optimum.

### `export_ewma_optimizer_covariance.py`

Exports EWMA covariance forecasts in a monthly, walk-forward format for optimizer experimentation.

At each monthly forecast origin, it:

1. Uses the trailing 60 complete common monthly return vectors available **before** that origin.
2. Estimates a one-month covariance matrix using **λ = 0.93**.
3. Writes forecasts for horizons 1, 2, and 3.

The same covariance matrix is carried forward across all three horizons. This is an explicit **static-covariance/persistence assumption**; EWMA does not dynamically forecast how covariance changes across those future months. The matrices are not cumulative multi-month covariances or cross-month covariance estimates. The lambda is not retuned on validation data.

The validation walk-forward uses information available at each origin. Returns from earlier validation months may enter the trailing estimation window for later origins once those returns would have been observed.

## Output format

The EWMA CSVs use the HMM-compatible long format:

```text
rebalance_date,month_ahead,forecast_month,asset,Equity,Bond,BTC
```

Each combination of `rebalance_date` and `month_ahead` has three rows—one per `asset`. The `Equity`, `Bond`, and `BTC` columns contain that row of the 3 × 3 covariance matrix. In the current convention, `month_ahead` 1, 2, and 3 refer to the rebalance month and the following two months.

## Outputs

The scripts write their results under `Covariance_Estimation/outputs/`.

### Diagnostic outputs

Common diagnostic files include:

- `covariance_validation_monthly_scores.csv` — validation scores by month and method.
- `covariance_validation_summary.csv` — aggregate validation scores.
- `covariance_validation_paired_comparison.csv` — paired method comparisons on shared months.
- `covariance_matrix_diagnostics.csv` — matrix properties such as positive definiteness and condition number.
- `covariance_validation_coverage.csv` and `covariance_validation_exclusions.csv` — validation coverage and exclusions.
- `covariance_training_walk_forward_scores.csv` and `covariance_training_walk_forward_summary.csv` — training walk-forward scores.
- `covariance_training_walk_forward_paired_comparison.csv` and `covariance_training_walk_forward_coverage.csv` — paired training comparisons and coverage.
- `ewma_lambda_training_selection.csv` — training-only lambda selection results.
- `ewma_lambda_validation_comparison.csv` — validation comparison of the EWMA lambdas; this is diagnostic, not for retuning lambda.

### EWMA export outputs

- `covariances_ewma_lambda_0_93_walk_forward_validation.csv` — validation origins, with three forecast horizons per origin.
- `covariances_ewma_lambda_0_93_latest_available_origin.csv` — horizons 1–3 from the latest available origin.
- `covariances_ewma_lambda_0_93_walk_forward_origin_details.csv` — details of the forecast origins.
- `covariances_ewma_lambda_0_93_coverage.csv` and `covariances_ewma_lambda_0_93_exclusions.csv` — export coverage and exclusions.
- `covariances_ewma_lambda_0_93_metadata.json` — export settings and metadata.
- `ewma_lambda_training_selection_export_check.csv` — check of the training-only lambda selection used for the export.

## Running the scripts

From this folder, run:

```bash
python covariance_diagnostics.py
python export_ewma_optimizer_covariance.py
```

The exporter imports the diagnostic module, so keep both scripts in this folder. Check the path settings near the top of each script if the project’s data files or folder layout change.

## Interpreting the results

- Lower multivariate Gaussian NLL is better; a negative NLL is valid.
- Mahalanobis distance measures how far the realized return vector is from the forecast center relative to the covariance. By itself, it does not penalize a covariance matrix for being overly large; NLL also accounts for covariance size through its log-determinant term.
- Check matrix diagnostics, including positive definiteness, as well as forecast scores.
- Validation contains 28 monthly observations in the current split and has already been examined. Treat validation comparisons as exploratory, not as a fresh independent holdout or definitive proof of a winning method.
- No covariance method is finalized here. The EWMA export is a candidate to compare with the HMM and other covariance inputs in a downstream optimizer using the same return forecasts, constraints, and evaluation assumptions.
- These scripts produce covariance diagnostics and covariance forecasts; they do not produce expected-return forecasts or portfolio return/Sharpe results. They also do not blend EWMA and HMM covariances.