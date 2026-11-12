ï»¿# HMM forecast inputs

Run `python hmm_regime_model.py` from this folder. The script reads the two
local return CSVs and writes only five files into `hmm_outputs/`:

| File | Used by |
|---|---|
| regime_probabilities.csv | Regime heuristics and optimizer/risk-aversion switching |
| expected_returns.csv | MVO and any CVaR expected-return constraints |
| covariances.csv | MVO MPO |
| cvar_scenarios.npz | CVaR MPO; array shape (10000, 3, 3), keyed by rebalance date |
| forecast_trading_days.csv | Both planners, to align monthly periods |

Assets are ordered Equity, Bond, BTC. Returns are decimals, not percentages.
Both analytic forecasts and scenarios use the same regularized Gaussian HMM.
The local `.diagnostic_dependencies/` folder contains Python libraries,
not diagnostic reports. It is excluded by `.gitignore` and should not be uploaded. On another machine, install dependencies with `python -m pip install -r requirements.txt`. No diagnostic files or plots are generated.



## Statistical jump alternative

Run `python jump_regime_model.py`. This short script imports the shared calendar,
monthly compounding and scenario functions from `hmm_regime_model.py`; keep both
Python files together. It writes the same five filenames and schemas under
`jump_outputs/`. Select either output directory when loading optimizer inputs.
It reads only training and validation CSVs, refitting before each validation month.

The discrete jump model minimizes feature distances plus a penalty for changing
state. Features are exponentially weighted returns, log downside deviation and
return/downside ratios (5, 20, 60-session half-lives), for all three assets.
Clipping and scaling use only the available history. The fixed penalty is 50,
with five reproducible initializations; it was not tuned on validation outcomes.
State return means and full covariance matrices are estimated from past assigned
returns, with historical-mean and diagonal-covariance shrinkage.

This is a simple **shared-regime adaptation**, not a reproduction of Shu, Yu and
Mulvey's asset-specific jump model plus supervised classifier, nor Shu and
Mulvey's sparse jump model plus Black-Litterman allocation. JM state assignments
are not posterior probabilities: `p_bear_today` is a hard 0/1 assignment. Future
probabilities come from an added empirical Markov transition model with one
pseudocount per transition cell. Their calibration is not established.
As with HMM, ?bear? means higher historical equity variance, not necessarily a
negative return. Gaussian daily emissions and daily state transitions generate
both monthly moments and joint scenarios; the regime is not fixed for a month.
These simplifying assumptions should be stated when presenting results.

References: Shu, Yu and Mulvey, *Dynamic asset allocation with asset-specific
regime forecasts*, https://doi.org/10.1007/s10479-024-06266-0; Shu and Mulvey,
*Dynamic Factor Allocation Leveraging Regime-Switching Signals*,
https://arxiv.org/abs/2410.14841. JM implementation:
https://github.com/Yizhan-Oliver-Shu/jump-models (`jumpmodels==0.1.1`).

## Training-tuned alternative

Run `python jump_regime_model_tuned.py`. Keep `jump_regime_model.py` and
`hmm_regime_model.py` beside it. Outputs use the same five-file schema in
`jump_tuned_outputs/`; original `jump_outputs/` and HMM outputs are preserved.
No saved diagnostic reports are generated.

The script evaluates jump penalties 10, 50 and 150 and historical-mean pooling
weights 0, 0.5 and 1. It uses the final 24 monthly origins within training
(February 2020-January 2022 for the current data), fitting strictly before each
origin and scoring only horizons entirely within training. The selection score
is a marginal Gaussian quasi-score for monthly means and variances, averaged
across assets and 1-3 month horizons; it is not a full mixture likelihood.
Overlapping horizons are not independent. Settings are frozen before validation;
the return model is then refitted monthly using only available past data.

For the current training data, selection chooses penalty 150 and pooling 1.
Thus expected daily returns are the expanding historical means in BOTH regimes;
jump regimes still determine covariance and transition forecasts. This is a
risk-regime model with pooled means, not evidence of predictable regime returns.
The fixed random seeds and multiple starts remain unchanged.

The tuned candidate modestly improved next-month validation mean errors and
bond interval coverage, but did not improve the overall 1-3 month validation
quasi-score. It remains an experimental alternative, not a replacement or a
claim of uniformly better forecasts. The validation period has already been
inspected repeatedly; training-only tuning does not make it a fresh holdout.
No test data is read or used for selecting settings.
