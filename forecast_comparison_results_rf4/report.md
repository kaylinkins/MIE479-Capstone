# Forecast validation comparison

Validation: 2022-02-01 through 2024-05-31; monthly rebalancing.

| Policy | Forecasts | Scenarios | H | Net return | Ann. Sharpe | Max drawdown | BTC | Equities | Bonds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Min CVaR | HMM | HMM | 1 | -6.52% | -0.351 | -15.14% | 0.11% | 3.13% | 96.76% |
| Min CVaR | HMM | HMM | 3 | -6.46% | -0.348 | -15.12% | 0.11% | 3.06% | 96.83% |
| Matched return | HMM | HMM | 1 | 28.31% | 0.831 | -25.16% | 19.96% | 19.25% | 60.79% |
| Matched return | HMM | HMM | 3 | 28.31% | 0.831 | -25.16% | 19.96% | 19.25% | 60.79% |
| Min CVaR | Jump Model | Jump Model | 1 | -4.80% | -0.247 | -15.46% | 0.57% | 7.76% | 91.67% |
| Min CVaR | Jump Model | Jump Model | 3 | -4.86% | -0.251 | -15.47% | 0.57% | 7.49% | 91.94% |
| Matched return | Jump Model | Jump Model | 1 | 29.66% | 0.760 | -27.53% | 20.00% | 49.26% | 30.74% |
| Matched return | Jump Model | Jump Model | 3 | 29.69% | 0.760 | -27.53% | 19.99% | 49.30% | 30.71% |
| Min CVaR | Regime regression | HMM | 1 | -6.52% | -0.351 | -15.14% | 0.11% | 3.13% | 96.76% |
| Min CVaR | Regime regression | HMM | 3 | -6.46% | -0.348 | -15.12% | 0.11% | 3.06% | 96.83% |
| Matched return | Regime regression | HMM | 1 | 28.34% | 0.839 | -25.16% | 19.62% | 17.81% | 62.57% |
| Matched return | Regime regression | HMM | 3 | 28.34% | 0.839 | -25.16% | 19.62% | 17.81% | 62.57% |
| Min CVaR | Regime regression | DCC GARCH | 1 | -3.29% | -0.158 | -15.17% | 0.27% | 9.41% | 90.32% |
| Min CVaR | Regime regression | DCC GARCH | 3 | -3.54% | -0.172 | -15.19% | 0.25% | 8.78% | 90.96% |
| Matched return | Regime regression | DCC GARCH | 1 | 28.14% | 0.854 | -25.62% | 18.55% | 18.98% | 62.47% |
| Matched return | Regime regression | DCC GARCH | 3 | 28.18% | 0.855 | -25.62% | 18.54% | 19.00% | 62.46% |
| Min CVaR | Static regression | DCC GARCH | 1 | -3.29% | -0.158 | -15.17% | 0.27% | 9.41% | 90.32% |
| Min CVaR | Static regression | DCC GARCH | 3 | -3.54% | -0.172 | -15.19% | 0.25% | 8.78% | 90.96% |
| Matched return | Static regression | DCC GARCH | 1 | 33.40% | 0.886 | -24.23% | 19.94% | 42.40% | 37.66% |
| Matched return | Static regression | DCC GARCH | 3 | 33.39% | 0.885 | -24.23% | 19.94% | 42.43% | 37.63% |

Net return is the cumulative realized return after trading fees. Sharpe is sqrt(252) times the mean net daily return divided by its sample standard deviation, using a zero risk-free rate. Drawdown is calculated from daily net wealth. Allocations are arithmetic averages of the executed monthly target weights, not future planned weights or averages of drifted daily holdings.

Both policies use CVaR95, risk aversion 0.5, 10-bps trading costs, full investment, long-only weights, a 20% BTC cap, initial cash, and no terminal liquidation. Both minimize summed stage CVaR plus costs; matched return additionally imposes the max-Sharpe floor (forecast mean dot reference weights minus 1e-6 per holding day). The reference compounds a 4% annual risk-free rate over each holding period. Reference covariance is calculated from the unshifted scenario ensemble.

HMM and Jump Model use their respective supplied mean forecasts and scenario archives. Regime regression selects bull or bear conditional mean vectors independently for each stage using the larger supplied month-end probability. It uses neither a probability-weighted mean nor the realized future regime. Its two variants use the same selected means with unchanged HMM or DCC-GARCH scenarios. Scenario paths are not filtered, reweighted, shifted, or rescaled to the selected regime. Static regression uses the supplied Static CSV, including its published method labels, with the exact same DCC-GARCH cube as the regime regression DCC variant. No extra volatility scaling or regime-based risk penalty is applied.

Changing mean forecasts alone cannot affect minimum CVaR; the equal-scenario minimum-CVaR pairs are checked for equal allocations and returns. The same principle explains duplicate rows in this table. Scenarios are stage-additive risk inputs, so preserved cross-month paths do not turn this optimizer into a nested or terminal-wealth risk model. H=1 and H=3 both rebalance every month.

Each scenario source uses 10,000 paths per origin. DCC scenarios are generated with seed 42 and the chronological origin index. Native scenarios retain their original joint path ordering. DCC retains its five-year rolling estimation window. The supplied native forecast files are not refitted; information dates and calendar alignment are checked, but their original model training and model-selection history cannot be independently proven from these exports. Test returns and post-validation forecasts are excluded.

See conditional_selections.csv for all selected regimes and both conditional vectors; expected_returns.csv for exact optimizer means; forecasts/*.npz for all raw cubes and mean arrays; decisions/*.json for plans saved before execution; portfolio_monthly.csv for individual allocations. portfolio_summary.csv also contains monthly CVaR and risk calibration diagnostics. Results cover only 28 validation months and do not establish statistical superiority.
