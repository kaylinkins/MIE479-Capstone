# Stage-switching revalidation: DCC-GARCH scenarios for every model

2022-02-01 to 2024-05-31; monthly rebalancing.

| Forecasts | H | Net return | Ann. Sharpe | Max drawdown | BTC | Equities | Bonds |
|---|---:|---:|---:|---:|---:|---:|---:|
| HMM | 1 | 24.16% | 0.994 | -15.17% | 4.72% | 24.29% | 70.99% |
| HMM | 3 | 24.12% | 0.992 | -15.18% | 4.74% | 23.88% | 71.38% |
| Jump Model | 1 | 31.32% | 1.100 | -15.17% | 8.43% | 35.79% | 55.78% |
| Jump Model | 3 | 31.14% | 1.094 | -15.19% | 8.43% | 35.57% | 56.00% |
| Regime regression A | 1 | 24.82% | 1.019 | -15.17% | 4.67% | 24.46% | 70.87% |
| Regime regression A | 3 | 24.82% | 1.018 | -15.20% | 4.67% | 24.21% | 71.12% |
| Regime regression B | 1 | 24.82% | 1.019 | -15.17% | 4.67% | 24.46% | 70.87% |
| Regime regression B | 3 | 24.82% | 1.018 | -15.20% | 4.67% | 24.21% | 71.12% |

Every row uses the exact same verified DCC-GARCH scenario cube at each origin. Only the scenario source changes from the previous switching run. Mean forecasts and forecast regime signals remain unchanged: native HMM, native Jump Model, and hard-selected conditional regression means with their original stage-specific month-end probabilities. Regime regression A/B retain the old HMM/DCC experiment slots; they now have identical inputs and portfolios. Static regression is excluded.

Bull stages enforce the max-Sharpe return floor and bear stages use minimum CVaR with no active floor. Every stage minimizes 0.5*CVaR95 plus 10-bps trading costs. The full horizon is solved jointly, but only stage one is executed; the schedule is reforecast at the next rebalance. Risk is summed stage CVaR, not nested/terminal-wealth CVaR. Portfolios start in cash, are fully invested and long-only, and cap BTC at 20%. Exact self-financing fees and daily drift apply; no terminal liquidation occurs. Scenario paths are not shifted, filtered, resampled or volatility-scaled. The matched reference covariance now comes from the DCC-GARCH scenarios for every variant.

Net return is cumulative, Sharpe is annualized from net daily returns with sample standard deviation and zero risk-free return, and maximum drawdown uses daily net wealth. Allocation columns average executed monthly target weights. conditional_sharpe.csv uses the same common HMM forecast month1-end labels for every model: the grouping does not alter any model's trading signal.

scenario_change_comparison.csv compares with the previous switching run on identical dates. The previously DCC-based regression portfolio is checked for unchanged results. stage_selections.csv exposes all signals and mean vectors; decisions/*.json saves plans before execution and identifies the verified source archive. No models are refitted, and no test returns are loaded. The original forecast-training provenance limitations remain. A short validation sample does not establish statistical superiority.
