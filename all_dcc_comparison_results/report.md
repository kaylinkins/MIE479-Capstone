# All-DCC-GARCH fixed-policy validation comparison

2022-02-01 to 2024-05-31; monthly rebalancing.

| Policy | Forecasts | Scenarios | H | Net return | Ann. Sharpe | Max drawdown | BTC | Equities | Bonds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Min CVaR | HMM | DCC-GARCH | 1 | -3.29% | -0.158 | -15.17% | 0.27% | 9.41% | 90.32% |
| Min CVaR | HMM | DCC-GARCH | 3 | -3.54% | -0.172 | -15.19% | 0.25% | 8.78% | 90.96% |
| Matched return | HMM | DCC-GARCH | 1 | 22.04% | 0.722 | -25.25% | 17.50% | 18.42% | 64.09% |
| Matched return | HMM | DCC-GARCH | 3 | 21.97% | 0.720 | -25.25% | 17.49% | 18.61% | 63.90% |
| Min CVaR | Jump Model | DCC-GARCH | 1 | -3.29% | -0.158 | -15.17% | 0.27% | 9.41% | 90.32% |
| Min CVaR | Jump Model | DCC-GARCH | 3 | -3.54% | -0.172 | -15.19% | 0.25% | 8.78% | 90.96% |
| Matched return | Jump Model | DCC-GARCH | 1 | 32.52% | 0.946 | -21.42% | 18.02% | 35.01% | 46.97% |
| Matched return | Jump Model | DCC-GARCH | 3 | 32.44% | 0.944 | -21.43% | 18.05% | 34.81% | 47.14% |
| Min CVaR | Regime regression A | DCC-GARCH | 1 | -3.29% | -0.158 | -15.17% | 0.27% | 9.41% | 90.32% |
| Min CVaR | Regime regression A | DCC-GARCH | 3 | -3.54% | -0.172 | -15.19% | 0.25% | 8.78% | 90.96% |
| Matched return | Regime regression A | DCC-GARCH | 1 | 20.06% | 0.671 | -25.23% | 17.01% | 13.45% | 69.55% |
| Matched return | Regime regression A | DCC-GARCH | 3 | 19.95% | 0.668 | -25.22% | 17.04% | 13.33% | 69.63% |
| Min CVaR | Regime regression B | DCC-GARCH | 1 | -3.29% | -0.158 | -15.17% | 0.27% | 9.41% | 90.32% |
| Min CVaR | Regime regression B | DCC-GARCH | 3 | -3.54% | -0.172 | -15.19% | 0.25% | 8.78% | 90.96% |
| Matched return | Regime regression B | DCC-GARCH | 1 | 20.06% | 0.671 | -25.23% | 17.01% | 13.45% | 69.55% |
| Matched return | Regime regression B | DCC-GARCH | 3 | 19.95% | 0.668 | -25.22% | 17.04% | 13.33% | 69.63% |
| Min CVaR | Static regression | DCC-GARCH | 1 | -3.29% | -0.158 | -15.17% | 0.27% | 9.41% | 90.32% |
| Min CVaR | Static regression | DCC-GARCH | 3 | -3.54% | -0.172 | -15.19% | 0.25% | 8.78% | 90.96% |
| Matched return | Static regression | DCC-GARCH | 1 | 29.37% | 0.899 | -21.76% | 18.04% | 27.53% | 54.44% |
| Matched return | Static regression | DCC-GARCH | 3 | 29.38% | 0.899 | -21.76% | 18.05% | 27.40% | 54.54% |

Regime regression A is the previous HMM-scenario slot; B is the previous DCC-GARCH slot. Both now use identical DCC-GARCH cubes and identical stage-selected conditional means, so they are duplicate configurations. Both labels are retained to reproduce the requested 20 rows (16 distinct forecast/policy/horizon combinations). All minimum-CVaR rows at a given horizon also coincide, since their separate mean vectors do not affect that policy.

This reruns the preceding 20 FIXED-policy configurations, not the eight stage-switching portfolios. Minimum CVaR is used throughout its plan; matched return imposes floors at all stages, including forecast bear stages. The static benchmark remains non-switching. HMM/Jump native mean vectors, hard-selected conditional regression means, and static means are preserved exactly. Only the scenario source is replaced for formerly non-DCC configurations. Scenario covariance used by the Sharpe reference now comes from DCC-GARCH for every model.

The exact same verified DCC cubes are used for every configuration, with no resampling, regime filtering, mean translation, volatility overlay or changes to forecast means. Source archives and all producer inputs/code/artifacts are checksum-verified before reuse. Each account begins independently in cash and incurs its own fees. Test returns are never loaded. The supplied forecast files are not refitted.

Settings: H=1/H=3, CVaR95, risk aversion 0.5, no separate return reward, costs 10 bps, long-only full investment, BTC cap 20%, no terminal liquidation. Monthly reoptimization executes only the first allocation, with exact self-financing fees and daily holding drift. Matched return uses forecast mean dot max-Sharpe weights minus 1e-6 per holding day. Risk is summed stage CVaR, not nested/terminal-wealth CVaR.

Net return is cumulative net realized return. Sharpe is sqrt(252)*mean(net daily returns)/sample standard deviation at a zero risk-free rate. Drawdown uses daily net wealth. Allocations average executed monthly target weights. The source comparison provides 10,000 joint three-month DCC paths per origin under the original draw convention. The original forecast-training provenance limitations remain; performance over 28 validation months does not establish statistical superiority.

expected_returns.csv records actual unchanged means and original scenario labels. scenario_change_comparison.csv reports changes from the preceding comparison using the same dates. portfolio_monthly.csv and portfolio_daily.csv contain weights, fees and outcomes. decisions/*.json records plans before execution and identifies the existing source forecast archive. Full scenario cubes remain in the verified source run identified by source_forecast_results in protocol.json. Earlier outputs are preserved.
