# Final validation: three risk models, H=1 versus a receding three-month horizon

This is the existing February 2022–May 2024 development validation period: **28 months, 586 trading days**. It has been used in earlier research, so it is not an untouched confirmation sample. Testing returns were not opened. Settings were fixed before this run, without a parameter sweep or retuning after the results.

The three models are Hybrid FHS, Hybrid DCC and Moment-Controlled Weighted Hybrid DCC. The last is abbreviated “Weighted controlled DCC” below. All use Equity/BTC GARCH, Bond EWMA (lambda .94), five-year fits, exactly 252 strictly prior observations for Bond drift, and common expected-return forecasts. The weighted challenger uses a 126-observation half-life and matches the innovation mean and full covariance to the uniform DCC reference pool.

## Fixed strategy and information timing

- **No 20% BTC cap:** long-only, fully invested, with each asset allowed up to 100%. No leverage or cash allocation.
- **H=3:** at each first trading day, generate 10,000 coherent simulated paths over the next three calendar months. Convert each path to three incremental monthly return vectors. Simulated GARCH/EWMA and DCC states continue across month boundaries.
- Jointly optimize the three planned allocations, execute **only month one**, apply exact realized drift and self-financing fees, then refit and reoptimize at the next month’s origin. Future planned allocations are discarded, not implemented as commitments.
- H=1 uses exactly the same next-month scenarios, constraints, costs and objective coefficients. It is the single-period comparator, not a three-month strategy.
- Objective: sum of common expected returns minus CVaR95 penalties and proportional trading costs. Return reward and risk aversion are both 1; cost is 10 bps on every bought/sold dollar. Initial investment is charged. No gross-return targets, CVaR budgets, forced terminal allocation or terminal liquidation.
- This is the Boyd-style convex planning approximation with **sum of stage CVaRs**, not CVaR of compounded terminal wealth or nested CVaR. Future planned trades ignore return-induced weight drift; actual execution uses exact drift. Future planned weights are common to all scenarios.

Each fit receives only training returns plus already elapsed validation returns, and every fitted timestamp is strictly before the rebalance date. Future trading dates come from the NYSE holiday calendar published in 2021, including early-close sessions. The 2022–2024 calendar agrees with all validation dates. The final May forecast includes June/July simulated months, but their outcomes are neither read nor scored.

Weighted covariance matching error never exceeds 2.33e-15; effective weighted innovation-pool size ranges from 362.8 to 362.8. DCC persistence a+b is at least .995 in 3 fits; near-boundary estimates are retained and disclosed, without post-result model changes.

## Risk forecasts, assessed before portfolio selection

All three risk models face identical observed outcomes. H=1 and H=3 have the same next-month distribution, so forecast quality is assessed once per model. Scoring each model on its own optimized allocations alone would mix forecasting quality with different portfolio exposures.

### Next-month distribution scores

Lower scores are better. Quantile scores average the 5%, 10% and 25% lower-return quantiles across assets. Returns are scaled using training-only standard deviations and each period’s day count for comparability; those scales do not alter scenarios or decisions. Energy uses common seeded Monte Carlo pairs; variogram uses exponent 0.5.

| Model | Normalized quantile score | Energy score | Variogram score |
| --- | --- | --- | --- |
| Hybrid FHS | 0.24322 | 1.52051 | 0.18497 |
| Hybrid DCC | 0.24309 | 1.51323 | 0.18054 |
| Weighted controlled DCC | 0.24095 | 1.51249 | 0.18129 |

### Marginal VaR95 calibration

Expected breaches at a calibrated 5% level are 1.4 per asset over 28 months. Exact binomial intervals and p values are descriptive iid diagnostics; small counts cannot establish calibration.

| Model | Asset | VaR breaches | VaR breach frequency | Breach CI95 lower | Breach CI95 upper | q5 normalized score | Mean squared standardized error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid FHS | Equity | 1 | 3.57% | 0.09% | 18.35% | 0.09258 | 0.88127 |
| Hybrid FHS | Bond | 2 | 7.14% | 0.88% | 23.50% | 0.15009 | 1.40680 |
| Hybrid FHS | BTC | 1 | 3.57% | 0.09% | 18.35% | 0.07887 | 0.77188 |
| Hybrid DCC | Equity | 1 | 3.57% | 0.09% | 18.35% | 0.09258 | 0.88127 |
| Hybrid DCC | Bond | 2 | 7.14% | 0.88% | 23.50% | 0.15059 | 1.40383 |
| Hybrid DCC | BTC | 1 | 3.57% | 0.09% | 18.35% | 0.07924 | 0.78164 |
| Weighted controlled DCC | Equity | 1 | 3.57% | 0.09% | 18.35% | 0.09103 | 0.87465 |
| Weighted controlled DCC | Bond | 2 | 7.14% | 0.88% | 23.50% | 0.14956 | 1.40616 |
| Weighted controlled DCC | BTC | 1 | 3.57% | 0.09% | 18.35% | 0.08042 | 0.79448 |

Mean squared standardized error above 1 suggests forecasts are too narrow, miss conditional means, or encounter heavy tail realizations; it is a diagnostic rather than an isolated variance-estimator test.

### Joint downside calibration

| Model | Predicted_probability | Observed_frequency | Predicted_count | Observed_count | Brier_score |
| --- | --- | --- | --- | --- | --- |
| Hybrid FHS | 10.82% | 25.00% | 3.02830 | 7 | 0.20767 |
| Hybrid DCC | 12.67% | 25.00% | 3.54870 | 7 | 0.20182 |
| Weighted controlled DCC | 13.32% | 25.00% | 3.73070 | 7 | 0.20016 |

“All negative” means all three monthly asset returns are below zero. It is a sign-dependence event, not a 5% tail-loss event. Full joint tables also assess each pair and simultaneous outcomes below the same DCC 10%/25% marginal thresholds. Shared thresholds keep realized events identical across models.

![Monthly joint downside forecasts](joint_negative_monthly.png)

### Identical holdings diagnostic

Every model values the same fixed holdings each month. These allocations are diagnostics, not extra optimized model contenders.

| Model | Allocation | VaR breaches | Mean forecast CVaR | Normalized quantile score | Mean ES identification moment |
| --- | --- | --- | --- | --- | --- |
| Hybrid FHS | 60/40 | 4 | 8.12% | 0.12466 | 0.00716 |
| Hybrid FHS | 40/40/20 | 2 | 10.00% | 0.11537 | -0.00668 |
| Hybrid FHS | Equal thirds | 2 | 13.71% | 0.09807 | 0.00308 |
| Hybrid DCC | 60/40 | 3 | 8.40% | 0.12101 | 0.01219 |
| Hybrid DCC | 40/40/20 | 2 | 10.67% | 0.11273 | 0.00270 |
| Hybrid DCC | Equal thirds | 2 | 14.37% | 0.09565 | 0.01335 |
| Weighted controlled DCC | 60/40 | 3 | 7.90% | 0.12081 | 0.00720 |
| Weighted controlled DCC | 40/40/20 | 2 | 10.15% | 0.11007 | 0.00007 |
| Weighted controlled DCC | Equal thirds | 2 | 13.87% | 0.09405 | 0.01074 |

ES identification moment = ES − VaR − max(realized loss − VaR, 0)/0.05. Its conditional expectation is zero when both tail forecasts are correct. A negative average is consistent with tail underestimation, but 28 observations do not permit precise ES validation.

### Forecasts beyond month one

Month 2 and Month 3 are incremental returns for those individual months, not cumulative prefixes. Cumulative 3 months compounds each asset along a simulated path and is a separate risk diagnostic, not the optimizer’s objective. Only fully observed validation outcomes are included: 27 origins for Month 2 and 26 for Month 3/cumulative 3 months. These observations overlap and are not independent.

| Forecast period | Model | Normalized quantile score | Energy score | Variogram score |
| --- | --- | --- | --- | --- |
| Month 2 | Hybrid FHS | 0.25755 | 1.53447 | 0.18600 |
| Month 2 | Hybrid DCC | 0.25775 | 1.53002 | 0.18374 |
| Month 2 | Weighted controlled DCC | 0.25322 | 1.52632 | 0.18780 |
| Month 3 | Hybrid FHS | 0.26402 | 1.56680 | 0.18808 |
| Month 3 | Hybrid DCC | 0.26480 | 1.56416 | 0.18553 |
| Month 3 | Weighted controlled DCC | 0.26156 | 1.56089 | 0.19195 |
| Cumulative 3 months | Hybrid FHS | 0.26411 | 1.56607 | 0.22034 |
| Cumulative 3 months | Hybrid DCC | 0.26409 | 1.56183 | 0.20947 |
| Cumulative 3 months | Weighted controlled DCC | 0.26290 | 1.56977 | 0.21513 |

| Forecast period | Model | Periods | Predicted_probability | Observed_frequency | Brier_score |
| --- | --- | --- | --- | --- | --- |
| Month 2 | Hybrid FHS | 27 | 10.88% | 25.93% | 0.21773 |
| Month 2 | Hybrid DCC | 27 | 12.46% | 25.93% | 0.21513 |
| Month 2 | Weighted controlled DCC | 27 | 13.28% | 25.93% | 0.21274 |
| Month 3 | Hybrid FHS | 26 | 11.19% | 26.92% | 0.22802 |
| Month 3 | Hybrid DCC | 26 | 12.54% | 26.92% | 0.22820 |
| Month 3 | Weighted controlled DCC | 26 | 13.58% | 26.92% | 0.22674 |
| Cumulative 3 months | Hybrid FHS | 26 | 9.64% | 19.23% | 0.17209 |
| Cumulative 3 months | Hybrid DCC | 26 | 11.19% | 19.23% | 0.17350 |
| Cumulative 3 months | Weighted controlled DCC | 26 | 12.06% | 19.23% | 0.17313 |

## Realized portfolio performance: six combinations

All returns below follow monthly buy-and-hold execution of the first allocation. Sharpe uses daily net returns, 252-day annualization and zero risk-free rate. CAGR uses the same trading-day convention. Empirical monthly CVaR95 is a fractional average over only 1.4 tail months, so its estimate is imprecise. Drawdown includes the initial wealth point. Costs are deducted at actual rebalance dates.

| Model | Horizon | Net total return | Net CAGR | Net Sharpe zero risk-free | Net maximum drawdown | Net empirical monthly CVaR95 | Net VaR breaches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid FHS | 1 | -3.46% | -1.50% | -0.153 | -16.79% | 5.04% | 4 |
| Hybrid FHS | 3 | -3.60% | -1.56% | -0.162 | -16.77% | 5.01% | 4 |
| Hybrid DCC | 1 | -3.39% | -1.47% | -0.157 | -16.76% | 4.85% | 4 |
| Hybrid DCC | 3 | -3.59% | -1.56% | -0.168 | -16.72% | 4.84% | 4 |
| Weighted controlled DCC | 1 | -1.98% | -0.86% | -0.073 | -17.05% | 4.95% | 4 |
| Weighted controlled DCC | 3 | -2.21% | -0.96% | -0.086 | -17.06% | 4.96% | 4 |

| Model | Horizon | Gross total return | Net annualized volatility | Cumulative one-way turnover | Average Equity | Average Bond | Average BTC | Max BTC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid FHS | 1 | -3.10% | 7.86% | 1.34237 | 18.23% | 80.29% | 1.48% | 7.48% |
| Hybrid FHS | 3 | -3.25% | 7.85% | 1.31323 | 17.77% | 80.76% | 1.47% | 7.43% |
| Hybrid DCC | 1 | -3.01% | 7.61% | 1.46509 | 15.39% | 83.47% | 1.14% | 6.52% |
| Hybrid DCC | 3 | -3.22% | 7.61% | 1.41920 | 14.90% | 83.99% | 1.12% | 6.41% |
| Weighted controlled DCC | 1 | -1.58% | 7.75% | 1.55572 | 16.65% | 81.59% | 1.76% | 6.71% |
| Weighted controlled DCC | 3 | -1.82% | 7.75% | 1.51684 | 16.17% | 82.08% | 1.74% | 6.70% |

Cumulative one-way turnover excludes initial investment; transaction fees include it. The 100% bounds are available to the optimizer even when the chosen BTC exposure is small.

![Net portfolio wealth](net_wealth.png)

![Three-month strategy allocations](h3_allocations.png)

### Portfolio tail calibration

Net forecasts apply the known exact rebalance fee to every scenario before computing VaR/ES. Net realized outcomes include that same fee. Gross forecasts are compared with gross outcomes separately, avoiding a cost/threshold mismatch.

| Model | Horizon | Gross VaR breaches | Net VaR breaches | Net Breach CI95 lower | Net Breach CI95 upper | Net Coverage exact p value | Net quantile score | Mean net ES identification moment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid FHS | 1 | 4 | 4 | 4.03% | 32.67% | 0.04907 | 0.00307 | -0.01689 |
| Hybrid FHS | 3 | 4 | 4 | 4.03% | 32.67% | 0.04907 | 0.00303 | -0.01632 |
| Hybrid DCC | 1 | 4 | 4 | 4.03% | 32.67% | 0.04907 | 0.00278 | -0.00857 |
| Hybrid DCC | 3 | 4 | 4 | 4.03% | 32.67% | 0.04907 | 0.00275 | -0.00822 |
| Weighted controlled DCC | 1 | 4 | 4 | 4.03% | 32.67% | 0.04907 | 0.00289 | -0.01299 |
| Weighted controlled DCC | 3 | 4 | 4 | 4.03% | 32.67% | 0.04907 | 0.00288 | -0.01287 |

### What the three-month horizon changes

H=3 minus H=1 uses the same risk-model scenarios and changes only the planning horizon and subsequent path of holdings. CAGR differences below are in basis points per year. A 1% L1 allocation difference means the sum of absolute asset-weight changes exceeds one percentage point.

| Model | Net CAGR difference | Net Sharpe difference | Net CVaR difference | Turnover difference | Mean first weight L1 difference | Months weight L1 differs above 1pct |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid FHS | -6.37 | -0.00862 | -0.03% | -0.02914 | 1.14% | 15 |
| Hybrid DCC | -8.90 | -0.01185 | -0.01% | -0.04589 | 1.34% | 15 |
| Weighted controlled DCC | -9.88 | -0.01290 | 0.01% | -0.03888 | 1.35% | 17 |

Future CVaR distributions and trading costs couple the planned stages. Extending the horizon need not materially change today’s allocation when all three stages prefer similar exposures. This formulation has no scenario-contingent future trading policy and no terminal-wealth risk term.

## Uncertainty and limits of this final development comparison

Paired circular block bootstrap: three-month blocks, 5,000 replications, fixed seed. Intervals are exploratory and unadjusted for multiple comparisons or the project’s earlier validation reuse. Paired score differences are candidate minus Hybrid DCC; negative favors the candidate.

| Model | Metric | Mean difference | CI95 lower | CI95 upper |
| --- | --- | --- | --- | --- |
| Hybrid FHS | Energy score | 0.00728 | 0.00251 | 0.01262 |
| Hybrid FHS | Variogram score | 0.00443 | -0.00272 | 0.01215 |
| Weighted controlled DCC | Energy score | -0.00074 | -0.00972 | 0.00725 |
| Weighted controlled DCC | Variogram score | 0.00075 | -0.00210 | 0.00335 |

The following portfolio intervals annualize bootstrapped monthly compounded returns using 12 months/year. Reported point CAGR above uses 252 trading days/year, so these intervals use a slightly different annualization convention. They are descriptive uncertainty estimates, not proof that the backtest is an independent confirmation.

| Model | Horizon | Reference | Reference horizon | Net CAGR difference | Monthly annualized CAGR difference CI95 lower | Monthly annualized CAGR difference CI95 upper |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid FHS | 3 | Hybrid FHS | 1 | -0.06% | -0.17% | 0.03% |
| Hybrid DCC | 3 | Hybrid DCC | 1 | -0.09% | -0.22% | 0.01% |
| Weighted controlled DCC | 3 | Weighted controlled DCC | 1 | -0.10% | -0.23% | 0.02% |
| Hybrid FHS | 1 | Hybrid DCC | 1 | -0.03% | -0.71% | 0.72% |
| Weighted controlled DCC | 1 | Hybrid DCC | 1 | 0.61% | -0.26% | 1.57% |
| Hybrid FHS | 3 | Hybrid DCC | 3 | -0.01% | -0.66% | 0.74% |
| Weighted controlled DCC | 3 | Hybrid DCC | 3 | 0.60% | -0.25% | 1.58% |

## Month-level loss diagnosis

The seven realized all-negative months, with asset returns and the probability each model assigned before that month began:

| Origin | Equity | Bond | BTC | HybridFHS | HybridDCC | MomentControlledWeightedHybridDCC |
| --- | --- | --- | --- | --- | --- | --- |
| 2022-04-01 | -8.72% | -3.81% | -15.26% | 7.34% | 8.42% | 8.86% |
| 2022-06-01 | -8.25% | -1.55% | -39.26% | 11.80% | 12.83% | 13.21% |
| 2022-08-01 | -4.08% | -3.04% | -16.39% | 11.16% | 14.06% | 15.18% |
| 2022-09-01 | -9.21% | -4.14% | -2.94% | 13.77% | 17.78% | 18.19% |
| 2022-12-01 | -5.76% | -0.87% | -3.40% | 14.86% | 22.24% | 23.26% |
| 2023-08-01 | -1.59% | -0.63% | -11.17% | 9.38% | 10.05% | 11.02% |
| 2024-04-01 | -4.08% | -2.48% | -14.39% | 8.08% | 7.25% | 7.74% |

Hybrid DCC H=3 portfolio breaches and gross asset contributions (weight times realized monthly asset return):

| Origin | Net period return | Forecast net VaR | Forecast net CVaR | Equity contribution | Bond contribution | BTC contribution | Bond share of gross loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-04-01 | -4.91% | 3.24% | 4.69% | -1.20% | -3.14% | -0.56% | 64.16% |
| 2022-09-01 | -4.66% | 4.39% | 5.88% | -0.93% | -3.72% | -0.00% | 79.96% |
| 2023-09-01 | -2.99% | 2.92% | 4.12% | -0.82% | -2.14% | 0.00% | 72.21% |
| 2024-04-01 | -2.86% | 1.85% | 2.80% | -0.67% | -2.05% | -0.14% | 71.89% |

Asset contributions sum to gross return; exact trading fees reconcile this to net return. The full [breach attribution](portfolio_breach_diagnosis.csv) contains all six strategies; [joint-loss monthly data](joint_loss_months.csv) includes all 28 months.

## Interpretation and decision

The implementation satisfies the requested information timing and receding-horizon mechanics. The economic and calibration results are mixed: **DCC improves some joint-distribution scores, the weighted challenger adds small improvements, but no model fixes portfolio tail coverage or produces a profitable validation strategy under this fixed objective.**

### What succeeded in the risk models

Hybrid DCC improves next-month energy score from 1.52051 to 1.51323 relative to Hybrid FHS. The exploratory paired block-bootstrap interval for FHS minus DCC is [0.00251, 0.01262], favoring DCC. DCC also has the lowest next-month variogram score and improves fixed-holdings quantile scores across all three diagnostic allocations. Its contribution is therefore visible when holdings are held constant; it is not merely a result of selecting different portfolios.

Weighted controlled DCC has the lowest next-month average lower-quantile score (0.24095 versus DCC's 0.24309), energy score (1.51249 versus 1.51323), and all-negative Brier score (0.20016 versus 0.20182). These are small improvements: approximately 0.88% in the quantile score and 0.05% in energy. Its energy and joint-negative Brier paired intervals against DCC both include zero. Its variogram score is worse than DCC's, and its cumulative-three-month energy score is also worse. The evidence supports keeping it as a challenger, not claiming a uniformly superior dependence model.

All three models have the same marginal VaR breach counts: Equity 1/28, Bond 2/28, BTC 1/28. These counts do not show severe asset-by-asset 5% undercoverage, but the intervals are wide. Bond remains the weakest marginal: mean squared standardized error is approximately 1.40, versus approximately 0.88 for equities and 0.77–0.79 for BTC. That finding is inconsistent with a general explanation that every asset's volatility is too low. Bond scale/shape and changing downside dependence remain more plausible contributors; these diagnostics cannot isolate their causal shares.

### What remains unresolved

Seven of 28 months have all three assets losing. The models predict only 3.03, 3.55 and 3.73 such months on average, respectively: 10.82%, 12.67% and 13.32% against 25% observed. The weighted innovation distribution moves in the desired direction while controlling its first two moments, but the residual discrepancy is large. Marginal sign calibration also matters: DCC predicts negative Bond months with average probability 56.35%, while 64.29% are observed. The joint-sign discrepancy is not a pure test of tail dependence or a proof that correlation alone is responsible.

Every optimized strategy breaches its monthly portfolio VaR in **the same four months**: April 2022, September 2022, September 2023 and April 2024. Thus the differences in scenario models and planning horizon are insufficient to move these events across their respective portfolio thresholds. Four breaches are 14.29%, against 5% intended. The unadjusted two-sided exact test gives p=0.0491; the conservative equal-tailed 95% interval is 4.03%–32.67%. These two exact procedures are not inversions of each other, so the interval containing 5% is not a calculation error. Given discreteness, validation reuse and multiple comparisons, treat this as persistent undercoverage with substantial uncertainty, not a precise estimate of the true failure rate.

The month-level attribution clarifies why portfolio coverage can be worse than individual-asset coverage. For Hybrid DCC H=3, Bond contributes 64%–80% of the gross loss in all four breach months, with equity losses adding to it. September 2023 is a portfolio breach even though BTC is positive. Capturing the all-three-negative event alone therefore cannot solve portfolio tail calibration. The actual portfolio loss distribution, particularly co-losses in the dominant Bond/equity holdings, is the relevant target. The negative average optimized-portfolio ES identification moments also give no basis to declare tail risk solved.

### Why portfolio results still differ little

The removed BTC cap was not binding in this run: the largest chosen BTC weight is 7.48%, and average BTC weights are only 1.12%–1.76%. The six strategies hold approximately 80%–84% Bond on average. With common expected-return forecasts and the same risk/return coefficient, the models prefer similar allocations because Bond has much smaller forecast downside than the other assets. Removing an inactive 20% bound cannot create a material change in these solutions.

Bond's validation buy-and-hold return is -7.71%, compared with +21.38% for equities and +75.18% for BTC. These are descriptive asset outcomes, not alternative strategies selected after observing returns. The portfolios' large Bond exposure helps explain their weak participation in the recovery and their losses during joint Bond/equity declines. Their 2022 losses are approximately 13%–13.5%, and their subsequent gains do not fully recover them.

H=3 changes the first allocation's L1 distance by only 1.14–1.35 percentage points on average relative to H=1. It modestly reduces turnover, but net annualized return falls by 6.37, 8.90 and 9.88 basis points for FHS, DCC and weighted DCC. All corresponding paired CAGR intervals include zero. This is a functioning multi-period optimizer with small economic benefit from looking ahead under the current forecasts and cost scale; it is not evidence that multi-period optimization generally cannot help. Stagewise risk preferences remain similar, costs are small, future trades have approximate weight dynamics, and the objective is not terminal-wealth risk.

Weighted controlled DCC H=3 has the highest net total return among the required three-month strategies, **-2.21%**, versus **-3.59%** for Hybrid DCC and **-3.60%** for Hybrid FHS. Its approximately 0.60 percentage-point annualized advantage over DCC has an exploratory monthly-bootstrap interval from roughly -0.25 to +1.58 percentage points. It also has a slightly worse maximum drawdown and realized monthly CVaR than DCC. The highest return in this comparison is still a loss and does not establish robust superiority.

### Recommendation before the untouched test

Keep **Hybrid FHS H=3 as the benchmark, Hybrid DCC H=3 as the primary specification, and weighted controlled Hybrid DCC H=3 as the conditional challenger**. H=1 is the diagnostic comparator for the value of planning ahead. Hybrid DCC has the clearer forecast-score improvement over FHS; the weighted variant has promising point estimates but does not establish a reliable additional gain or fix the calibration problems. This retains the prior roles on evidence, rather than choosing the least negative validation return as a proven winner.

Freeze these definitions, the unconstrained BTC policy, the three-month receding horizon, costs, objective and scoring before opening testing returns. A later test comparison should report all prespecified outcomes once, without switching the primary model or retuning parameters in response to test performance. The defensible capstone conclusion at this stage is that joint forecasting improved in some respects, while risk calibration and realized portfolio benefits remained limited under the specified optimization policy. The test is a confirmation assessment, not a remedy for weak validation results.

## Artifacts and reproduction

- [Fixed validation protocol](validation_protocol.json) and [forecast run manifest](forecast_run_manifest.json).
- [Information cutoff audit](information_audit.csv), including every prior-data cutoff, fitted-state convergence flag and simulation clipping/cap diagnostic.
- [Portfolio summary](portfolio_summary.csv), [monthly decisions and outcomes](portfolio_monthly.csv), [daily net/gross returns](portfolio_daily.csv), [all planned allocations](planned_allocations.csv), and [paired strategy comparisons](portfolio_paired_comparisons.csv).
- [Marginal calibration](marginal_summary.csv), [joint event calibration](joint_summary.csv), [proper scores](proper_scores_summary.csv), [fixed-holdings diagnostics](fixed_allocation_summary.csv), and [paired score uncertainty](paired_forecast_scores.csv).
- [Month-by-month joint forecasts](joint_monthly.csv), [scenario means](scenario_mean_audit.csv), [independent verification](verification.json), and [regression test log](regression_tests.log).

Run from the project root (forecast generation reuses verified, matching caches):

```powershell
& .venv/Scripts/python.exe "Volatility Modelling/experiments/multiperiod/validation_forecasts.py"
& .venv/Scripts/python.exe "Volatility Modelling/experiments/multiperiod/validation_evaluation.py"
& .venv/Scripts/python.exe "Volatility Modelling/experiments/multiperiod/verify_validation_run.py"
& .venv/Scripts/python.exe "Volatility Modelling/experiments/multiperiod/validation_diagnosis.py"
& .venv/Scripts/python.exe "Volatility Modelling/experiments/multiperiod/validation_report.py"
```

Method references: [Boyd et al., multi-period trading](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf), [NYSE 2022–2024 schedule published in 2021](https://ir.theice.com/press/news-details/2021/NYSE-Group-Announces-2022-2023-and-2024-Holiday-and-Early-Closings-Calendar/default.aspx), and [SciPy exact binomial tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
