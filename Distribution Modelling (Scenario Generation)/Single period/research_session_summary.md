# Research session summary: from four risk models to a final shortlist

## Scope, starting question and how to read this summary

The starting question was why improvements in volatility and joint-distribution forecasts did not consistently reduce portfolio VaR breaches or improve CVaR-optimized investment performance. This summary follows the research sequence documented in the saved results: the original four-model comparison; marginal, joint-loss and allocation diagnostics; whitening and bond-mean controls; asymmetric dependence; recent correlation; and the five-step horizon/innovation/copula/portfolio investigation. The original comparison is included as the starting evidence, rather than being presented as a newly run experiment at every subsequent stage.

All reported development results use **February 2022–May 2024: 28 nonoverlapping monthly holding periods and 586 daily observations**. Scenario experiments use 10,000 paths per monthly origin and prior five-year estimation windows. The development period was not expanded and untouched test returns were not used. Completed experiments are distinguished from the final conditional challenger, which has been recommended but has not been tested.

Forecast horizons match each month's observed trading-day count (19–23 days). Earlier fixed-21-day outputs are archived and excluded from the tables below. Calendar counts follow the project's retrospective month convention; a live implementation would need an exchange calendar known at the rebalance. Shared seeds/shock rows are used where the experimental design permits. Fitting uses information strictly before each rebalance.

A **marginal forecast** is one asset's distribution. **Dependence** describes how scenario outcomes are paired across assets. A **fixed-weight comparison** values identical holdings under different forecast distributions; its realized returns are identical. **Reoptimization** lets the distribution change the holdings and therefore the realized returns. These answer different questions.

All-negative monthly returns are a joint **sign** event, not necessarily an extreme-tail event. Quantile, Brier, energy and variogram scores are losses: lower is better. Joint Brier comparisons require the same realized event definition. Energy/variogram values should be compared within their own experiment: scoring draws, implementations or normalizations can differ between stages. Gross returns exclude fees; net returns include them. Total returns below cover the entire validation period, rather than an annual rate.

## 1. Starting point: four model families and the original portfolio puzzle

The four original specifications were: `GARCH-FHS` (GARCH for all assets); `GARCH-EWMA-GARCH-FHS` (Equity/BTC GARCH and Bond EWMA); `DCC-GARCH Gaussian` (all-GARCH marginals plus DCC); and `DCC-GARCH EWMA GARCH` (hybrid marginals plus DCC). Both DCC generators use Gaussian quasi-likelihood for fitting and empirical whitened innovations for scenarios. Their names do not mean that this stage already tested the later matched Gaussian-versus-Student-t copula experiment. An unfiltered Historical strategy was also retained as an auxiliary benchmark in the original comparison; the four filtered models were the focus of the subsequent investigations.

| Model | Bond 5% hits / 28 | Own Min-CVaR hits / 28 | All-negative probability | Energy score | Variogram score |
| --- | --- | --- | --- | --- | --- |
| FHS / all GARCH | 6 | 7 | 9.20% | 1.539469 | 0.183116 |
| Hybrid FHS | 4 | 4 | 9.31% | 1.517025 | 0.183001 |
| DCC / all GARCH | 6 | 7 | 11.03% | 1.534505 | 0.183145 |
| Hybrid DCC / five-year Bond mean | 4 | 4 | 11.01% | 1.510410 | 0.179973 |

[Unrounded table data](research_session_summary_tables/01_original_forecasts.csv).

The realized all-negative frequency was **7/28 = 25%**. The hybrid volatility specification improved point forecast scores and bond tail calibration. For example, all-GARCH DCC had six bond 5% hits and seven own minimum-CVaR portfolio hits; Hybrid DCC had four of each. This is evidence that the early model changes did matter. However, forecasts of all-negative returns remained around 9–11%.

| Model | Min-CVaR mean Bond weight | Target 3 net total return | Target 6 net total return | Matched-return gross Sharpe |
| --- | --- | --- | --- | --- |
| FHS / all GARCH | 86.79% | 11.11% | 37.93% | 0.924 |
| Hybrid FHS | 85.23% | 11.11% | 37.96% | 0.842 |
| DCC / all GARCH | 90.41% | 10.42% | 37.45% | 0.947 |
| Hybrid DCC / five-year Bond mean | 88.91% | 10.06% | 37.33% | 0.807 |

[Unrounded table data](research_session_summary_tables/02_original_portfolios.csv).

The auxiliary unfiltered Historical minimum-CVaR portfolio was almost entirely bonds (99.97% on average), returned -7.95% net, and had five breaches. Its full target/maximum-Sharpe/matched-return results remain in the [original comparison](cvar_comparison/comparison_report.md). Hybrid FHS is the more informative final simple benchmark for this research question because it shares adaptive marginal volatility with Hybrid DCC, allowing the role of DCC to be examined more cleanly.

The original common-target portfolios had similar returns across the filtered models. All-GARCH DCC had the highest matched-return gross Sharpe among these four (0.947), while simpler FHS was competitive (0.924) and slightly stronger at the common targets in this sample. The matched-return strategy used each model's own maximum-Sharpe expected-return target; it was not a comparison at an identical expected-return requirement across models.

**Interpretation and next question:** better estimated distributions need not produce better realized returns. Minimum-CVaR portfolios held roughly 85–90% bonds, while higher targets induced much larger equity/BTC exposures. We therefore needed to separate marginal errors, dependence errors and constraints on the optimizer, rather than attribute the whole problem to DCC.

## 2. First diagnosis: individual assets, individual loss months and fixed holdings

### 2.1 Bond location and scale were immediate suspects

| Model | Forecast_mean | Realized_mean | Forecast_negative_probability | Realized_negative_frequency |
| --- | --- | --- | --- | --- |
| FHS / all GARCH | 0.08% | -0.26% | 46.78% | 64.29% |
| Hybrid FHS | 0.08% | -0.26% | 47.47% | 64.29% |
| DCC / all GARCH | 0.08% | -0.26% | 46.38% | 64.29% |
| Hybrid DCC / five-year Bond mean | 0.09% | -0.26% | 47.14% | 64.29% |

[Unrounded table data](research_session_summary_tables/03_original_bond_location.csv).

All original bond forecasts had a small positive average monthly mean, around +0.08% to +0.09%, against -0.26% realized. They assigned negative bond returns probabilities near 47%, while 64.29% of the validation months were negative. A positive location error can reduce forecast joint-negative probabilities even if dependence is unchanged. Monthly mean/PIT diagnostics alone cannot separate location, variance and residual-shape errors, but they identified a concrete bond sensitivity to investigate.

### 2.2 The seven joint-loss months did not all have the same marginal surprise

Equity/BTC PITs below concern the **original Hybrid DCC**. Both original all-GARCH and hybrid Bond PITs are shown; the last column gives the **later current reference** probability after the bond-mean change. They are deliberately labelled as different stages. PIT is the forecast fraction below the realized return, so a small PIT denotes an unexpectedly poor marginal outcome.

| Rebalance | Original Equity PIT | Original all-GARCH Bond PIT | Original hybrid Bond PIT | Original BTC PIT | Current reference joint-negative probability |
| --- | --- | --- | --- | --- | --- |
| 2022-04-01 | 4.40% | 0.43% | 2.79% | 8.80% | 8.70% |
| 2022-06-01 | 8.82% | 11.74% | 18.95% | 2.68% | 12.45% |
| 2022-08-01 | 16.69% | 3.57% | 7.64% | 15.67% | 14.83% |
| 2022-09-01 | 6.05% | 1.44% | 2.99% | 33.88% | 16.91% |
| 2022-12-01 | 14.33% | 28.72% | 33.59% | 38.31% | 22.12% |
| 2023-08-01 | 17.99% | 31.49% | 32.57% | 14.86% | 10.58% |
| 2024-04-01 | 6.48% | 3.10% | 3.34% | 20.85% | 7.44% |

[Unrounded table data](research_session_summary_tables/04_realized_joint_loss_months.csv).

In April 2022, the Bond PIT was about 0.4% under all-GARCH DCC and 2.8% under Hybrid DCC; changing the bond volatility model had already reduced the severity of that surprise. December 2022 was an all-negative month with much less extreme marginal PITs. Thus a single mechanism such as 'all three assets had extreme shocks' could not explain every joint-loss event. Aggregate joint sign frequencies had hidden this distinction.

### 2.3 Holding weights fixed confirmed that risk thresholds changed

Every row below evaluates the original **all-GARCH FHS minimum-CVaR holdings**. Only the forecast model changes:

| Forecast model | VaR breaches | Mean forecast 95% VaR | Gross total return |
| --- | --- | --- | --- |
| DCC / all GARCH | 7 | 2.67% | -3.10% |
| Hybrid DCC / five-year Bond mean | 4 | 3.07% | -3.10% |
| FHS / all GARCH | 7 | 2.50% | -3.10% |
| Hybrid FHS | 5 | 2.91% | -3.10% |

[Unrounded table data](research_session_summary_tables/05_same_fhs_holdings.csv).

The change from seven to four breaches under Hybrid DCC occurs at the same holdings and realized returns. This isolates a change in assessed risk, rather than an investment-return gain. It also explains why a forecasting improvement could coexist with nearly unchanged portfolio performance.

### 2.4 Common means and binding constraints explained part of the allocation similarity

The first common-mean reoptimization translated every distribution to the all-GARCH FHS mean vector, preserving centered covariance and copulas. The moderate-target gross outcomes were:

| Model | Gross total return | VaR breaches | Mean half L1 distance from original allocation |
| --- | --- | --- | --- |
| DCC / all GARCH | 10.97% | 3 | 0.51% |
| Hybrid DCC / five-year Bond mean | 10.53% | 3 | 0.50% |
| FHS / all GARCH | 11.72% | 5 | 0.00% |
| Hybrid FHS | 11.64% | 4 | 0.15% |

[Unrounded table data](research_session_summary_tables/06_original_common_mean_reoptimization.csv).

These weights generally moved little from the original solutions; changing expected-return conventions did not remove the broad performance pattern. Higher-target solutions were even more constrained:

| Allocation model | BTC cap binding count | Return constraint binding count | Mean Bond weight | Mean BTC weight |
| --- | --- | --- | --- | --- |
| DCC-GARCH Gaussian | 22 | 28 | 5.54% | 19.77% |
| DCC-GARCH EWMA GARCH | 22 | 28 | 5.53% | 19.75% |
| GARCH-FHS | 24 | 28 | 5.54% | 19.85% |
| GARCH-EWMA-GARCH-FHS | 23 | 28 | 5.50% | 19.84% |

[Unrounded table data](research_session_summary_tables/07_original_high_target_constraints.csv).

At target 6 the return constraint bound in all 28 months, and the 20% BTC cap bound in 22–24 months. Full investment plus the binding return target substantially reduces the available allocation choices; a binding BTC cap can reduce them further. At minimum CVaR, the high bond exposure made bond marginal calibration particularly important. Higher-target portfolios' lower breach rates were therefore not proof that joint downside modelling had become accurate.

**How this informed the next experiments:** use fixed holdings to assess forecast changes, common means to control optimization incentives, and exact marginal controls to isolate dependence. Test the bond mean and innovation construction before adding further correlation complexity.

## 3. Controlled whitening and bond-mean sensitivities

The static-whitened DCC simulation was replayed using the original shock indices. Conditional whitening instead used the fitted historical correlation at each residual date, followed by pooled centering and symmetric covariance normalization. A rank-matched control restored every static monthly marginal value exactly; only cross-asset scenario pairing changed. Separately, zero and strictly prior-252-observation bond means changed simulated bond drift while leaving variance/correlation paths and shock indices fixed.

| Model | Variant | Bond 5% hits / 28 | Bond 5% score | Bond mean forecast | Joint-negative probability | Mean Brier score |
| --- | --- | --- | --- | --- | --- | --- |
| DCC-GARCH | static | 6 | 0.003139 | 0.08% | 11.03% | 0.209391 |
| DCC-GARCH | conditional | 6 | 0.003163 | 0.08% | 10.90% | 0.209344 |
| DCC-GARCH | conditional_rank_matched | 6 | 0.003139 | 0.08% | 10.94% | 0.209301 |
| DCC-GARCH | bond_mean_zero | 6 | 0.002920 | 0.00% | 11.46% | 0.207958 |
| DCC-GARCH | bond_mean_trailing252 | 4 | 0.002500 | -0.39% | 13.40% | 0.199968 |
| DCC-GARCH-EWMA-GARCH | static | 4 | 0.002166 | 0.09% | 11.01% | 0.209350 |
| DCC-GARCH-EWMA-GARCH | conditional | 4 | 0.002187 | 0.08% | 10.91% | 0.209383 |
| DCC-GARCH-EWMA-GARCH | conditional_rank_matched | 4 | 0.002166 | 0.09% | 10.94% | 0.209313 |
| DCC-GARCH-EWMA-GARCH | bond_mean_zero | 3 | 0.002072 | 0.01% | 11.34% | 0.208071 |
| DCC-GARCH-EWMA-GARCH | bond_mean_trailing252 | 2 | 0.001966 | -0.39% | 12.84% | 0.201989 |

[Unrounded table data](research_session_summary_tables/08_whitening_and_bond_mean.csv).

**Whitening result:** conditional whitening changed mean joint-negative probabilities by less than 0.14 percentage points in these comparisons, and marginal-controlled bond hits remained unchanged. The tested whitening construction was therefore not the main explanation for the sign-probability gap. This does not establish that every alternative innovation construction would be equivalent.

**Mean result:** for Hybrid DCC, a zero bond mean reduced bond hits 4→3; the prior-252 mean reduced them 4→2, lowered the bond quantile score from 0.002166 to 0.001966, and raised joint-negative probability 11.01%→12.84%. For all-GARCH DCC, the prior-252 mean reduced bond hits 6→4 and raised joint-negative probability 11.03%→13.40%. These are location sensitivities under otherwise fixed simulated paths, not complete refits of the marginal model under a new mean specification.

The mean change worked on the measured bond quantile criterion because the forecast location became less optimistic in this validation period. It did not close the joint-sign gap, establish an optimal mean estimator, or guarantee better returns. The prior-252 convention was retained as the reference for subsequent controlled experiments.

## 4. Daily filters: investigate scale adaptation and pairwise dependence

A sequential diagnostic used all 586 validation days, updating forecasts only after each observed return. These are adapted one-day filters, not the same object as ex-ante full-month scenario forecasts.

| Model | Asset | Mean z squared | PIT below 5pct rate Static | Lag1 z |
| --- | --- | --- | --- | --- |
| DCC-GARCH | Equity | 0.997 | 5.80% | 0.012 |
| DCC-GARCH | Bond | 1.324 | 6.83% | -0.019 |
| DCC-GARCH | BTC | 0.798 | 4.10% | 0.030 |
| DCC-GARCH-EWMA-GARCH | Equity | 0.997 | 5.80% | 0.012 |
| DCC-GARCH-EWMA-GARCH | Bond | 1.086 | 5.97% | -0.040 |
| DCC-GARCH-EWMA-GARCH | BTC | 0.798 | 4.10% | 0.030 |

[Unrounded table data](research_session_summary_tables/09_daily_filters.csv).

Bond mean squared standardized errors moved from 1.324 with GARCH to 1.086 with EWMA, while Equity was near 1 and BTC around 0.798. This supports investigating bond-specific adaptation; it gives no reason to inflate every asset's volatility uniformly. The z-squared values are descriptive and can also reflect mean errors and tail variation.

| Pair | Mean predicted correlation | Mean z product | Realized within-period z correlation |
| --- | --- | --- | --- |
| Equity & Bond | 0.169 | 0.290 | 0.279 |
| Equity & BTC | 0.285 | 0.299 | 0.336 |
| Bond & BTC | 0.119 | 0.131 | 0.140 |

[Unrounded table data](research_session_summary_tables/10_daily_pair_diagnostics.csv).

The hybrid Equity–Bond predicted correlation averaged 0.169, against a realized within-period standardized-error correlation of 0.279. This suggested examining dependence adaptation. It was not a direct observation of each month's true conditional correlation: standardization errors and sequential updating also affect these diagnostics.

**How this informed the next experiments:** retain EWMA bonds, investigate a faster bond location estimate, then test whether asymmetric responses or a recent correlation estimate improves joint forecasts at controlled marginals.

## 5. Asymmetric DCC: separate a negative-shock response from the bond mean

A factorial comparison tested DCC and scalar ADCC under both the five-year and prior-252 bond means. Rank-matched ADCC variants preserved same-mean DCC marginal values exactly. Original minimum-CVaR holdings were held fixed for the coverage columns.

| Variant | Joint-negative probability | Joint-sign Brier score | Bond VaR hits | Bond quantile score | Fixed minimum-CVaR hits |
| --- | --- | --- | --- | --- | --- |
| DCC / five-year mean | 11.01% | 0.209350 | 4 | 0.002166 | 4 |
| DCC / prior-252 mean | 12.84% | 0.201989 | 2 | 0.001966 | 3 |
| ADCC / five-year mean | 10.51% | 0.210117 | 4 | 0.002172 | 4 |
| ADCC / prior-252 mean | 12.33% | 0.202474 | 2 | 0.001963 | 3 |
| ADCC ranks / five-year mean | 10.49% | 0.210209 | 4 | 0.002166 | 4 |
| ADCC ranks / prior-252 mean | 12.31% | 0.202556 | 2 | 0.001966 | 3 |

[Unrounded table data](research_session_summary_tables/11_bond_mean_and_adcc.csv).

With the prior-252 mean fixed, ADCC reduced the predicted all-negative probability from 12.84% to 12.33% and slightly worsened its Brier score. Fixed minimum-CVaR hits stayed at three. Dependence-only rank matching did not change this conclusion.

| Metric | Mean paired difference | Exploratory 95% lower | Exploratory 95% upper |
| --- | --- | --- | --- |
| Standardized energy score | 0.000884 | -0.000981 | 0.002700 |
| Standardized variogram score p=0.5 | 0.000584 | -0.002743 | 0.003637 |
| Brier score | 0.000486 | -0.002261 | 0.003572 |

[Unrounded table data](research_session_summary_tables/12_adcc_score_uncertainty.csv).

The displayed ADCC-minus-DCC intervals all include zero. All fitted asymmetry coefficients were positive (about 0.011–0.023), but 22/28 fits were near the generalized persistence bound. Positive fitted asymmetry does not mechanically raise joint-loss probabilities: the intercept is adjusted for targeting, other parameters and filtered states are reestimated, and correlation matrices are normalized. Better nested in-sample likelihood therefore need not yield better forecasts.

Native- and common-mean portfolio reoptimizations were also run for minimum CVaR and common targets 3/6. For example, under common means, target-3 gross return was 7.75% for DCC/prior-252 and 8.75% for ADCC/prior-252, with three breaches for both. Those isolated gross-return differences did not come with a consistent fixed-weight risk or joint-score improvement; trading costs were excluded at this stage. See the [full ADCC portfolio table](asymmetric_downside/optimized_summary.csv).

**Decision and next question:** retain DCC as the dependence reference. A particular asymmetric update had not established an advantage; next test a simpler recent-correlation benchmark to see whether the dependence level, rather than this asymmetry mechanism, was the limiting feature.

## 6. Recent correlation at exact reference monthly marginals

At each monthly origin, a correlation matrix was estimated from the strictly prior 252 standardized residual rows and held fixed within the month. The five-year innovation pool, marginal variance models, means and sampled shocks remained shared. The primary rank-matched version retained all reference monthly marginal values. No correlation-window search, portfolio reoptimization or development expansion was performed.

| Model | Joint-negative probability | Joint-sign Brier score | Fixed minimum-CVaR breaches | Fixed minimum-CVaR quantile score |
| --- | --- | --- | --- | --- |
| DCC reference | 12.84% | 0.201989 | 3 | 0.002481 |
| Recent correlation: raw | 13.93% | 0.202173 | 3 | 0.002496 |
| Recent correlation: fixed marginals | 13.97% | 0.202202 | 3 | 0.002498 |

[Unrounded table data](research_session_summary_tables/13_recent_correlation.csv).

The fixed-marginal benchmark raised average Equity–Bond monthly scenario correlation from 0.137 to 0.176 and mean joint-negative probability from 12.84% to 13.97%. Nevertheless, the Brier score and fixed minimum-CVaR quantile score slightly worsened; breaches remained at three. Energy/variogram point scores improved slightly, but uncertainty included zero:

| Metric | Mean paired difference | Exploratory 95% lower | Exploratory 95% upper |
| --- | --- | --- | --- |
| Standardized energy score | -0.000421 | -0.002859 | 0.002248 |
| Standardized variogram score p=0.5 | -0.001942 | -0.009482 | 0.003920 |
| Brier score | 0.000214 | -0.004624 | 0.005086 |

[Unrounded table data](research_session_summary_tables/14_recent_correlation_uncertainty.csv).

A mean probability closer to the observed frequency can still allocate probabilities less accurately across individual months. Also, marginal hits could not improve in the primary comparison because those distributions were held exactly fixed. Equity/Bond 10% and 25% hit counts still suggested marginal-distribution discrepancies, so remaining joint events could not be attributed solely to dependence.

Shared-threshold checks illustrated this: the reference forecast about 0.50 Equity–Bond joint 10% events against four observed, and 2.24 joint 25% events against eight. The recent-correlation control forecast about 0.50 and 2.34, respectively. These are expected counts over 28 months, not forecast probabilities of 50% or 234%. Marginal misspecification also affects such joint-threshold failures.

**Decision and next question:** the tested recent matrix did not justify replacing DCC. The next sequence examined forecast horizon, temporal innovation sampling and copula family separately, then asked whether a different portfolio objective made risk changes economically visible.

## 7. Five-step sequence, step 1: forecasts from one origin at different horizons

Reference forecasts were captured after 1, 5, 10 and all trading days of each holding month. Every horizon started at the same monthly origin, without updating from realized intervening returns. Consequently there are 28 outcomes per horizon, not 586 independent forecasts per horizon. Nominal expected 5%/10%/25% hit counts are 1.4/2.8/7.

| Horizon | Asset | q05 hits | q10 hits | q25 hits | Mean squared standardized error |
| --- | --- | --- | --- | --- | --- |
| 1d | Equity | 0 | 0 | 3 | 0.473 |
| 1d | Bond | 3 | 6 | 10 | 1.871 |
| 1d | BTC | 0 | 3 | 6 | 0.432 |
| 5d | Equity | 0 | 2 | 7 | 0.568 |
| 5d | Bond | 0 | 4 | 13 | 1.265 |
| 5d | BTC | 0 | 1 | 3 | 0.455 |
| 10d | Equity | 1 | 3 | 7 | 0.814 |
| 10d | Bond | 5 | 6 | 11 | 1.790 |
| 10d | BTC | 3 | 4 | 7 | 0.996 |
| month | Equity | 1 | 5 | 11 | 0.888 |
| month | Bond | 2 | 5 | 10 | 1.419 |
| month | BTC | 1 | 4 | 7 | 0.780 |

[Unrounded table data](research_session_summary_tables/15_reference_horizon_calibration.csv).

Bond 5% hits were 3, 0, 5 and 2 by horizon; its 10% hits were 6, 4, 6 and 5. The discrepancies already appeared at the first-day horizon and did not increase monotonically with horizon. Equity and BTC had different patterns. This weakens a blanket explanation based solely on monthly aggregation or uniformly insufficient volatility; it does not uniquely identify the correct marginal model.

![Reference coverage across horizons](calibration_sequence/horizon_coverage.png)

## 8. Five-step sequence, step 2: recency weighting and five-day joint blocks

Recency weighting used a fixed 126-observation half-life, giving an effective innovation-pool size near 363. Five-day circular blocks retained aligned innovation sequences. `Weighted_raw` and `Block5_raw` changed marginals and dependence jointly. Their marginal controls transferred new marginal values onto reference ordinal ranks. `Block5_dependence` transferred block ranks onto exact reference marginal values. At one day, empirical ties limit a literal unique-copula interpretation of ordinal ranks; monthly controls remain exact sorted-value comparisons.

| Model | Asset | Forecast mean | q05 hits | q10 hits | q25 hits | Mean normalized primary quantile score |
| --- | --- | --- | --- | --- | --- | --- |
| Current Hybrid DCC reference | Equity | 1.15% | 1 | 5 | 11 | 0.211684 |
| Current Hybrid DCC reference | Bond | -0.39% | 2 | 5 | 10 | 0.357958 |
| Current Hybrid DCC reference | BTC | 5.62% | 1 | 4 | 7 | 0.163959 |
| Weighted / marginal control | Equity | 0.74% | 1 | 4 | 11 | 0.209687 |
| Weighted / marginal control | Bond | -1.11% | 1 | 1 | 7 | 0.371672 |
| Weighted / marginal control | BTC | 2.55% | 0 | 1 | 6 | 0.153417 |
| Block5 / marginal control | Equity | 1.12% | 1 | 5 | 11 | 0.218215 |
| Block5 / marginal control | Bond | -0.41% | 1 | 5 | 10 | 0.355689 |
| Block5 / marginal control | BTC | 5.86% | 1 | 3 | 7 | 0.162129 |

[Unrounded table data](research_session_summary_tables/16_innovation_marginal_scores.csv).

Weighted marginal Bond hits improved from 2/5/10 to 1/1/7 at 5%/10%/25%, but its average normalized primary quantile loss worsened by **3.83%**, from 0.357958 to 0.371672. Its mean forecast moved to -1.11%, versus -0.38% for the reference and -0.26% realized. Weighting was not recentered or variance-normalized, so a location change and a covariance change accompanied the altered empirical shape. Lower hit counts partly reflect a more pessimistic distribution. That is the key unresolved mechanism behind the final conditional challenger.

Block marginal controls modestly improved Bond/BTC primary scores but worsened Equity's. They did not provide a broad calibration improvement. Volatility clustering already exists in the GARCH/EWMA recursions; five-day shock blocks test additional temporal innovation structure. These results do not establish that such residual dependence is absent, only that this fixed block specification supplied mixed benefits.

The full monthly comparison also separates joint signs from overall distribution and fixed-portfolio risk:

| Model | Predicted_probability | Brier_score | Energy score | Variogram score | VaR_breaches | Mean_quantile_score |
| --- | --- | --- | --- | --- | --- | --- |
| Current Hybrid DCC reference | 12.84% | 0.201989 | 1.517250 | 0.180422 | 3 | 0.002481 |
| Weighted / raw | 21.49% | 0.190013 | 1.599569 | 0.186757 | 1 | 0.002584 |
| Block5 / raw | 12.56% | 0.202334 | 1.516156 | 0.179936 | 2 | 0.002439 |
| Weighted / marginal control | 18.28% | 0.192224 | 1.605647 | 0.194891 | 1 | 0.002561 |
| Block5 / marginal control | 11.95% | 0.203514 | 1.516659 | 0.180100 | 2 | 0.002423 |
| Block5 / dependence control | 13.47% | 0.200911 | 1.516349 | 0.180115 | 3 | 0.002483 |
| Gaussian / fixed marginals | 13.96% | 0.202208 | 1.515360 | 0.178312 | 3 | 0.002506 |
| Student-t / fixed marginals | 13.98% | 0.202448 | 1.515696 | 0.178472 | 3 | 0.002518 |

[Unrounded table data](research_session_summary_tables/17_innovation_and_copula_forecasts.csv).

`VaR_breaches` and `Mean_quantile_score` in that table concern the **same original Hybrid-DCC minimum-CVaR weights**, evaluated using native forecasts. Weighted raw joint-negative probability rose to 21.49%; the marginal-only control still reached 18.28%. This shows that marginal changes contributed substantially to the improvement in the mean sign probability. The weighted marginal control improved the sign Brier point score but worsened the energy point score and the fixed minimum-CVaR quantile score. Its favorable Brier change had an exploratory interval spanning zero. Improving one event forecast was not equivalent to improving the full risk distribution.

![Marginal scores by horizon](calibration_sequence/marginal_scores_by_horizon.png)

## 9. Five-step sequence, step 3: matched Gaussian versus fitted Student-t copulas

Both copulas used a common prior-252-row Kendall-implied latent correlation held fixed within the forecast month, identical empirical daily innovation margins and marginal variance recursions, and common normal draws. Student-t degrees of freedom were estimated from prior data, with a Gaussian-limit alternative. Every horizon was rank-matched to the exact reference marginal scenario values.

All 28 fitted t degrees of freedom were finite/interior: **7.63–86.62, median 13.33**. This was daily t dependence propagated through monthly aggregation, rather than a direct monthly t copula or a deliberately very heavy-tailed fixed degree of freedom.

Gaussian and Student-t joint-negative forecasts were almost identical, 13.96% and 13.98%. They also left substantial shared-quantile discrepancies:

| Model | Assets | Quantile | Predicted_count | Observed_count |
| --- | --- | --- | --- | --- |
| Current Hybrid DCC reference | Equity & Bond | 10% | 0.496 | 4 |
| Current Hybrid DCC reference | Equity & Bond | 25% | 2.235 | 8 |
| Current Hybrid DCC reference | Equity & Bond & BTC | 10% | 0.183 | 1 |
| Current Hybrid DCC reference | Equity & Bond & BTC | 25% | 1.030 | 3 |
| Gaussian / fixed marginals | Equity & Bond | 10% | 0.436 | 4 |
| Gaussian / fixed marginals | Equity & Bond | 25% | 2.240 | 8 |
| Gaussian / fixed marginals | Equity & Bond & BTC | 10% | 0.125 | 1 |
| Gaussian / fixed marginals | Equity & Bond & BTC | 25% | 1.019 | 3 |
| Student-t / fixed marginals | Equity & Bond | 10% | 0.455 | 4 |
| Student-t / fixed marginals | Equity & Bond | 25% | 2.248 | 8 |
| Student-t / fixed marginals | Equity & Bond & BTC | 10% | 0.138 | 1 |
| Student-t / fixed marginals | Equity & Bond & BTC | 25% | 1.037 | 3 |

[Unrounded table data](research_session_summary_tables/18_copula_shared_threshold_events.csv).

The same reference thresholds define every realized event above. Student-t increased some extreme joint-event probabilities slightly, but only one all-three shared-10% event was observed; inference about extremes is sparse. Marginal VaR outcomes were identical by construction. Comparisons against the original DCC also change within-month correlation dynamics, whereas Student-t-versus-Gaussian is the cleaner family comparison.

| Metric | Mean paired difference | Exploratory 95% lower | Exploratory 95% upper |
| --- | --- | --- | --- |
| Energy score | 0.000336 | -0.000177 | 0.000919 |
| Variogram score | 0.000160 | -0.000346 | 0.000669 |
| Brier score | 0.000240 | -0.000177 | 0.000773 |

[Unrounded table data](research_session_summary_tables/19_student_t_versus_gaussian.csv).

Differences are Student-t minus Gaussian: all three displayed point scores worsened slightly and all intervals include zero. The experiment does not justify promoting this fitted t family, and does not rule out every other asymmetric or dynamic tail specification. Exact sorted marginal error was zero for all 12 dependence-control/horizon checks.

## 10. Five-step sequence, step 4: make the optimizer's response visible

Six controlled models were reoptimized under common reference expected returns, full investment, long-only holdings and BTC <=20%. The first strategy minimized CVaR at the original moderate target 3. The second maximized expected return minus estimated trading cost under a shared feasible CVaR budget. That monthly budget was `max(model minimum CVaR) + 0.5 * [max(model CVaR at 40/40/20) - max(model minimum CVaR)]`. It is a forecast-derived relative frontier operating point, not a constant investor-selected loss limit. All models share the same budget within each month; all 28 monthly risk constraints bound for every model.

**Support amendment before portfolio results:** additive common-mean translation would push 13 Block5 BTC draws across eight months below -100%. Instead every model used positive gross-return scaling, `r*=(1+r)*(1+mu_reference)/(1+mu_native)-1`. This preserves valid support, ranks, correlations and standardized shapes, while rescaling centered volatility. The weighted BTC factor ranged about 0.9414–1.1581; block factors differed from one by less than 1%. It is therefore not an unchanged-centered-covariance comparison. Steps 1–3 used native distributions; these portfolios used common-mean scaled distributions. The [full adjustment audit](calibration_sequence/sequence_portfolio_common_mean_translations.csv) records every factor.

Exact self-financing costs were 10 bps per dollar bought or sold, with initial purchase charged and no terminal liquidation. Holdings drifted between rebalances. The risk-budget objective used a linear turnover-cost proxy; realized and net forecast evaluation used exact fees. The moderate strategy's objective remained CVaR, with fees applied in evaluation.

| Model | Portfolio | Net Total return | Net Sharpe (zero risk-free) | Net Maximum drawdown | Gross VaR breaches | Cumulative one-way turnover |
| --- | --- | --- | --- | --- | --- | --- |
| Current Hybrid DCC reference | Common target 3 | 6.98% | 0.285 | -23.77% | 3 | 3.099 |
| Current Hybrid DCC reference | Common risk budget with costs | 12.24% | 0.432 | -23.77% | 3 | 2.577 |
| Weighted / marginal control | Common target 3 | 5.19% | 0.230 | -23.94% | 3 | 3.558 |
| Weighted / marginal control | Common risk budget with costs | 16.35% | 0.550 | -23.03% | 3 | 3.377 |
| Block5 / marginal control | Common target 3 | 6.52% | 0.271 | -23.84% | 3 | 3.034 |
| Block5 / marginal control | Common risk budget with costs | 9.39% | 0.372 | -22.77% | 3 | 2.244 |
| Block5 / dependence control | Common target 3 | 6.15% | 0.260 | -23.92% | 3 | 3.342 |
| Block5 / dependence control | Common risk budget with costs | 10.44% | 0.388 | -23.58% | 3 | 2.629 |
| Gaussian / fixed marginals | Common target 3 | 6.76% | 0.278 | -23.82% | 3 | 3.230 |
| Gaussian / fixed marginals | Common risk budget with costs | 12.63% | 0.430 | -24.38% | 3 | 2.628 |
| Student-t / fixed marginals | Common target 3 | 6.65% | 0.275 | -23.82% | 3 | 3.216 |
| Student-t / fixed marginals | Common risk budget with costs | 12.57% | 0.430 | -24.29% | 3 | 2.603 |

[Unrounded table data](research_session_summary_tables/20_common_mean_costed_portfolios.csv).

For weighted marginals, moderate-target net total return fell **6.98%→5.19%**. Under the common risk budget it rose **12.24%→16.35%**, with Sharpe **0.432→0.550** and cumulative one-way turnover **2.58→3.38**. All six models under both strategies still had **3/28 VaR breaches**. These are meaningful differences in realized allocations and returns, but their direction depends on the decision rule.

| Portfolio | Model B | Mean half L1 allocation distance | Maximum half L1 allocation distance |
| --- | --- | --- | --- |
| Common target 3 | Weighted / marginal control | 3.31% | 8.81% |
| Common target 3 | Block5 / marginal control | 1.83% | 5.54% |
| Common target 3 | Block5 / dependence control | 2.22% | 4.91% |
| Common target 3 | Gaussian / fixed marginals | 2.79% | 7.96% |
| Common target 3 | Student-t / fixed marginals | 2.83% | 6.56% |
| Common risk budget with costs | Weighted / marginal control | 8.88% | 25.29% |
| Common risk budget with costs | Block5 / marginal control | 4.63% | 9.15% |
| Common risk budget with costs | Block5 / dependence control | 2.41% | 4.27% |
| Common risk budget with costs | Gaussian / fixed marginals | 4.50% | 10.92% |
| Common risk budget with costs | Student-t / fixed marginals | 4.62% | 12.85% |

[Unrounded table data](research_session_summary_tables/21_allocation_changes.csv).

The weighted model reallocated 8.88% of portfolio mass on average under the risk budget, with a 25.29% maximum, versus 3.31% on average under the moderate target. The moderate return constraint bound in every month; the risk-budget version exposed more of the difference in forecast risk–return trade-offs. Equal achieved forecast CVaR across risk-budget models follows from the binding shared budget at different holdings; it does not imply the distributions agree.

![Net portfolio wealth and drawdown](calibration_sequence/portfolio_net_wealth_drawdown.png)

## 11. Five-step sequence, step 5: link calibration, holdings and net outcomes

The evidence synthesis joined marginal scores, joint Brier scores, fixed-weight portfolio risk, allocation distances and net outcomes. These remain heterogeneous controlled comparisons, rather than a causal decomposition: calibration uses native distributions, while portfolio optimization uses common means and gross-return scaling.

At fixed minimum-CVaR holdings, weighted marginals reduced hits **3→1** while the quantile score worsened **0.002481→0.002561**. Block marginals reduced hits **3→2** and improved the score slightly to **0.002423**; all dependence-only variants retained three. This supplies a direct answer to the original puzzle: a lower breach count, an improved sign probability, a changed allocation and a better realized return are four different outcomes.

| Model | Reference | Portfolio | Mean paired difference | Exploratory 95% lower | Exploratory 95% upper |
| --- | --- | --- | --- | --- | --- |
| Weighted / marginal control | Current Hybrid DCC reference | Common target 3 | -0.062% | -0.159% | 0.004% |
| Weighted / marginal control | Current Hybrid DCC reference | Common risk budget with costs | 0.130% | -0.084% | 0.357% |
| Block5 / marginal control | Current Hybrid DCC reference | Common target 3 | -0.015% | -0.050% | 0.015% |
| Block5 / marginal control | Current Hybrid DCC reference | Common risk budget with costs | -0.105% | -0.234% | 0.028% |
| Block5 / dependence control | Current Hybrid DCC reference | Common target 3 | -0.028% | -0.085% | 0.017% |
| Block5 / dependence control | Current Hybrid DCC reference | Common risk budget with costs | -0.063% | -0.126% | -0.005% |
| Gaussian / fixed marginals | Current Hybrid DCC reference | Common target 3 | -0.007% | -0.066% | 0.052% |
| Gaussian / fixed marginals | Current Hybrid DCC reference | Common risk budget with costs | 0.021% | -0.096% | 0.135% |
| Student-t / fixed marginals | Current Hybrid DCC reference | Common target 3 | -0.012% | -0.065% | 0.042% |
| Student-t / fixed marginals | Current Hybrid DCC reference | Common risk budget with costs | 0.017% | -0.097% | 0.125% |
| Student-t / fixed marginals | Gaussian / fixed marginals | Common target 3 | -0.004% | -0.027% | 0.013% |
| Student-t / fixed marginals | Gaussian / fixed marginals | Common risk budget with costs | -0.004% | -0.023% | 0.018% |

[Unrounded table data](research_session_summary_tables/22_monthly_net_return_uncertainty.csv).

Differences in this table are monthly net return changes, displayed as percentage points (for example +0.130% means +0.130 percentage points of monthly return). Weighted risk-budget mean advantage was +0.130 percentage points, with an exploratory 95% interval **-0.084 to +0.357**. Thus the higher full-period return did not establish a reliable advantage. Block-dependence risk-budget returns had an adverse interval below zero; some portfolio quantile-score intervals also excluded zero in different directions. These unadjusted comparisons did not form a consistent positive evidence chain for any replacement.

Intervals used three-month circular blocks and 1,999 bootstrap replications. They are exploratory and unadjusted for repeated comparisons. The half-split scenario CVaR diagnostic measures risk-estimate Monte Carlo variability at fixed weights. It does **not** establish allocation stability; independent-scenario reoptimization remains pending.

## 12. What the combined evidence supports, and what remains uncertain

| Hits / 28 | Observed rate | Exact 95% lower | Exact 95% upper | P(at least this many | 5%) |
| --- | --- | --- | --- | --- |
| 2.0 | 7.14% | 0.88% | 23.50% | 0.411691 |
| 3.0 | 10.71% | 2.27% | 28.23% | 0.162665 |
| 4.0 | 14.29% | 4.03% | 32.67% | 0.049074 |
| 7.0 | 25.00% | 10.69% | 44.87% | 0.000364 |

[Unrounded table data](research_session_summary_tables/23_monthly_coverage_uncertainty.csv).

The exact coverage intervals and tail probabilities above assume independent, constant-rate monthly hits; they are descriptive reference calculations, not a full conditional-calibration test. Three observed breaches imply a 10.71% sample rate, but a wide 95% interval of about **2.27%–28.23%**, and at least three occurs with probability **16.27%** under 5% coverage. The current reference is therefore not conclusively shown to have a persistently excessive true breach rate. Seven original breaches were much stronger adverse evidence under that same simplified null. Earlier changes did help; the later variants did not add a consistent gain.

Similarly, the observed joint-negative frequency of 25% comes from only seven months. Its descriptive iid-binomial 95% interval is about 10.69%–44.87%; it is not a precise target that forecasts should be forced to match. Forecast-specific heterogeneous probabilities and serial dependence require their own calibration assessment. At 95% monthly CVaR, the realized sample supplies about 1.4 observations in the tail. Simulating more paths improves forecast Monte Carlo precision, not the number of realized tail outcomes.

The supported explanation is layered: bond location and volatility adaptation mattered; the tested whitening and dependence extensions supplied limited incremental evidence; allocation constraints and objectives determined how risk differences affected holdings; and realized returns reflected a small, regime-dependent validation sample. Monthly diagnostics do not uniquely identify a true innovation distribution or prove a macroeconomic regime mechanism.

### Checks, data and completed-versus-proposed work

The completed five-step run passed 44 focused tests and consistency checks covering all 336 optimized allocations, prior-only fit dates, exact marginal controls and matching daily/monthly self-financing accounting. Reference caches replayed within about 2.23e-7 absolute return. Existing variance caps and daily -99% simulation support safeguards were preserved and counted; rare clipping does not prove irrelevance to extreme-tail estimates. See [verification.json](calibration_sequence/verification.json). No statistical guarantee follows from implementation tests.

Each table links to its unrounded CSV. [summary_manifest.json](research_session_summary_tables/summary_manifest.json) records source/result hashes and which artifacts were read. This summary is generated by `Volatility Modelling/reporting/summarize_research_session.py` using existing result CSVs; it does not rerun experiments. Detailed reports remain in [joint downside](joint_downside/investigation_report.md), [asymmetric dependence](asymmetric_downside/experiment_report.md), [recent correlation](recent_correlation/experiment_report.md), and [the five-step sequence](calibration_sequence/five_step_report.md).

**Not yet tested:** the centered-only and centered-plus-marginal-variance-matched weighted innovation variants; their forecast/portfolio advantages; the proposed additional proper joint VaR–Expected Shortfall score comparison; and independent-scenario allocation stability. None is presented as a successful result. A joint VaR–ES scoring comparison is a suitable next assessment for a CVaR objective, rather than relying on VaR breach counts alone; see [Fissler, Ziegel and Gneiting](https://arxiv.org/abs/1507.00244).

## 13. Final recommendation: one benchmark, one main contender, one conditional challenger

The evidence supports a small final shortlist, rather than another broad family/window search:

| Role | Exact recommended specification | Why it remains | Evidence status |
| --- | --- | --- | --- |
| **One benchmark: Hybrid FHS** | GARCH for Equity/BTC, EWMA for Bond, joint empirical innovation resampling without a DCC correlation update. Align its Bond drift to the same prior-252 convention and its marginal/return assumptions with the main contender for the final clean comparison. | Simpler FHS was competitive at the original common targets; the hybrid preserves the bond-volatility improvement while testing whether DCC adds practical value. | Original Hybrid FHS was tested. Its final matched prior-252 Bond-mean alignment is still pending; existing original-mean results are not claimed to be that aligned head-to-head. |
| **One main contender: current Hybrid DCC reference** | GARCH for Equity/BTC, Bond EWMA lambda 0.94, DCC dependence, full prior five-year empirical innovations, and strictly prior-252-observation Bond mean; latest identifier `Reference`. | Bond mean/volatility changes improved measured calibration, while ADCC, recent correlation, blocks and fitted Student-t did not establish a consistent replacement advantage. | Implemented and tested; retain as the default final specification, with calibration uncertainty acknowledged. |
| **One conditional challenger: moment-controlled weighted Hybrid DCC** | Same main specification and fixed 126-observation weighting half-life. Center the weighted innovation pool to the reference means, then rescale each asset to its reference marginal variance. Use the existing marginal/rank controls for diagnosis and common expected-return controls for portfolio comparison. | Weighting produced the largest allocation response and improved joint-sign point forecasts, but its location/scale shifts confounded interpretation and broader scores were mixed. Moment controls ask whether useful shape information remains after those shifts are removed. | Proposed, not tested. It earns a place only after a predeclared calibration comparison supports it; the existing raw weighted result is not evidence that this corrected version works. |

The raw weighted and centered-only variants are diagnostic controls for that final question, not extra finalists. ADCC, the recent-correlation replacement, five-day blocks and the fitted Student-t variant leave the shortlist because their tested incremental benefits were mixed or too small/uncertain. All-GARCH variants remain historical comparators rather than preferred final specifications because the bond EWMA sensitivity improved relevant diagnostics. This is a development decision, not proof that these families are universally inferior.

The conditional test should keep all other choices fixed and assess predetermined bond marginal quantile scores plus fixed-holding VaR/ES forecasts, with breach counts secondary. It should not restart a search over half-lives, windows or families. If that evidence remains mixed, finish with the aligned **Hybrid FHS benchmark versus the current Hybrid DCC main contender**. If the moment-controlled weighted version earns its place, include it as the **single conditional challenger**. Freeze the final specification and comparison rules before any one-time untouched-test assessment: this is how the experiment sequence leads to one benchmark, one main contender and at most one conditional challenger.
