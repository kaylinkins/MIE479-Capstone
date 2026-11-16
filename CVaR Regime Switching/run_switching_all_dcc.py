"""Revalidate stage-switching CVaR with DCC-GARCH scenarios for every forecast."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from regime_switching_cvar import optimize_switching_cvar
from run_regime_switching import (HERE, FORECASTS, PROJECT, ASSETS, SWITCH_EXPERIMENTS,
    cvar, sha, write_json, load_returns, performance, stage_signal, verify_input_run)
from compute_regime_sharpe import LABEL_PATH, load_common_labels, attach_labels, annualized_sharpe

DISPLAY = {'hmm': 'HMM', 'jump': 'Jump Model',
           'regression_hmm': 'Regime regression A', 'regression_dcc': 'Regime regression B'}


def markdown_table(summary):
    text = '| Forecasts | H | Net return | Ann. Sharpe | Max drawdown | BTC | Equities | Bonds |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for _, r in summary.iterrows():
        text += (f'| {DISPLAY[r.experiment]} | {r.horizon} | {r.net_total_return:.2%} | '
                 f'{r.net_sharpe_zero_rf:.3f} | {r.net_max_drawdown:.2%} | '
                 f'{r.average_BTC:.2%} | {r.average_Equity:.2%} | {r.average_Bond:.2%} |\n')
    return text


def run(*, output=HERE/'regime_switching_all_dcc_results',
        forecast_results=HERE/'forecast_comparison_results',
        previous_switching=HERE/'regime_switching_results', limit_months=None):
    output, forecast_results, previous_switching = map(lambda p: Path(p).resolve(),
                                                      (output, forecast_results, previous_switching))
    producer = verify_input_run(forecast_results)
    if limit_months is not None and (not isinstance(limit_months, int) or limit_months < 1):
        raise ValueError('limit_months must be positive')
    previous = json.loads((previous_switching/'run_manifest.json').read_text())
    if not previous['completed']:
        raise ValueError('Previous switching run is incomplete')
    for relative, checksum in previous['output_sha256'].items():
        if sha(previous_switching/relative) != checksum:
            raise ValueError(f'Previous switching artifact changed: {relative}')
    if previous['protocol']['source_forecast_fingerprint'] != producer['fingerprint']:
        raise ValueError('Previous switching run used a different forecast input ensemble')
    train_path, validation_path = PROJECT/'Data'/'portfolio_train.csv', PROJECT/'Data'/'portfolio_validation.csv'
    train, validation = load_returns(train_path), load_returns(validation_path)
    if train.index[-1] >= validation.index[0]:
        raise ValueError('Training must precede validation')
    source_protocol = producer['protocol']
    evaluation = validation.loc[source_protocol['evaluation_start']:source_protocol['evaluation_end']]
    groups = list(evaluation.groupby(evaluation.index.to_period('M')))
    if limit_months:
        groups = groups[:limit_months]
    signal_paths = {'hmm': FORECASTS/'HMM Forecasts'/'regime_probabilities.csv',
        'jump': FORECASTS/'Jump Model Forecasts'/'regime_probabilities.csv',
        'regression_hmm': FORECASTS/'Regression Forecasts'/'Regime-dependent'/'hmm_regime_conditional_returns.csv'}
    signals = {e: pd.read_csv(p, parse_dates=['rebalance_date', 'information_date']) for e, p in signal_paths.items()}
    signals['regression_dcc'] = signals['regression_hmm']
    labels = load_common_labels(LABEL_PATH)
    protocol = {'policy': 'stage_regime_switching', 'scenario_override': 'DCC GARCH',
        'experiments': {e: {'forecast': name, 'scenarios': 'DCC GARCH', 'original_scenarios': old}
                        for e, (name, old) in SWITCH_EXPERIMENTS.items()},
        'horizons': [1, 3], 'bull_policy': 'matched_return', 'bear_policy': 'min_cvar',
        'selection': 'Forecast month-end regime separately per stage; p_bear<0.5 bull, >0.5 bear; reject ties',
        'alpha': .95, 'risk_aversion': .5, 'return_reward': 0., 'cost_rate': .001,
        'upper': [1., 1., .2], 'risk_free': 0., 'initial_holdings': [0., 0., 0.],
        'n_scenarios': source_protocol['n_scenarios'], 'seed': source_protocol['seed'],
        'evaluation_start': str(groups[0][1].index[0].date()),
        'evaluation_end': str(groups[-1][1].index[-1].date()), 'evaluation_months': len(groups),
        'source_forecast_results': str(forecast_results), 'source_forecast_fingerprint': producer['fingerprint'],
        'previous_switching_results': str(previous_switching), 'static_model_included': False,
        'execution': 'Monthly; execute stage one; exact self-financing fees and daily drift',
        'terminal_liquidation': False, 'scenario_translation': False, 'regime_volatility_overlay': False,
        'conditional_sharpe_labels': 'Common HMM forecast month1-end labels for every portfolio'}
    input_files = [forecast_results/'run_manifest.json', previous_switching/'run_manifest.json',
        train_path, validation_path, LABEL_PATH, *signal_paths.values()]
    source_files = [Path(__file__), HERE/'regime_switching_cvar.py', HERE/'regime_cvar.py',
        HERE/'run_regime_switching.py', HERE/'run_regime_cvar.py', HERE/'compute_regime_sharpe.py', Path(cvar.__file__)]
    identity = {'protocol': protocol, 'input_sha256': {str(p): sha(p) for p in input_files},
        'source_sha256': {str(p): sha(p) for p in source_files},
        'environment': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy')}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output/'run_manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior['fingerprint'] != fingerprint:
            raise ValueError('Different code/inputs/settings; select a new --output directory')
        if prior['completed']:
            for relative, checksum in prior['output_sha256'].items():
                if sha(output/relative) != checksum:
                    raise ValueError(f'Completed artifact changed: {relative}')
            summary = pd.read_csv(output/'portfolio_summary.csv')
            print('Reusing verified all-DCC switching run.\n'+markdown_table(summary), flush=True)
            return summary
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', protocol)
    write_json(manifest_path, {**identity, 'fingerprint': fingerprint, 'completed': False})
    log = output/'run.log'
    log.write_text('', encoding='utf-8')

    def emit(message):
        print(message, flush=True)
        with log.open('a', encoding='utf-8') as stream:
            stream.write(message+'\n')

    states = {(e, h): {'prior': np.zeros(3), 'wealth': 1.} for e in SWITCH_EXPERIMENTS for h in (1, 3)}
    monthly, daily, planned, selections = ([] for _ in range(4))
    elapsed, started = evaluation.iloc[:0], time.monotonic()
    for i, (month, block) in enumerate(groups):
        origin, key = block.index[0], str(block.index[0].date())
        history = pd.concat([train, elapsed])
        history_end = history.index[-1]
        if history_end >= origin:
            raise AssertionError('Current/future returns reached the decision history')
        path = forecast_results/'forecasts'/f'{key}.npz'
        if sha(path) != producer['output_sha256'][f'forecasts/{key}.npz']:
            raise ValueError('Source scenario cache changed')
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            if pd.Timestamp(metadata['information_date']) != history_end:
                raise ValueError('Source forecast cutoff disagrees with available history')
            cube, days = saved['DCC GARCH'], saved['holding_days']
            if len(cube) != protocol['n_scenarios'] or len(block) > days[0]:
                raise ValueError('Scenario count or holding-period length disagrees with the source')
            pool, plans, regimes_by_model, p_by_model = cvar.CVaRCutPool(), {}, {}, {}
            for e, (forecast, _) in SWITCH_EXPERIMENTS.items():
                p, regimes, signal_source, dates = stage_signal(signals[e], e, origin, history_end)
                regimes_by_model[e], p_by_model[e] = regimes, p
                means = saved['means_'+e]
                if e.startswith('regression_'):
                    rows = signals[e][(signals[e].rebalance_date == origin)
                        & (signals[e].origin_type == 'validation')].sort_values('month_ahead')
                    expected_means = [[row[f'{a}_conditional_on_{regimes[k]}_end'] for a in ASSETS]
                                      for k, (_, row) in enumerate(rows.iterrows())]
                    np.testing.assert_array_equal(means, expected_means)
                for k in range(3):
                    selections.append({'origin': key, 'experiment': e, 'forecast': forecast, 'scenarios': 'DCC GARCH',
                        'stage': k+1, 'forecast_month': str(month+k), 'holding_days': int(days[k]),
                        'information_date': str(dates[k].date()), 'signal_source': signal_source,
                        'p_bear_month_end': p[k], 'selected_regime': regimes[k],
                        'stage_policy': 'matched_return' if regimes[k] == 'bull' else 'min_cvar',
                        **dict(zip(ASSETS, means[k]))})
                for h in (1, 3):
                    plans[e, h] = optimize_switching_cvar(cube[:, :h], states[e, h]['prior'], days[:h],
                        means[:h], regimes[:h], mean_source=forecast, scenario_source='DCC GARCH', cut_pool=pool)
            write_json(output/'decisions'/f'{key}.json', {'origin': key, 'information_date': str(history_end.date()),
                'source_forecast_path': str(path), 'source_forecast_sha256': sha(path), 'scenario_key': 'DCC GARCH',
                'current_month_outcomes_supplied_to_decisions': 0,
                'plans': {f'{e} | H={h}': asdict(plan) for (e, h), plan in plans.items()}})
            for (e, h), plan in plans.items():
                state = states[e, h]
                executed = cvar.execute_first_period(plan, block.to_numpy(), initial_wealth=state['wealth'])
                first, targets = plan.first_weights, plan.metadata['gross_return_targets']
                net_distribution = (1-executed.fee_fraction)*(1+cube[:, 0] @ first)-1
                net_var, net_es = cvar.loss_var_cvar(net_distribution)
                common = {'experiment': e, 'forecast': SWITCH_EXPERIMENTS[e][0], 'scenarios': 'DCC GARCH',
                    'policy': 'stage_regime_switching', 'horizon': h, 'origin': key}
                monthly.append({**common, **dict(zip(ASSETS, first)), 'complete_month': len(block) == days[0],
                    'days': len(block), 'executed_regime': regimes_by_model[e][0],
                    'executed_policy': plan.metadata['stage_policies'][0], 'p_bear_month1_end': p_by_model[e][0],
                    'gross_return': executed.gross_period_return, 'net_return': executed.net_period_return,
                    'ending_wealth': executed.ending_wealth, 'fee_fraction': executed.fee_fraction,
                    'one_way_turnover': .5*np.abs(first-state['prior']).sum() if state['prior'].any() else 0.,
                    'expected_return': plan.expected_returns[0], 'return_floor': targets[0],
                    'gross_forecast_cvar': plan.forecast_cvar[0], 'net_forecast_var': net_var, 'net_forecast_cvar': net_es})
                for date, gross, net in zip(block.index, executed.gross_daily_returns, executed.net_daily_returns):
                    daily.append({**common, 'date': str(date.date()), 'gross_return': gross, 'net_return': net})
                for k in range(h):
                    planned.append({**common, 'stage': k+1, 'forecast_month': str(month+k),
                        'selected_regime': regimes_by_model[e][k], 'stage_policy': plan.metadata['stage_policies'][k],
                        'p_bear_month_end': p_by_model[e][k], **dict(zip(ASSETS, plan.weights[k])),
                        'expected_return': plan.expected_returns[k], 'return_floor': targets[k],
                        'forecast_cvar': plan.forecast_cvar[k], 'estimated_cost': plan.estimated_costs[k]})
                states[e, h] = {'prior': executed.ending_weights, 'wealth': executed.ending_wealth}
        elapsed = pd.concat([elapsed, block])
        emit(f'{i+1:02d}/{len(groups)} {key} | 8 stage-switching portfolios | DCC-GARCH scenarios throughout')
    monthly, daily, planned, selections = map(pd.DataFrame, (monthly, daily, planned, selections))
    labeled = attach_labels(daily, labels)
    previous_monthly = pd.read_csv(previous_switching/'portfolio_monthly.csv')
    previous_daily = pd.read_csv(previous_switching/'portfolio_daily.csv')
    previous_selections = pd.read_csv(previous_switching/'stage_selections.csv')
    summary, comparisons, frequencies, conditional = [], [], [], []
    for e, h in states:
        m = monthly[(monthly.experiment == e) & (monthly.horizon == h)]
        d = daily[(daily.experiment == e) & (daily.horizon == h)]
        p = planned[(planned.experiment == e) & (planned.horizon == h)]
        np.testing.assert_allclose(np.prod(1+d.net_return), m.ending_wealth.iloc[-1], atol=1e-10)
        common = {'experiment': e, 'forecast': SWITCH_EXPERIMENTS[e][0], 'scenarios': 'DCC GARCH', 'horizon': h}
        metrics = performance(d, m)
        summary.append({**common, 'policy': 'stage_regime_switching', **metrics})
        cond = dict(common)
        observations = labeled[(labeled.experiment == e) & (labeled.horizon == h)]
        for regime in ('bull', 'bear'):
            group = observations[observations.common_regime == regime]
            cond.update({regime+'_sharpe': annualized_sharpe(group.net_return.to_numpy()),
                         regime+'_days': len(group), regime+'_months': group.origin.nunique()})
        conditional.append(cond)
        frequencies.append({**common, 'executed_bull_months': int((m.executed_regime == 'bull').sum()),
            'executed_bear_months': int((m.executed_regime == 'bear').sum()),
            'planned_bull_stages': int((p.selected_regime == 'bull').sum()),
            'planned_bear_stages': int((p.selected_regime == 'bear').sum())})
        old_m = previous_monthly[(previous_monthly.experiment == e) & (previous_monthly.horizon == h)
            & (previous_monthly.origin.isin(m.origin))]
        old_d = previous_daily[(previous_daily.experiment == e) & (previous_daily.horizon == h)
            & (previous_daily.origin.isin(m.origin))]
        old_metrics = performance(old_d, old_m)
        comparisons.append({**common, **{f'all_dcc_minus_previous_{metric}': metrics[metric]-old_metrics[metric]
            for metric in ('net_total_return', 'net_sharpe_zero_rf', 'net_max_drawdown', 'net_monthly_cvar95')}})
        if e == 'regression_dcc':
            np.testing.assert_allclose(m[list(ASSETS)], old_m[list(ASSETS)], atol=2e-5, rtol=0)
            np.testing.assert_allclose(m.net_return, old_m.net_return, atol=2e-6, rtol=0)
    for e in SWITCH_EXPERIMENTS:
        new = selections[selections.experiment == e].sort_values(['origin', 'stage'])
        old = previous_selections[(previous_selections.experiment == e)
            & (previous_selections.origin.isin(new.origin))].sort_values(['origin', 'stage'])
        assert new.selected_regime.tolist() == old.selected_regime.tolist()
        assert new.stage_policy.tolist() == old.stage_policy.tolist()
        np.testing.assert_allclose(new[list(ASSETS)], old[list(ASSETS)], atol=1e-14)
    for h in (1, 3):
        a = monthly[(monthly.experiment == 'regression_hmm') & (monthly.horizon == h)]
        b = monthly[(monthly.experiment == 'regression_dcc') & (monthly.horizon == h)]
        np.testing.assert_allclose(a[list(ASSETS)], b[list(ASSETS)], atol=2e-5, rtol=0)
        np.testing.assert_allclose(a.net_return, b.net_return, atol=2e-6, rtol=0)
    summary, conditional = pd.DataFrame(summary), pd.DataFrame(conditional)
    for name, frame in {'portfolio_summary': summary, 'portfolio_monthly': monthly, 'portfolio_daily': daily,
        'planned_allocations': planned, 'stage_selections': selections, 'policy_frequency': pd.DataFrame(frequencies),
        'scenario_change_comparison': pd.DataFrame(comparisons), 'conditional_sharpe': conditional}.items():
        frame.to_csv(output/f'{name}.csv', index=False)
    table = markdown_table(summary)
    (output/'summary_table.md').write_text(table, encoding='utf-8')
    report = '# Stage-switching revalidation: DCC-GARCH scenarios for every model\n\n'
    report += f"{protocol['evaluation_start']} to {protocol['evaluation_end']}; monthly rebalancing.\n\n"
    report += table+'\n'
    report += ('Every row uses the exact same verified DCC-GARCH scenario cube at each origin. '
        'Only the scenario source changes from the previous switching run. Mean forecasts and '
        'forecast regime signals remain unchanged: native HMM, native Jump Model, and hard-selected '
        'conditional regression means with their original stage-specific month-end probabilities. '
        'Regime regression A/B retain the old HMM/DCC experiment slots; they now have identical '
        'inputs and portfolios. Static regression is excluded.\n\n'
        'Bull stages enforce the max-Sharpe return floor and bear stages use minimum CVaR with '
        'no active floor. Every stage minimizes 0.5*CVaR95 plus 10-bps trading costs. The full '
        'horizon is solved jointly, but only stage one is executed; the schedule is reforecast '
        'at the next rebalance. Risk is summed stage CVaR, not nested/terminal-wealth CVaR. '
        'Portfolios start in cash, are fully invested and long-only, and cap BTC at 20%. '
        'Exact self-financing fees and daily drift apply; no terminal liquidation occurs. '
        'Scenario paths are not shifted, filtered, resampled or volatility-scaled. The matched '
        'reference covariance now comes from the DCC-GARCH scenarios for every variant.\n\n'
        'Net return is cumulative, Sharpe is annualized from net daily returns with sample '
        'standard deviation and zero risk-free return, and maximum drawdown uses daily net '
        'wealth. Allocation columns average executed monthly target weights. conditional_sharpe.csv '
        'uses the same common HMM forecast month1-end labels for every model: the grouping does '
        'not alter any model\'s trading signal.\n\n'
        'scenario_change_comparison.csv compares with the previous switching run on identical '
        'dates. The previously DCC-based regression portfolio is checked for unchanged results. '
        'stage_selections.csv exposes all signals and mean vectors; decisions/*.json saves '
        'plans before execution and identifies the verified source archive. No models are '
        'refitted, and no test returns are loaded. The original forecast-training provenance '
        'limitations remain. A short validation sample does not establish statistical superiority.\n')
    (output/'report.md').write_text(report, encoding='utf-8')
    emit('\n'+table)
    emit('\nCommon HMM conditional annualized Sharpe:\n'+conditional.to_string(index=False))
    emit(f'Elapsed: {time.monotonic()-started:.1f}s. Results: {output}')
    write_json(manifest_path, {**identity, 'fingerprint': fingerprint, 'completed': True,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'output_sha256': {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
                          if p.is_file() and p != manifest_path}})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'regime_switching_all_dcc_results')
    parser.add_argument('--forecast-results', type=Path, default=HERE/'forecast_comparison_results')
    parser.add_argument('--previous-switching', type=Path, default=HERE/'regime_switching_results')
    parser.add_argument('--limit-months', type=int)
    run(**vars(parser.parse_args()))
