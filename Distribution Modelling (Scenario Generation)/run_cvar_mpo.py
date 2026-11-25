"""Run the four original scenario generators with matched-return CVaR MPO.

This runner is self-contained: it needs only the two sibling Python modules,
the scientific dependencies and the supplied training/validation return CSVs.
No old experiment manifests, archive paths or scenario caches are needed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path"""Run the four original scenario generators with matched-return CVaR MPO.

The original validation run uses the two sibling modules and the supplied
training and validation return CSVs. Use --run-testing for the held-out CVaR
strategies, heuristic and sensitivity sweeps.
Use --plot-loss-distribution to display a loss histogram from saved regime-aware
DCC-GARCH results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import cvar_mpo as cvar

# The scenario library lives in the Regime switching subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'Regime switching'))
import scenario_generation as scenarios

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'Data'
REGIME_RESULTS = HERE.parent / 'CVaR Regime Models' / 'all_dcc_comparison_results_rf4'
REGIME_FORECASTS = REGIME_RESULTS.parent / 'forecast_comparison_results_rf4'
TEST_RESULTS = HERE.parent / 'Testing Results'
TEST_FORECASTS = TEST_RESULTS / '_forecast_inputs'
SENSITIVITY_RESULTS = HERE.parent / 'sensitivity analysis'
TEST_MODELS = {
    'Jump Regime-Aware CVaR': ('jump', False),
    'Jump Regime-Switching CVaR': ('jump', True),
    'HMM Regime-Switching CVaR': ('hmm', True),
}
TEST_FORECAST_NAMES = {'hmm': 'HMM', 'jump': 'Jump Model'}
TEST_SWEEPS = {
    'CVaR Confidence Level': ('alpha', (.90, .95, .99), '.0%'),
    'Risk Aversion Parameter': ('risk_aversion', (.10, .25, .50, .75, 1.), 'g'),
    'Bitcoin Allocation Cap': ('btc_cap', (.15, .20, .25, .30), '.0%'),
    'Transaction Costs': ('cost_rate', (.0005, .001, .0015, .002, .0025), 'bps'),
}
ASSETS = scenarios.ASSETS
MODELS = scenarios.MODELS
# Original NYSE schedules published in 2021, before every supplied evaluation
# origin. Early-close sessions remain. Do not build future calendars from returns.
HOLIDAYS = {
    2022: ['01-17', '02-21', '04-15', '05-30', '06-20', '07-04', '09-05', '11-24', '12-26'],
    2023: ['01-02', '01-16', '02-20', '04-07', '05-29', '06-19', '07-04', '09-04', '11-23', '12-25'],
    2024: ['01-01', '01-15', '02-19', '03-29', '05-27', '06-19', '07-04', '09-02', '11-28', '12-25']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False)+'\n', encoding='utf-8')


def known_calendar(path=None):
    """Use a supplied ex-ante calendar or the frozen 2022-2024 NYSE schedule."""
    if path is None:
        closed = pd.DatetimeIndex([f'{year}-{day}' for year, days in HOLIDAYS.items() for day in days])
        return pd.bdate_range('2022-01-01', '2024-12-31').difference(closed)
    frame = pd.read_csv(path)
    if 'Date' not in frame.columns:
        raise ValueError('Calendar CSV must contain a Date column with complete future trading sessions')
    dates = pd.DatetimeIndex(pd.to_datetime(frame['Date'], errors='raise'))
    if (dates.empty or dates.hasnans or dates.tz is not None or dates.has_duplicates
            or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())):
        raise ValueError('Calendar dates must be ordered, unique, timezone-naive daily dates')
    return dates


def load_returns(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if set(frame.columns) != set(ASSETS):
        raise ValueError(f'{Path(path).name} must contain exactly Equity, Bond and BTC daily return columns')
    frame = frame.loc[:, list(ASSETS)].astype(float)
    if (frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
            or frame.index.tz is not None or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or not frame.index.equals(frame.index.normalize()) or not np.isfinite(frame.to_numpy()).all()
            or (frame <= -1).any().any()):
        raise ValueError(f'Invalid chronology or simple returns in {Path(path).name}')
    return frame


def testing_inputs():
    """Load the fixed split and an ex-ante calendar for all three planned months."""
    import exchange_calendars as xcals

    train, validation, test = (load_returns(DATA / f'portfolio_{split}.csv')
                               for split in ('train', 'validation', 'test'))
    history = pd.concat([train, validation, test])
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError('Training, validation and test dates must be disjoint and ordered')
    end = (test.index[-1].to_period('M') + 2).end_time.date()
    calendar = xcals.get_calendar('XNYS', start=str(train.index[0].date()), end=str(end))
    if not test.index.equals(calendar.sessions_in_range(test.index[0], test.index[-1])):
        raise ValueError('Test returns must contain every NYSE trading day in their date range')
    return train, validation, test, history, calendar


def prepare_testing_forecasts(source):
    """Prepare one set of forecasts for the held-out test split."""
    from functools import partial

    train, _, test, history, calendar = testing_inputs()
    folder = TEST_FORECASTS / source
    folder.mkdir(parents=True, exist_ok=True)
    forecast_rows = []
    settings = {}
    hmm_dir = HERE.parent / '2-State HMM' / 'HMM Forecasts'
    jump_dir = hmm_dir.parent / 'Jump Model Forecasts'
    code = []
    if source == 'dcc':
        code.append(Path(scenarios.__file__))
    else:
        sys.path[:0] = [str(hmm_dir), str(jump_dir)]
        import hmm_regime_model as hmm

        code.append(Path(hmm.__file__))
        if source == 'jump':
            import jump_regime_model_tuned as jump

            code.extend((Path(jump.__file__), Path(jump.j.__file__)))
        else:
            def fit(window, seed):
                model = hmm.fit_hmm(window.to_numpy(), seed=seed)
                return model, hmm.filtered_state_probabilities(model, window.to_numpy())[-1]

    # Reuse complete inputs only when their underlying data and estimators still agree.
    identity = {
        'input_sha256': {split: sha(DATA / f'portfolio_{split}.csv') for split in ('train', 'validation', 'test')},
        'estimator_sha256': {path.name: sha(path) for path in code},
        'first_test_date': str(test.index[0].date()), 'last_test_date': str(test.index[-1].date()),
        'horizon': 3, 'scenario_count': 10000 if source == 'dcc' else None, 'seed': 42,
    }
    manifest_path = folder / 'input_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['identity'] == identity and all((folder / name).is_file() and sha(folder / name) == checksum
                for name, checksum in manifest['output_sha256'].items()):
            print(f'Reusing verified test forecasts for {source}', flush=True)
            return
    if source == 'jump':
        penalty, pooling = jump.select_settings(train, calendar)
        fit = partial(jump.fit_jump, penalty=penalty, pooling=pooling)
        settings = {'penalty': float(penalty), 'mean_pooling': float(pooling), 'tuned_on': 'training only'}

    groups = list(test.groupby(test.index.to_period('M')))
    outputs = []
    for i, (month, block) in enumerate(groups):
        origin = block.index[0]
        prior = history.loc[history.index < origin]
        sessions = [calendar.sessions_in_range((month+k).start_time, (month+k).end_time.normalize())
                    for k in range(3)]
        if not block.index.equals(sessions[0][:len(block)]) or (i < len(groups)-1 and len(block) != len(sessions[0])):
            raise ValueError(f'Invalid test holding-period calendar at {origin.date()}')
        days = [len(s) for s in sessions]
        if source == 'dcc':
            forecast = scenarios.generate_scenarios(prior, origin, sessions, i, n_scenarios=10000, seed=42)
            if forecast.metadata['history_end'] != str(prior.index[-1].date()):
                raise ValueError(f'DCC-GARCH information cutoff differs at {origin.date()}')
            name = f'{origin.date()}.npz'
            pending = (folder / name).with_suffix('.pending.npz')
            np.savez_compressed(pending, returns=forecast.returns['DCC GARCH'].astype(np.float32),
                                days=days, information_date=str(prior.index[-1].date()))
            pending.replace(folder / name)
            outputs.append(name)
        else:
            model, p = fit(prior, seed=i)
            _, means, _ = hmm.forecast_moments_exact(model, p, days)
            for k in range(3):
                p_end = (p @ np.linalg.matrix_power(model.transmat_, sum(days[:k+1])))[1]
                forecast_rows.append({'origin': str(origin.date()), 'stage': k+1,
                    'information_date': str(prior.index[-1].date()), 'forecast_month': str(month+k),
                    'holding_days': days[k], 'p_bear_today': float(p[1]),
                    'p_bear_month_end': float(p_end), **dict(zip(ASSETS, means[k]))})
        print(f'{source} test forecast {i+1}/{len(groups)} {origin.date()}', flush=True)
    if source != 'dcc':
        name = 'forecast_inputs.csv'
        pending = folder / 'forecast_inputs.pending.csv'
        pd.DataFrame(forecast_rows).to_csv(pending, index=False)
        pending.replace(folder / name)
        outputs.append(name)
    pending = manifest_path.with_suffix('.pending.json')
    write_json(pending, {'identity': identity, 'settings': settings,
                         'output_sha256': {name: sha(folder / name) for name in outputs}})
    pending.replace(manifest_path)


@dataclass(frozen=True)
class TestSettings:
    alpha: float = .95
    risk_aversion: float = .5
    btc_cap: float = .20
    cost_rate: float = .001


def load_testing_backtest_inputs():
    """Verify the shared test forecasts and load the three return splits."""
    for source in ('hmm', 'jump', 'dcc'):
        prepare_testing_forecasts(source)
    train, validation, test, history, _ = testing_inputs()
    forecasts = {key: pd.read_csv(TEST_FORECASTS / key / 'forecast_inputs.csv',
                                 parse_dates=['origin', 'information_date'])
                 for key in TEST_FORECAST_NAMES}
    return train, validation, test, history, forecasts


def load_testing_rebalance(origin, opening_date, forecasts):
    """Read a common DCC ensemble and forecasts available before the rebalance."""
    with np.load(TEST_FORECASTS / 'dcc' / f'{origin.date()}.npz', allow_pickle=False) as saved:
        cube, days = saved['returns'].astype(float), saved['days']
        if cube.shape != (10000, 3, 3) or pd.Timestamp(str(saved['information_date'])) != opening_date:
            raise ValueError(f'Unexpected DCC-GARCH inputs at {origin.date()}')
    signals, means = {}, {}
    for source, frame in forecasts.items():
        rows = frame.loc[frame.origin.eq(origin)].sort_values('stage')
        if (rows.stage.tolist() != [1, 2, 3] or not rows.information_date.eq(opening_date).all()
                or not np.array_equal(rows.holding_days, days)):
            raise ValueError(f'Invalid {source} forecast inputs at {origin.date()}')
        signals[source] = rows.p_bear_month_end.to_numpy(float)
        means[source] = rows[list(ASSETS)].to_numpy(float)
    return cube, days, signals, means


def walk_forward_test(test, history, forecasts, settings):
    """Evaluate each optimizer and setting on a separate account starting in cash."""
    sys.path.insert(0, str(REGIME_RESULTS.parent))
    from regime_cvar import optimize_forecast_cvar
    from regime_switching_cvar import optimize_switching_cvar, regimes_from_bear_probabilities

    configs = [(model, setting) for model in TEST_MODELS for setting in settings]
    accounts = {key: {'prior': np.zeros(3), 'wealth': 1., 'monthly': [], 'daily': []} for key in configs}
    groups = list(test.groupby(test.index.to_period('M')))
    for i, (_, block) in enumerate(groups):
        origin = block.index[0]
        opening_date = history.index[history.index < origin][-1]
        cube, days, signals, means = load_testing_rebalance(origin, opening_date, forecasts)
        regimes = {source: regimes_from_bear_probabilities(p) for source, p in signals.items()}
        pools = {rate: cvar.CVaRCutPool() for rate in {cfg.alpha for cfg in settings}}
        common = {'origin': origin, 'common_regime': regimes['hmm'][0]}
        for model, cfg in configs:
            source, switching = TEST_MODELS[model]
            state = accounts[model, cfg]
            prior = state['prior']
            kwargs = dict(alpha=cfg.alpha, risk_aversion=cfg.risk_aversion, cost_rate=cfg.cost_rate,
                          upper=(1., 1., cfg.btc_cap), mean_source=TEST_FORECAST_NAMES[source],
                          scenario_source='DCC GARCH', cut_pool=pools[cfg.alpha])
            if switching:
                plan = optimize_switching_cvar(cube, prior, days, means[source], regimes[source], **kwargs)
            else:
                plan = optimize_forecast_cvar(cube, prior, days, means[source], policy='matched_return', **kwargs)
            w = plan.first_weights
            execution = cvar.execute_first_period(plan, block.to_numpy(), initial_wealth=state['wealth'])
            state['monthly'].append({**common, **dict(zip(ASSETS, w)), 'days': len(block),
                'complete_month': len(block) == days[0], 'net_return': execution.net_period_return,
                'fee_fraction': execution.fee_fraction, 'bear_signal': signals[source][0], 'threshold': .5,
                'one_way_turnover': .5*np.abs(w-prior).sum() if prior.any() else 0.,
                'ending_wealth': execution.ending_wealth,
                'return_floor': plan.metadata['gross_return_targets'][0]})
            state['daily'].extend({**common, 'date': date, 'net_return': ret}
                                  for date, ret in zip(block.index, execution.net_daily_returns))
            state.update(prior=execution.ending_weights, wealth=execution.ending_wealth)
        print(f'Test {i+1:02d}/{len(groups)} {origin.date()} | {len(configs)} CVaR accounts', flush=True)
    return {key: (pd.DataFrame(state['daily']), pd.DataFrame(state['monthly']))
            for key, state in accounts.items()}


def test_heuristic(train, test, forecasts):
    """Apply the original three-asset rule with both pre-rebalance signals."""
    from simple_regime_strategy import comparison_rotation_paths, simulate
    from regime_switching_cvar import regimes_from_bear_probabilities

    first = {name: data.loc[data.stage.eq(1)].sort_values('origin') for name, data in forecasts.items()}
    signals = pd.concat([table.rename(columns={'origin': 'rebalance_date'}).assign(source=TEST_FORECAST_NAMES[key])
                         for key, table in first.items()], ignore_index=True)
    labels = pd.Series(regimes_from_bear_probabilities(first['hmm'].p_bear_month_end),
                       index=first['hmm'].origin)
    expected_days = first['hmm'].set_index('origin').holding_days
    result = {}
    for label, (net, weights, probability, threshold) in comparison_rotation_paths(train, test, signals).items():
        targets = {origin: row.to_numpy(float) for origin, row in weights.iterrows()}
        daily_net, trades = simulate(test, weights.index, targets)
        np.testing.assert_array_equal(net, daily_net)
        trades = trades.set_index('date')
        wealth = (1 + net).cumprod()
        opening = wealth.shift(fill_value=1.)
        monthly, daily = [], []
        for _, block in net.groupby(net.index.to_period('M')):
            origin = block.index[0]
            common = {'origin': origin, 'common_regime': labels[origin]}
            monthly.append({**common, **weights.loc[origin].to_dict(), 'days': len(block),
                'complete_month': len(block) == expected_days[origin],
                'net_return': float((1+block).prod()-1), 'ending_wealth': wealth[block.index[-1]],
                'fee_fraction': trades.at[origin, 'cost']/opening[origin],
                'one_way_turnover': trades.at[origin, 'traded_pct']*.005 if monthly else 0.,
                'bear_signal': probability[origin], 'threshold': threshold})
            daily.extend({**common, 'date': date, 'net_return': ret} for date, ret in block.items())
        result[label] = pd.DataFrame(daily), pd.DataFrame(monthly)
    return result


def write_test_report(output, runs, caps, *, forecast, initial_date, parameter=None, heuristic=False,
                      model_context=None):
    """Write the two validation-format tables and three figures for one result folder."""
    import importlib.util
    from regime_comparison_figures import save_testing_figures

    source = HERE.parent / 'Regime Aware CVaR Results v0' / 'build_tables.py'
    spec = importlib.util.spec_from_file_location('validation_table_metrics', source)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    rows = []
    for label, (daily, monthly) in runs.items():
        metrics.BTC_CAP = caps[label]
        episodes = metrics.regime_episodes(daily)
        np.testing.assert_allclose((1+daily.net_return).groupby(daily.origin).prod()-1,
                                   monthly.net_return, atol=1e-12, rtol=0)
        np.testing.assert_allclose((1+monthly.net_return).cumprod(), monthly.ending_wealth, atol=1e-12, rtol=0)
        for regime_key, regime in metrics.REGIMES:
            ds = daily if regime_key == 'all' else daily.loc[daily.common_regime.eq(regime_key)]
            ms = monthly if regime_key == 'all' else monthly.loc[monthly.common_regime.eq(regime_key)]
            dd = None if regime_key == 'all' else episodes.loc[episodes.common_regime.eq(regime_key), 'max_drawdown'].min()
            values = {} if ds.empty else metrics.group_statistics(ds, ms, episode_drawdown=dd)
            rows.append({**values, 'forecast': label if parameter or heuristic else forecast,
                         'scenarios': 'Rule-based' if heuristic else 'DCC-GARCH', 'regime': regime})
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    first_daily, first_monthly = next(iter(runs.values()))
    start, end = (first_daily.date.iloc[i].date() for i in (0, -1))
    bull = first_monthly.common_regime.eq('bull')
    complete = first_monthly.complete_month
    context = (f'{start} to {end}, {len(first_daily)} trading sessions and {len(first_monthly)} monthly rebalances. '
               'Bull and bear reporting uses the pre-rebalance HMM month-one-end forecast. '
               f'That forecast classified {int(bull.sum())} rebalances as bull and {int((~bull).sum())} as bear. '
               'Reported Sharpe uses a 4% annual risk-free rate. '
               f'Monthly VaR uses the {int(complete.sum())} complete holding months, of which '
               f'{int((complete & ~bull).sum())} were classified as bear.\n\n')
    if heuristic:
        context += ('The 3-asset heuristic uses 80% equity and 20% Bitcoin in bull states and 100% bonds '
                    'in bear states. The HMM and tuned Jump signals are evaluated separately.\n\n')
    elif model_context:
        context += f'{model_context}\n\n'
    else:
        context += ('Each plan uses three months, 10,000 DCC-GARCH scenarios per rebalance, and a '
                    '4% annual risk-free rate for the max-Sharpe reference. The validation settings are '
                    '95% CVaR confidence, 0.5 risk aversion, 20% Bitcoin cap, and 10 bps transaction costs.\n\n')
    if parameter:
        context += f'This sweep varies {parameter.lower()} for the {forecast} strategy. Other settings use their validation values.\n\n'
    for filename, title, columns in (
        ('table_1_performance', 'Table 1. Portfolio risk and return performance', metrics.PERFORMANCE),
        ('table_2_allocations', 'Table 2. Portfolio allocation behaviour', metrics.ALLOCATIONS),
    ):
        frame = metrics.presentation_table(table, columns)
        if parameter:
            frame = frame.rename(columns={'Forecasts': parameter})
        frame.to_csv(output / f'{filename}.csv', index=False)
        (output / f'{filename}.md').write_text(f'# {title}\n\n{context}{metrics.markdown_table(frame)}', encoding='utf-8')
    save_testing_figures(runs, initial_date=initial_date, output=output,
                         forecast=forecast, heuristic=heuristic, parameter=parameter)
    print(f'Saved tables and figures in {output.relative_to(HERE.parent)}', flush=True)


def run_testing(mode='all'):
    """Run the fixed test portfolios, comparison plots and one-at-a-time sensitivities."""
    sys.path[:0] = [str(HERE.parent), str(REGIME_RESULTS.parent)]
    train, validation, test, history, forecasts = load_testing_backtest_inputs()
    baseline = TestSettings()
    common = dict(initial_date=validation.index[-1])
    if mode in ('baseline', 'comparison', 'all'):
        results = walk_forward_test(test, history, forecasts, [baseline])
        heuristic = test_heuristic(train, test, forecasts)
    if mode in ('baseline', 'all'):
        for model, (source, _) in TEST_MODELS.items():
            forecast = TEST_FORECAST_NAMES[source]
            write_test_report(TEST_RESULTS / model, {model: results[model, baseline]},
                              {model: baseline.btc_cap}, forecast=forecast, **common)
        write_test_report(TEST_RESULTS / '3-Asset Simple Heuristic', heuristic,
                          dict.fromkeys(heuristic, baseline.btc_cap), forecast='HMM and Jump Model',
                          heuristic=True, **common)
    if mode in ('comparison', 'all'):
        from regime_comparison_figures import save_testing_paths

        comparison = {model: results[model, baseline] for model in TEST_MODELS}
        comparison.update({f'3-asset heuristic ({forecast})': run for forecast, run in heuristic.items()})
        save_testing_paths(comparison, output=TEST_RESULTS, **common)
        print(f'Saved comparison figures in {TEST_RESULTS.relative_to(HERE.parent)}', flush=True)
    if mode in ('sensitivity', 'all'):
        grid = {parameter: {
            f'{value*10000:g} bps' if form == 'bps' else format(value, form): replace(baseline, **{field: value})
            for value in values} for parameter, (field, values, form) in TEST_SWEEPS.items()}
        unique = list(dict.fromkeys(cfg for sweep in grid.values() for cfg in sweep.values()))
        results = walk_forward_test(test, history, forecasts, unique)
        for model, (source, _) in TEST_MODELS.items():
            for parameter, sweep in grid.items():
                write_test_report(SENSITIVITY_RESULTS / model / parameter,
                                  {label: results[model, cfg] for label, cfg in sweep.items()},
                                  {label: cfg.btc_cap for label, cfg in sweep.items()},
                                  forecast=TEST_FORECAST_NAMES[source], parameter=parameter, **common)


def performance(daily, monthly):
    returns = daily['Net daily return'].to_numpy()
    wealth = np.r_[1., np.cumprod(1+returns)]
    volatility = returns.std(ddof=1)*np.sqrt(252)
    full = monthly[monthly['Complete month']]
    es = cvar.loss_var_cvar(full['Net period return'].to_numpy())[1] if len(full) else np.nan
    return {'Net total return': wealth[-1]-1, 'Net CAGR': wealth[-1]**(252/len(returns))-1,
        'Net annualized volatility': volatility,
        'Net Sharpe zero risk-free': returns.mean()*252/volatility if volatility else np.nan,
        'Net maximum drawdown': np.min(wealth/np.maximum.accumulate(wealth)-1),
        'Net empirical monthly CVaR95': es,
        'Net empirical daily CVaR95': cvar.loss_var_cvar(returns)[1],
        'Net VaR breaches': int(full['Net VaR breach'].sum()), 'Complete months': len(full),
        'Executed months': len(monthly), 'Executed days': len(daily),
        'Cumulative one-way turnover': monthly['One-way turnover'].sum(),
        'Sum fee fractions': monthly['Fee fraction'].sum(),
        **{f'Average {asset}': monthly[asset].mean() for asset in ASSETS},
        'Max target BTC': monthly.BTC.max()}


def plot_loss_distribution(portfolio_returns, *, alpha=.95, title='Portfolio loss distribution'):
    """Display and return the histogram of simulated gross portfolio losses."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    returns = np.asarray(portfolio_returns, dtype=float)
    var, cvar_loss = cvar.loss_var_cvar(returns, alpha)
    confidence = f'{alpha:.0%}'
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    ax.hist(-returns, bins=50, color='#447b91', edgecolor='white', linewidth=.35)
    right = ax.get_xlim()[1]
    ax.axvspan(var, right, color='#c35646', alpha=.13, zorder=0, label='Losses beyond VaR')
    ax.axvline(0, color='#87949a', linewidth=.9)
    ax.axvline(var, color='#a7601e', linewidth=1.6, label=f'{confidence} VaR ({var:.2%})')
    ax.axvline(cvar_loss, color='#962f38', linewidth=1.6, label=f'{confidence} CVaR ({cvar_loss:.2%})')
    ax.set_title(title)
    ax.set_xlabel('Gross portfolio loss')
    ax.set_ylabel('Scenario count')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e0e6e9', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    plt.show()
    return fig


def plot_saved_regime_losses(origin, forecast, stage):
    """Use the saved three-month matched-return plan and its own DCC-GARCH draws."""
    origin = date.fromisoformat(str(origin)).isoformat()
    manifest = json.loads((REGIME_RESULTS / 'run_manifest.json').read_text(encoding='utf-8'))
    if not manifest['completed']:
        raise ValueError('The regime-aware CVaR run is incomplete')
    decision_name = f'decisions/{origin}.json'
    decision_file = REGIME_RESULTS / decision_name
    if sha(decision_file) != manifest['output_sha256'].get(decision_name):
        raise ValueError(f'Saved CVaR decision differs from its run manifest for {origin}')
    decision = json.loads(decision_file.read_text(encoding='utf-8'))
    forecast_file = REGIME_FORECASTS / 'forecasts' / f'{origin}.npz'
    if sha(forecast_file) != decision['source_forecast_sha256'] or decision['scenario_key'] != 'DCC GARCH':
        raise ValueError(f'DCC-GARCH draws differ from the saved CVaR decision for {origin}')

    plan = decision['plans'][f'{forecast} | matched_return | H=3']
    with np.load(forecast_file, allow_pickle=False) as data:
        scenario_returns = data['DCC GARCH'][:, stage-1]
    returns = scenario_returns @ np.asarray(plan['weights'][stage-1])
    holding_month = pd.Period(origin, freq='M') + stage - 1
    forecast_label = {'hmm': 'HMM', 'jump': 'Jump Model'}[forecast]
    title = (f'{forecast_label} matched-return CVaR\n'
             f'{holding_month:%B %Y} DCC-GARCH scenarios, planned {origin}')
    return plot_loss_distribution(returns, alpha=plan['alpha'], title=title)


def run(train_path=DATA/'portfolio_train.csv', validation_path=DATA/'portfolio_validation.csv',
        output=HERE/'results', *, calendar_path=None, models=MODELS, horizons=(1, 3),
        n_scenarios=10000, seed=42, solver='cuts'):
    models, horizons, output = tuple(models), tuple(horizons), Path(output).resolve()
    if not models or len(set(models)) != len(models) or not set(models).issubset(MODELS):
        raise ValueError('Select distinct models from FHS GARCH, FHS Hybrid, DCC GARCH and DCC Hybrid')
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h not in (1, 3) for h in horizons)):
        raise ValueError('Select horizon 1, 3, or both')
    if solver not in ('cuts', 'full'):
        raise ValueError('Solver must be cuts or full')
    # Validate configuration before loading data or writing any results.
    scenarios.ForecastConfig(n_scenarios=n_scenarios, seed=seed)
    train_path, validation_path = Path(train_path).resolve(), Path(validation_path).resolve()
    if train_path == validation_path:
        raise ValueError('Training and validation files must be different')
    train, validation = load_returns(train_path), load_returns(validation_path)
    if train.index[-1] >= validation.index[0]:
        raise ValueError('Training data must end strictly before validation starts')
    calendar = known_calendar(calendar_path)
    expected = calendar[(calendar >= validation.index[0]) & (calendar <= validation.index[-1])]
    if not validation.index.equals(expected):
        raise ValueError('Evaluation dates disagree with the known calendar; supply --calendar for other dates')
    groups = list(validation.groupby(validation.index.to_period('M')))
    periods_by_origin = [scenarios.monthly_periods(block.index[0], calendar, max(horizons)) for _, block in groups]
    for i, ((_, block), periods) in enumerate(zip(groups, periods_by_origin)):
        if not block.index.equals(periods[0][:len(block)]):
            raise ValueError('Each evaluation month must start at its first known trading session')
        if not block.index.equals(periods[0]) and i != len(groups)-1:
            raise ValueError('Only the final evaluation month may be partial')
    protocol = {'models': list(models), 'horizons': list(horizons), 'alpha': .95, 'risk_aversion': .5,
        'return_reward': 0., 'upper': [1., 1., .2], 'cost_rate': .001, 'scenarios': n_scenarios, 'seed': seed,
        'rolling_years': 5, 'minimum_prior_observations': 750,
        'mean': 'Native stage scenario sample means; original five-year Bond drift',
        'target': 'Original maximum-Sharpe expected return minus 1e-6 per holding day',
        'risk': 'Sum of incremental monthly CVaRs; not terminal-wealth CVaR',
        'execution': 'Monthly reoptimization; first allocation only; exact self-financing fees and daily holding drift',
        'initial_holdings': [0., 0., 0.], 'terminal_liquidation': False, 'solver': solver,
        'calendar': 'Original published 2022-2024 NYSE schedule' if calendar_path is None else 'Explicit supplied known calendar',
        'partial_month': 'Included in daily performance, excluded from monthly CVaR and VaR coverage'}
    environment = {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                    for p in ('numpy', 'pandas', 'scipy', 'arch')}}
    identity = {'protocol': protocol, 'environment': environment,
        'source_sha256': {Path(p).name: sha(p) for p in (__file__, cvar.__file__, scenarios.__file__)},
        'input_sha256': {'train': sha(train_path), 'validation': sha(validation_path)},
        'calendar_sha256': None if calendar_path is None else sha(calendar_path)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output/'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Output contains another configuration; use a different --output directory')
        if previous['completed']:
            for relative, checksum in previous['output_sha256'].items():
                if sha(output/relative) != checksum:
                    raise ValueError(f'Completed output changed: {relative}')
            print('Reusing checksum-verified completed run. Use --output for a fresh run.', flush=True)
            return pd.read_csv(output/'portfolio_summary.csv')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', protocol)
    write_json(manifest_path, {**identity, 'fingerprint': fingerprint, 'completed': False,
        'started_utc': datetime.now(timezone.utc).isoformat()})
    states = {(model, h): {'prior': np.zeros(3), 'wealth': 1.} for model in models for h in horizons}
    monthly, daily, planned = [], [], []
    elapsed = validation.iloc[:0].copy()
    for i, ((_, block), periods) in enumerate(zip(groups, periods_by_origin)):
        started = time.monotonic()
        origin = block.index[0]
        history = pd.concat([train, elapsed])
        if history.index.max() >= origin:
            raise AssertionError('Forecast received a current or future return')
        path = output/'forecasts'/f'{origin.date()}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved['metadata']))
                if metadata['fingerprint'] != fingerprint:
                    raise ValueError('Forecast checkpoint belongs to a different run')
                cubes = {model: saved[model] for model in MODELS}
                for model, values in cubes.items():
                    checksum = hashlib.sha256(values.tobytes()).hexdigest()
                    if checksum != metadata['array_sha256'][model]:
                        raise ValueError('Forecast checkpoint checksum failed')
        else:
            forecast = scenarios.generate_scenarios(history, origin, periods, i,
                                                    n_scenarios=n_scenarios, seed=seed)
            cubes = forecast.returns
            metadata = {**forecast.metadata, 'fingerprint': fingerprint,
                'array_sha256': {model: hashlib.sha256(values.tobytes()).hexdigest() for model, values in cubes.items()}}
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **cubes, metadata=json.dumps(metadata, default=plain, allow_nan=False))
        if pd.Timestamp(metadata['history_end']) >= origin:
            raise AssertionError('Forecast checkpoint violates the information cutoff')
        plans, references = {}, {}
        for model in models:
            days = [len(period) for period in periods]
            reference = cvar.matched_return_targets(cubes[model], days)
            references[model] = reference
            cuts = cvar.CVaRCutPool()
            for h in horizons:
                plans[model, h] = cvar.optimize_matched_cvar(cubes[model][:, :h], states[model, h]['prior'],
                    days[:h], reference=reference, upper=[1., 1., .2], alpha=.95, risk_aversion=.5,
                    cost_rate=.001, use_cut_backend=solver == 'cuts', cut_pool=cuts if solver == 'cuts' else None)
        # Save every decision before allowing current-month values into execution.
        write_json(output/'decisions'/f'{origin.date()}.json', {
            'origin': str(origin.date()), 'history_end': metadata['history_end'], 'forecast_sha256': sha(path),
            'current_month_outcomes_supplied_to_decisions': 0,
            'references': {m: asdict(ref) for m, ref in references.items()},
            'plans': {f'{m} H={h}': asdict(p) for (m, h), p in plans.items()}})
        observed = block.to_numpy()
        complete = block.index.equals(periods[0])
        for (model, h), plan in plans.items():
            previous = states[model, h]
            executed = cvar.execute_first_period(plan, observed, initial_wealth=previous['wealth'])
            first = plan.first_weights
            var, es = cvar.loss_var_cvar((1-executed.fee_fraction)*(1+cubes[model][:, 0] @ first)-1)
            common = {'Origin': str(origin.date()), 'Model': model, 'Horizon': h}
            monthly.append({**common, **dict(zip(ASSETS, first)), 'Complete month': complete, 'Days': len(block),
                'Gross return floor': references[model].targets[0], 'Expected return': plan.expected_returns[0],
                'Gross period return': executed.gross_period_return, 'Net period return': executed.net_period_return,
                'Fee fraction': executed.fee_fraction,
                'One-way turnover': .5*np.abs(first-previous['prior']).sum() if previous['prior'].any() else 0.,
                'Gross forecast CVaR': plan.forecast_cvar[0], 'Net forecast VaR': var, 'Net forecast CVaR': es,
                'Net VaR breach': int(-executed.net_period_return > var) if complete else None})
            for date, gross, net in zip(block.index, executed.gross_daily_returns, executed.net_daily_returns):
                daily.append({**common, 'Date': str(date.date()), 'Gross daily return': gross, 'Net daily return': net})
            for k in range(h):
                planned.append({**common, 'Stage': k+1, **dict(zip(ASSETS, plan.weights[k])),
                    'Gross return floor': references[model].targets[k], 'Expected return': plan.expected_returns[k],
                    'Forecast CVaR': plan.forecast_cvar[k], 'Estimated cost': plan.estimated_costs[k]})
            states[model, h] = {'prior': executed.ending_weights, 'wealth': executed.ending_wealth}
        elapsed = pd.concat([elapsed, block])
        print(f'{i+1:02d}/{len(groups)} {origin.date()}: {len(plans)} decisions | '
              f'{time.monotonic()-started:.1f}s'+(' | partial month' if not complete else ''), flush=True)
    monthly, daily, planned = map(pd.DataFrame, (monthly, daily, planned))
    rows = []
    for model in models:
        for h in horizons:
            m = monthly[(monthly.Model == model) & (monthly.Horizon == h)]
            d = daily[(daily.Model == model) & (daily.Horizon == h)]
            rows.append({'Model': model, 'Horizon': h, **performance(d, m)})
    summary = pd.DataFrame(rows)
    wide = summary.pivot(index='Model', columns='Horizon', values='Net Sharpe zero risk-free').reindex(models)
    wide = wide.rename(columns={h: f'H={h} Sharpe' for h in horizons})
    for name, frame in {'portfolio_monthly': monthly, 'portfolio_daily': daily, 'planned_allocations': planned,
                        'portfolio_summary': summary, 'sharpe_comparison': wide.reset_index()}.items():
        frame.to_csv(output/f'{name}.csv', index=False)
    write_json(manifest_path, {**identity, 'fingerprint': fingerprint, 'completed': True,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'evaluation_start': str(validation.index[0].date()),
        'evaluation_end': str(validation.index[-1].date()), 'evaluation_days': len(validation),
        'monthly_decisions': len(monthly), 'output_sha256': {
            p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p != manifest_path}})
    print(wide.to_string(float_format=lambda x: f'{x:.6f}'), flush=True)
    print(f'Results: {output}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=DATA/'portfolio_train.csv')
    parser.add_argument('--validation', type=Path, default=DATA/'portfolio_validation.csv')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--calendar', type=Path, help='Optional complete ex-ante trading calendar CSV with Date column')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--horizons', nargs='+', type=int, choices=(1, 3), default=(1, 3))
    parser.add_argument('--scenarios', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--solver', choices=('cuts', 'full'), default='cuts')
    parser.add_argument('--plot-loss-distribution', type=date.fromisoformat, metavar='YYYY-MM-DD',
                        help='Display saved regime-aware DCC-GARCH losses for this rebalance')
    parser.add_argument('--plot-forecast', choices=('hmm', 'jump'), default='jump')
    parser.add_argument('--plot-stage', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--prepare-test-forecasts', choices=('hmm', 'jump', 'dcc'),
                        help='Prepare one forecast source for the held-out test split')
    parser.add_argument('--run-testing', nargs='?', const='all', choices=('all', 'baseline', 'comparison', 'sensitivity'),
                        help='Run the held-out portfolios, comparison figures and sensitivity analysis')
    args = parser.parse_args()
    if args.plot_loss_distribution is not None:
        plot_saved_regime_losses(args.plot_loss_distribution, args.plot_forecast, args.plot_stage)
        return
    if args.prepare_test_forecasts:
        prepare_testing_forecasts(args.prepare_test_forecasts)
        return
    if args.run_testing:
        run_testing(args.run_testing)
        return
    run(args.train, args.validation, args.output, calendar_path=args.calendar, models=args.models,
        horizons=args.horizons, n_scenarios=args.scenarios, seed=args.seed, solver=args.solver)


if __name__ == '__main__':
    main()

import platform
import time

import numpy as np
import pandas as pd
import cvar_mpo as cvar
import scenario_generation as scenarios

HERE = Path(__file__).resolve().parent
ASSETS = scenarios.ASSETS
MODELS = scenarios.MODELS
# Original NYSE schedules published in 2021, before every supplied evaluation
# origin. Early-close sessions remain. Do not build future calendars from returns.
HOLIDAYS = {
    2022: ['01-17', '02-21', '04-15', '05-30', '06-20', '07-04', '09-05', '11-24', '12-26'],
    2023: ['01-02', '01-16', '02-20', '04-07', '05-29', '06-19', '07-04', '09-04', '11-23', '12-25'],
    2024: ['01-01', '01-15', '02-19', '03-29', '05-27', '06-19', '07-04', '09-02', '11-28', '12-25']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False)+'\n', encoding='utf-8')


def known_calendar(path=None):
    """Use a supplied ex-ante calendar or the frozen 2022-2024 NYSE schedule."""
    if path is None:
        closed = pd.DatetimeIndex([f'{year}-{day}' for year, days in HOLIDAYS.items() for day in days])
        return pd.bdate_range('2022-01-01', '2024-12-31').difference(closed)
    frame = pd.read_csv(path)
    if 'Date' not in frame.columns:
        raise ValueError('Calendar CSV must contain a Date column with complete future trading sessions')
    dates = pd.DatetimeIndex(pd.to_datetime(frame['Date'], errors='raise'))
    if (dates.empty or dates.hasnans or dates.tz is not None or dates.has_duplicates
            or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())):
        raise ValueError('Calendar dates must be ordered, unique, timezone-naive daily dates')
    return dates


def load_returns(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if set(frame.columns) != set(ASSETS):
        raise ValueError(f'{Path(path).name} must contain exactly Equity, Bond and BTC daily return columns')
    frame = frame.loc[:, list(ASSETS)].astype(float)
    if (frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
            or frame.index.tz is not None or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or not frame.index.equals(frame.index.normalize()) or not np.isfinite(frame.to_numpy()).all()
            or (frame <= -1).any().any()):
        raise ValueError(f'Invalid chronology or simple returns in {Path(path).name}')
    return frame


def performance(daily, monthly):
    returns = daily['Net daily return'].to_numpy()
    wealth = np.r_[1., np.cumprod(1+returns)]
    volatility = returns.std(ddof=1)*np.sqrt(252)
    full = monthly[monthly['Complete month']]
    es = cvar.loss_var_cvar(full['Net period return'].to_numpy())[1] if len(full) else np.nan
    return {'Net total return': wealth[-1]-1, 'Net CAGR': wealth[-1]**(252/len(returns))-1,
        'Net annualized volatility': volatility,
        'Net Sharpe zero risk-free': returns.mean()*252/volatility if volatility else np.nan,
        'Net maximum drawdown': np.min(wealth/np.maximum.accumulate(wealth)-1),
        'Net empirical monthly CVaR95': es,
        'Net empirical daily CVaR95': cvar.loss_var_cvar(returns)[1],
        'Net VaR breaches': int(full['Net VaR breach'].sum()), 'Complete months': len(full),
        'Executed months': len(monthly), 'Executed days': len(daily),
        'Cumulative one-way turnover': monthly['One-way turnover'].sum(),
        'Sum fee fractions': monthly['Fee fraction'].sum(),
        **{f'Average {asset}': monthly[asset].mean() for asset in ASSETS},
        'Max target BTC': monthly.BTC.max()}


def run(train_path=HERE/'data/portfolio_train.csv', validation_path=HERE/'data/portfolio_validation.csv',
        output=HERE/'results', *, calendar_path=None, models=MODELS, horizons=(1, 3),
        n_scenarios=10000, seed=42, solver='cuts'):
    models, horizons, output = tuple(models), tuple(horizons), Path(output).resolve()
    if not models or len(set(models)) != len(models) or not set(models).issubset(MODELS):
        raise ValueError('Select distinct models from FHS GARCH, FHS Hybrid, DCC GARCH and DCC Hybrid')
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h not in (1, 3) for h in horizons)):
        raise ValueError('Select horizon 1, 3, or both')
    if solver not in ('cuts', 'full'):
        raise ValueError('Solver must be cuts or full')
    # Validate configuration before loading data or writing any results.
    scenarios.ForecastConfig(n_scenarios=n_scenarios, seed=seed)
    train_path, validation_path = Path(train_path).resolve(), Path(validation_path).resolve()
    if train_path == validation_path:
        raise ValueError('Training and validation files must be different')
    train, validation = load_returns(train_path), load_returns(validation_path)
    if train.index[-1] >= validation.index[0]:
        raise ValueError('Training data must end strictly before validation starts')
    calendar = known_calendar(calendar_path)
    expected = calendar[(calendar >= validation.index[0]) & (calendar <= validation.index[-1])]
    if not validation.index.equals(expected):
        raise ValueError('Evaluation dates disagree with the known calendar; supply --calendar for other dates')
    groups = list(validation.groupby(validation.index.to_period('M')))
    periods_by_origin = [scenarios.monthly_periods(block.index[0], calendar, max(horizons)) for _, block in groups]
    for i, ((_, block), periods) in enumerate(zip(groups, periods_by_origin)):
        if not block.index.equals(periods[0][:len(block)]):
            raise ValueError('Each evaluation month must start at its first known trading session')
        if not block.index.equals(periods[0]) and i != len(groups)-1:
            raise ValueError('Only the final evaluation month may be partial')
    protocol = {'models': list(models), 'horizons': list(horizons), 'alpha': .95, 'risk_aversion': .5,
        'return_reward': 0., 'upper': [1., 1., .2], 'cost_rate': .001, 'scenarios': n_scenarios, 'seed': seed,
        'rolling_years': 5, 'minimum_prior_observations': 750,
        'mean': 'Native stage scenario sample means; original five-year Bond drift',
        'target': 'Original maximum-Sharpe expected return minus 1e-6 per holding day',
        'risk': 'Sum of incremental monthly CVaRs; not terminal-wealth CVaR',
        'execution': 'Monthly reoptimization; first allocation only; exact self-financing fees and daily holding drift',
        'initial_holdings': [0., 0., 0.], 'terminal_liquidation': False, 'solver': solver,
        'calendar': 'Original published 2022-2024 NYSE schedule' if calendar_path is None else 'Explicit supplied known calendar',
        'partial_month': 'Included in daily performance, excluded from monthly CVaR and VaR coverage'}
    environment = {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                    for p in ('numpy', 'pandas', 'scipy', 'arch')}}
    identity = {'protocol': protocol, 'environment': environment,
        'source_sha256': {Path(p).name: sha(p) for p in (__file__, cvar.__file__, scenarios.__file__)},
        'input_sha256': {'train': sha(train_path), 'validation': sha(validation_path)},
        'calendar_sha256': None if calendar_path is None else sha(calendar_path)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output/'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Output contains another configuration; use a different --output directory')
        if previous['completed']:
            for relative, checksum in previous['output_sha256'].items():
                if sha(output/relative) != checksum:
                    raise ValueError(f'Completed output changed: {relative}')
            print('Reusing checksum-verified completed run. Use --output for a fresh run.', flush=True)
            return pd.read_csv(output/'portfolio_summary.csv')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', protocol)
    write_json(manifest_path, {**identity, 'fingerprint': fingerprint, 'completed': False,
        'started_utc': datetime.now(timezone.utc).isoformat()})
    states = {(model, h): {'prior': np.zeros(3), 'wealth': 1.} for model in models for h in horizons}
    monthly, daily, planned = [], [], []
    elapsed = validation.iloc[:0].copy()
    for i, ((_, block), periods) in enumerate(zip(groups, periods_by_origin)):
        started = time.monotonic()
        origin = block.index[0]
        history = pd.concat([train, elapsed])
        if history.index.max() >= origin:
            raise AssertionError('Forecast received a current or future return')
        path = output/'forecasts'/f'{origin.date()}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved['metadata']))
                if metadata['fingerprint'] != fingerprint:
                    raise ValueError('Forecast checkpoint belongs to a different run')
                cubes = {model: saved[model] for model in MODELS}
                for model, values in cubes.items():
                    checksum = hashlib.sha256(values.tobytes()).hexdigest()
                    if checksum != metadata['array_sha256'][model]:
                        raise ValueError('Forecast checkpoint checksum failed')
        else:
            forecast = scenarios.generate_scenarios(history, origin, periods, i,
                                                    n_scenarios=n_scenarios, seed=seed)
            cubes = forecast.returns
            metadata = {**forecast.metadata, 'fingerprint': fingerprint,
                'array_sha256': {model: hashlib.sha256(values.tobytes()).hexdigest() for model, values in cubes.items()}}
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **cubes, metadata=json.dumps(metadata, default=plain, allow_nan=False))
        if pd.Timestamp(metadata['history_end']) >= origin:
            raise AssertionError('Forecast checkpoint violates the information cutoff')
        plans, references = {}, {}
        for model in models:
            days = [len(period) for period in periods]
            reference = cvar.matched_return_targets(cubes[model], days)
            references[model] = reference
            cuts = cvar.CVaRCutPool()
            for h in horizons:
                plans[model, h] = cvar.optimize_matched_cvar(cubes[model][:, :h], states[model, h]['prior'],
                    days[:h], reference=reference, upper=[1., 1., .2], alpha=.95, risk_aversion=.5,
                    cost_rate=.001, use_cut_backend=solver == 'cuts', cut_pool=cuts if solver == 'cuts' else None)
        # Save every decision before allowing current-month values into execution.
        write_json(output/'decisions'/f'{origin.date()}.json', {
            'origin': str(origin.date()), 'history_end': metadata['history_end'], 'forecast_sha256': sha(path),
            'current_month_outcomes_supplied_to_decisions': 0,
            'references': {m: asdict(ref) for m, ref in references.items()},
            'plans': {f'{m} H={h}': asdict(p) for (m, h), p in plans.items()}})
        observed = block.to_numpy()
        complete = block.index.equals(periods[0])
        for (model, h), plan in plans.items():
            previous = states[model, h]
            executed = cvar.execute_first_period(plan, observed, initial_wealth=previous['wealth'])
            first = plan.first_weights
            var, es = cvar.loss_var_cvar((1-executed.fee_fraction)*(1+cubes[model][:, 0] @ first)-1)
            common = {'Origin': str(origin.date()), 'Model': model, 'Horizon': h}
            monthly.append({**common, **dict(zip(ASSETS, first)), 'Complete month': complete, 'Days': len(block),
                'Gross return floor': references[model].targets[0], 'Expected return': plan.expected_returns[0],
                'Gross period return': executed.gross_period_return, 'Net period return': executed.net_period_return,
                'Fee fraction': executed.fee_fraction,
                'One-way turnover': .5*np.abs(first-previous['prior']).sum() if previous['prior'].any() else 0.,
                'Gross forecast CVaR': plan.forecast_cvar[0], 'Net forecast VaR': var, 'Net forecast CVaR': es,
                'Net VaR breach': int(-executed.net_period_return > var) if complete else None})
            for date, gross, net in zip(block.index, executed.gross_daily_returns, executed.net_daily_returns):
                daily.append({**common, 'Date': str(date.date()), 'Gross daily return': gross, 'Net daily return': net})
            for k in range(h):
                planned.append({**common, 'Stage': k+1, **dict(zip(ASSETS, plan.weights[k])),
                    'Gross return floor': references[model].targets[k], 'Expected return': plan.expected_returns[k],
                    'Forecast CVaR': plan.forecast_cvar[k], 'Estimated cost': plan.estimated_costs[k]})
            states[model, h] = {'prior': executed.ending_weights, 'wealth': executed.ending_wealth}
        elapsed = pd.concat([elapsed, block])
        print(f'{i+1:02d}/{len(groups)} {origin.date()}: {len(plans)} decisions | '
              f'{time.monotonic()-started:.1f}s'+(' | partial month' if not complete else ''), flush=True)
    monthly, daily, planned = map(pd.DataFrame, (monthly, daily, planned))
    rows = []
    for model in models:
        for h in horizons:
            m = monthly[(monthly.Model == model) & (monthly.Horizon == h)]
            d = daily[(daily.Model == model) & (daily.Horizon == h)]
            rows.append({'Model': model, 'Horizon': h, **performance(d, m)})
    summary = pd.DataFrame(rows)
    wide = summary.pivot(index='Model', columns='Horizon', values='Net Sharpe zero risk-free').reindex(models)
    wide = wide.rename(columns={h: f'H={h} Sharpe' for h in horizons})
    for name, frame in {'portfolio_monthly': monthly, 'portfolio_daily': daily, 'planned_allocations': planned,
                        'portfolio_summary': summary, 'sharpe_comparison': wide.reset_index()}.items():
        frame.to_csv(output/f'{name}.csv', index=False)
    write_json(manifest_path, {**identity, 'fingerprint': fingerprint, 'completed': True,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'evaluation_start': str(validation.index[0].date()),
        'evaluation_end': str(validation.index[-1].date()), 'evaluation_days': len(validation),
        'monthly_decisions': len(monthly), 'output_sha256': {
            p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p != manifest_path}})
    print(wide.to_string(float_format=lambda x: f'{x:.6f}'), flush=True)
    print(f'Results: {output}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=HERE/'data/portfolio_train.csv')
    parser.add_argument('--validation', type=Path, default=HERE/'data/portfolio_validation.csv')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--calendar', type=Path, help='Optional complete ex-ante trading calendar CSV with Date column')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--horizons', nargs='+', type=int, choices=(1, 3), default=(1, 3))
    parser.add_argument('--scenarios', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--solver', choices=('cuts', 'full'), default='cuts')
    args = parser.parse_args()
    run(args.train, args.validation, args.output, calendar_path=args.calendar, models=args.models,
        horizons=args.horizons, n_scenarios=args.scenarios, seed=args.seed, solver=args.solver)


if __name__ == '__main__':
    main()
