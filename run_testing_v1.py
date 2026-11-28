"""Run the selected held-out portfolios and one-at-a-time CVaR sensitivities."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import scenario_generation as scenarios

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'CVaR Regime Models')]

import portfolio_metrics as metrics
from regime_cvar import (cvar, load_returns, optimize_forecast_cvar, optimize_switching_cvar,
                         regimes_from_bear_probabilities, sha, write_json)

ASSETS = cvar.ASSETS
DATA = ROOT / 'Data'
OUTPUT = ROOT / 'Testing Results v1'
FORECAST_ROOT = OUTPUT / '_forecast_inputs'
SENSITIVITY_SUMMARY = ROOT / 'sensitivity analysis' / 'Sensitivity Summary v1.md'
TEST_RESULTS = ROOT / 'Testing Results'
TEST_FORECASTS = TEST_RESULTS / '_forecast_inputs'
SENSITIVITY_RESULTS = ROOT / 'sensitivity analysis'
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

@dataclass(frozen=True)
class TestSettings:
    alpha: float = .95
    risk_aversion: float = .5
    btc_cap: float = .20
    cost_rate: float = .001


MODELS = {
    "HMM Regime-Aware CVaR": ("hmm", False),
    "HMM Regime-Switching CVaR": ("hmm", True),
    "Jump Regime-Switching CVaR": ("jump", True),
}
BASELINE = TestSettings()
SWEEPS = {
    "Transaction Costs": ("cost_rate", (.0005, .001, .0015, .002), "bps"),
    "Bitcoin Allocation Cap": TEST_SWEEPS["Bitcoin Allocation Cap"],
    "CVaR Confidence Level": TEST_SWEEPS["CVaR Confidence Level"],
    "Risk Aversion Parameter": TEST_SWEEPS["Risk Aversion Parameter"],
}


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


def prepare_testing_forecasts(source, *, forecast_root=TEST_FORECASTS):
    """Prepare one set of forecasts for the held-out test split."""
    from functools import partial

    train, _, test, history, calendar = testing_inputs()
    folder = forecast_root / source
    folder.mkdir(parents=True, exist_ok=True)
    forecast_rows = []
    settings = {}
    hmm_dir = ROOT / '2-State HMM' / 'HMM Forecasts'
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
        penalty = jump.select_settings(train, calendar)
        fit = partial(jump.fit_jump, penalty=penalty)
        settings = {'penalty': float(penalty), 'mean_estimation': 'Unpooled state means',
                    'tuned_on': 'training only'}

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
    write_json(manifest_path, {'identity': identity, 'settings': settings,
                               'output_sha256': {name: sha(folder / name) for name in outputs}})


def load_testing_backtest_inputs(*, forecast_root=TEST_FORECASTS):
    """Verify the shared test forecasts and load the three return splits."""
    for source in ('hmm', 'jump', 'dcc'):
        prepare_testing_forecasts(source, forecast_root=forecast_root)
    train, validation, test, history, _ = testing_inputs()
    forecasts = {key: pd.read_csv(forecast_root / key / 'forecast_inputs.csv',
                                 parse_dates=['origin', 'information_date'])
                 for key in TEST_FORECAST_NAMES}
    return train, validation, test, history, forecasts


def load_testing_rebalance(origin, opening_date, forecasts, *, forecast_root=TEST_FORECASTS):
    """Read a common DCC ensemble and forecasts available before the rebalance."""
    with np.load(forecast_root / 'dcc' / f'{origin.date()}.npz', allow_pickle=False) as saved:
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


def walk_forward_test(test, history, forecasts, settings, *, models=TEST_MODELS,
                      forecast_root=TEST_FORECASTS, stage_audit=None):
    """Evaluate each optimizer and setting on a separate account starting in cash."""
    configs = [(model, setting) for model in models for setting in settings]
    accounts = {key: {'prior': np.zeros(3), 'wealth': 1., 'monthly': [], 'daily': []} for key in configs}
    groups = list(test.groupby(test.index.to_period('M')))
    for i, (_, block) in enumerate(groups):
        origin = block.index[0]
        opening_date = history.index[history.index < origin][-1]
        cube, days, signals, means = load_testing_rebalance(origin, opening_date, forecasts,
                                                          forecast_root=forecast_root)
        regimes = {source: regimes_from_bear_probabilities(p) for source, p in signals.items()}
        pools = {rate: cvar.CVaRCutPool() for rate in {cfg.alpha for cfg in settings}}
        common = {'origin': origin, 'common_regime': regimes['hmm'][0]}
        for model, cfg in configs:
            source, switching = models[model]
            state = accounts[model, cfg]
            prior = state['prior']
            kwargs = dict(alpha=cfg.alpha, risk_aversion=cfg.risk_aversion, cost_rate=cfg.cost_rate,
                          upper=(1., 1., cfg.btc_cap), mean_source=TEST_FORECAST_NAMES[source],
                          scenario_source='DCC GARCH', cut_pool=pools[cfg.alpha])
            if switching:
                plan = optimize_switching_cvar(cube, prior, days, means[source], regimes[source], **kwargs)
            else:
                plan = optimize_forecast_cvar(cube, prior, days, means[source], policy='matched_return', **kwargs)
            if stage_audit is not None:
                floors = plan.metadata['gross_return_targets']
                active = regimes[source] == 'bull' if switching else np.ones(len(days), dtype=bool)
                np.testing.assert_array_equal([floor is not None for floor in floors], active)
                portfolio_means = np.sum(means[source] * plan.weights, axis=1)
                np.testing.assert_allclose(plan.expected_returns, portfolio_means, atol=1e-12, rtol=0)
                np.testing.assert_array_less(np.asarray(floors, dtype=float)[active] - 2e-7,
                                             portfolio_means[active])
                for k, (mean, weights) in enumerate(zip(means[source], plan.weights, strict=True)):
                    stage_audit.append({
                        'model': model, 'forecast': TEST_FORECAST_NAMES[source], 'origin': origin,
                        'information_date': opening_date, 'stage': k + 1, 'holding_days': days[k],
                        'scenario_count': len(cube), 'p_bear_month_end': signals[source][k],
                        'forecast_regime': regimes[source][k],
                        'stage_policy': 'matched_return' if active[k] else 'min_cvar',
                        **dict(zip(ASSETS, mean)), 'return_floor': floors[k],
                        'expected_portfolio_return': portfolio_means[k],
                        **{f'weight_{asset}': weight for asset, weight in zip(ASSETS, weights)},
                    })
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


def test_report_metrics(runs, caps):
    """Compute the validation-format statistics from a set of test accounts."""
    rows = []
    for label, (daily, monthly) in runs.items():
        episodes = metrics.regime_episodes(daily)
        np.testing.assert_allclose((1+daily.net_return).groupby(daily.origin).prod()-1,
                                   monthly.net_return, atol=1e-12, rtol=0)
        np.testing.assert_allclose((1+monthly.net_return).cumprod(), monthly.ending_wealth, atol=1e-12, rtol=0)
        for regime_key, regime in metrics.REGIMES:
            ds = daily if regime_key == 'all' else daily.loc[daily.common_regime.eq(regime_key)]
            ms = monthly if regime_key == 'all' else monthly.loc[monthly.common_regime.eq(regime_key)]
            dd = None if regime_key == 'all' else episodes.loc[episodes.common_regime.eq(regime_key), 'max_drawdown'].min()
            values = {} if ds.empty else metrics.group_statistics(
                ds, ms, episode_drawdown=dd, btc_cap=caps[label])
            rows.append({**values, 'label': label, 'regime': regime})
    return metrics, pd.DataFrame(rows)


def write_test_report(output, runs, caps, *, forecast, initial_date, parameter=None, heuristic=False,
                      model_context=None, figures=True):
    """Write the two validation-format tables and, optionally, figures."""
    metrics, table = test_report_metrics(runs, caps)
    table['forecast'] = table.label if parameter or heuristic else forecast
    table['scenarios'] = 'Rule-based' if heuristic else 'DCC-GARCH'
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
    if figures:
        from regime_comparison_figures import save_testing_figures

        save_testing_figures(runs, initial_date=initial_date, output=output,
                             forecast=forecast, heuristic=heuristic, parameter=parameter)
    print(f'Saved report in {output.relative_to(ROOT)}', flush=True)


def allocation_figure(runs, output, end, title):
    from regime_comparison_figures import save_allocations

    allocations, probabilities = {}, {}
    for forecast, (label, monthly) in runs.items():
        indexed = monthly.set_index("origin")
        allocations[forecast] = {label: indexed[list(ASSETS)]}
        probabilities[forecast] = {label: (indexed.bear_signal, indexed.threshold.iloc[0])}
    save_allocations(allocations, probabilities, end, output / "figure_2_portfolio_allocations.png",
                     width=10 if len(runs) > 1 else 8, title=title)


def comparison_figures(results, heuristic, initial_date):
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    from regime_comparison_figures import HEURISTIC, save_paths

    styles = {
        "HMM Regime-Aware CVaR": ("#176d8a", 1.8),
        "HMM Regime-Switching CVaR": ("#b95623", 1.5),
        "Jump Regime-Switching CVaR": ("#176d8a", 1.8),
        HEURISTIC: ("#675094", 1.5),
    }
    for source, name in TEST_FORECAST_NAMES.items():
        runs = {model: results[model, BASELINE] for model, (key, _) in MODELS.items() if key == source}
        runs[HEURISTIC] = heuristic[name]
        returns = pd.DataFrame({model: daily.set_index("date").net_return for model, (daily, _) in runs.items()})
        if returns.empty or returns.isna().any().any() or initial_date >= returns.index[0]:
            raise ValueError(f"{name} comparison requires aligned test returns and a prior start date")
        initial = pd.DataFrame(1., index=[initial_date], columns=returns.columns)
        wealth = pd.concat([initial, (1 + returns).cumprod()])
        drawdowns = wealth.div(wealth.cummax()).sub(1.)
        options = dict(styles=styles, legend_columns=len(runs), title=f"{name} forecasts")
        save_paths({name: wealth}, OUTPUT / f"{source}_cumulative_wealth.png",
                   ylabel="Portfolio wealth", formatter=StrMethodFormatter("${x:.2f}"), reference=1, **options)
        save_paths({name: drawdowns}, OUTPUT / f"{source}_drawdowns.png",
                   ylabel="Drawdown", formatter=PercentFormatter(1), reference=0, **options)


def verify_audit(audit, forecasts):
    for model, (source, switching) in MODELS.items():
        actual = audit[audit.model.eq(model)].reset_index(drop=True)
        expected = forecasts[source].sort_values(["origin", "stage"]).reset_index(drop=True)
        columns = ["origin", "information_date", "stage", "holding_days", "p_bear_month_end", *ASSETS]
        pd.testing.assert_frame_equal(actual[columns], expected[columns], check_exact=True)
        assert actual.scenario_count.eq(10_000).all()
        enabled = actual.forecast_regime.eq("bull") if switching else np.ones(len(actual), dtype=bool)
        np.testing.assert_array_equal(actual.return_floor.notna(), enabled)
    print(f"Verified the expected-return inputs and return-floor policies for {len(audit)} stages.", flush=True)


def write_baseline(train, validation, test, history, forecasts):
    from regime_comparison_figures import HEURISTIC

    audit_rows = []
    results = walk_forward_test(test, history, forecasts, [BASELINE], models=MODELS,
                                       forecast_root=FORECAST_ROOT, stage_audit=audit_rows)
    audit = pd.DataFrame(audit_rows)
    verify_audit(audit, forecasts)
    heuristic = test_heuristic(train, test, forecasts)
    initial_date = validation.index[-1]
    for model, (source, switching) in MODELS.items():
        output = OUTPUT / model
        daily, monthly = results[model, BASELINE]
        rule = ("The max-Sharpe return floor applies in forecast bull stages. Bear stages minimize CVaR plus trading costs."
                if switching else "The max-Sharpe return floor applies in every planning stage.")
        context = ("Each plan uses three monthly stages and 10,000 DCC-GARCH scenarios per rebalance. "
                   "The settings are 95% CVaR confidence, 0.5 risk aversion, a 20% Bitcoin cap, "
                   "and 10 bps per dollar traded. The max-Sharpe reference uses a 4% annual risk-free rate. "
                   "Expected returns average over possible future regime paths. " + rule)
        name = TEST_FORECAST_NAMES[source]
        write_test_report(output, {model: (daily, monthly)}, {model: BASELINE.btc_cap}, forecast=name,
                                 initial_date=initial_date, model_context=context, figures=False)
        audit[audit.model.eq(model)].to_csv(output / "forecast_and_plan_audit.csv", index=False)
        daily.to_csv(output / "portfolio_daily.csv", index=False)
        monthly.to_csv(output / "portfolio_monthly.csv", index=False)
        allocation_figure({name: (model, monthly)}, output, test.index[-1], model)

    output = OUTPUT / "3-Asset Simple Heuristic"
    write_test_report(output, heuristic, dict.fromkeys(heuristic, BASELINE.btc_cap),
                             forecast="HMM and Jump Model", initial_date=initial_date, heuristic=True, figures=False)
    for filename, index in (("portfolio_daily.csv", 0), ("portfolio_monthly.csv", 1)):
        pd.concat([run[index].assign(forecast=name) for name, run in heuristic.items()], ignore_index=True).to_csv(
            output / filename, index=False,
        )
    allocation_figure({name: (HEURISTIC, monthly) for name, (_, monthly) in heuristic.items()},
                      output, test.index[-1], "Three-Asset Heuristic")
    comparison_figures(results, heuristic, initial_date)
    return results


def format_table(columns, rows, label_columns):
    yield "| " + " | ".join(columns) + " |"
    yield "| " + " | ".join(["---"] * label_columns + ["---:"] * (len(columns) - label_columns)) + " |"
    for row in rows:
        yield "| " + " | ".join(row) + " |"


def setting_label(value, form):
    return f"{value * 10_000:g} bps" if form == "bps" else format(value, form)


PERFORMANCE = [
    ('net_cagr', 'Net CAGR', 100, 2),
    ('annualized_sharpe', 'Sharpe', 1, 3),
    ('monthly_var95', 'Monthly 95% VaR', 100, 2),
    ('max_drawdown', 'Max drawdown', 100, 2),
]
ALLOCATION = [
    ('average_btc_weight', 'Avg BTC weight', 100, 2),
    ('average_monthly_turnover', 'Avg monthly turnover', 100, 2),
]
FEE = ('annualized_transaction_cost', 'Annualized costs', 100, 3)
BINDING = ('btc_cap_binding_frequency', 'BTC cap binding frequency', 100, 2)


def detail_value(row, fields):
    return [f"{row[key] * scale:.{decimals}f}" for key, _, scale, decimals in fields]


def heading(fields, units):
    return [f"{name} ({units})" if scale == 100 else name for _, name, scale, _ in fields]


def sensitivity_summary(test, results, grid):
    labels = [(model, cfg) for model, cfg in results]
    _, metrics = test_report_metrics(results, {(model, cfg): cfg.btc_cap for model, cfg in labels})
    metrics = {row['label']: row for row in metrics.loc[metrics.regime.eq("Overall")].to_dict("records")}
    start, end = (date.strftime("%B %Y") for date in (test.index[0], test.index[-1]))
    months = test.index.to_period("M").nunique()
    sections = [
        "# CVaR sensitivity summary",
        "",
        f"These comparisons use the overall results across {len(test)} test sessions and {months} monthly "
        f"rebalances from {start} through {end}. All runs use 10,000 DCC-GARCH scenarios and a three-month "
        "planning horizon. Each sweep varies one parameter. Bold settings mark the validation settings.",
        "",
        "## Sensitivity at a glance",
        "",
        "Each entry is the spread between the highest and lowest observed value across the tested settings. "
        "Spreads for percentage metrics are in percentage points (pp). Sharpe is unitless.",
        "",
    ]
    spread_rows = []
    for parameter, sweep in grid.items():
        for model in MODELS:
            records = [metrics[model, cfg] for cfg in sweep.values()]
            spread = {key: max(row[key] for row in records) - min(row[key] for row in records)
                      for key, _, _, _ in PERFORMANCE + ALLOCATION}
            spread_rows.append([parameter, model, *detail_value(spread, PERFORMANCE + ALLOCATION)])
    sections.extend(format_table(["Parameter", "Model", *heading(PERFORMANCE + ALLOCATION, "pp")], spread_rows, 2))
    for parameter, sweep in grid.items():
        sections.extend(["", f"## {parameter}", ""])
        fields = PERFORMANCE + ALLOCATION
        if parameter == "Transaction Costs":
            sections.extend(["The setting is the fee per dollar traded. Annualized costs are the fees "
                             "incurred in the backtest expressed per year.", ""])
            fields = PERFORMANCE + [FEE] + ALLOCATION
        elif parameter == "Bitcoin Allocation Cap":
            fields = PERFORMANCE + [ALLOCATION[0], BINDING, ALLOCATION[1]]
        rows = []
        for label, cfg in sweep.items():
            setting = f"**{label}**" if cfg == BASELINE else label
            rows.extend([setting, model, *detail_value(metrics[model, cfg], fields)] for model in MODELS)
        sections.extend(format_table(["Setting", "Model", *heading(fields, "%")], rows, 2))
    SENSITIVITY_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_SUMMARY.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {SENSITIVITY_SUMMARY.name} for {len(metrics)} distinct portfolios.", flush=True)


def main():
    train, validation, test, history, forecasts = load_testing_backtest_inputs(forecast_root=FORECAST_ROOT)
    results = write_baseline(train, validation, test, history, forecasts)
    grid = {parameter: {setting_label(value, form): replace(BASELINE, **{field: value}) for value in values}
            for parameter, (field, values, form) in SWEEPS.items()}
    settings = list(dict.fromkeys(cfg for sweep in grid.values() for cfg in sweep.values() if cfg != BASELINE))
    results.update(walk_forward_test(test, history, forecasts, settings, models=MODELS,
                                            forecast_root=FORECAST_ROOT))
    sensitivity_summary(test, results, grid)


if __name__ == "__main__":
    main()

