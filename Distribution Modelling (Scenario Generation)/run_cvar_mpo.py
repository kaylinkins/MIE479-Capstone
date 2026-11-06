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
from pathlib import Path
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
