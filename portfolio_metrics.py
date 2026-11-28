"""Portfolio metrics and report tables for the 4% risk-free runs."""
import numpy as np
import pandas as pd


REGIMES = [('all', 'Overall'), ('bull', 'Bull'), ('bear', 'Bear')]
PERIODS = 252
ANNUAL_RF = .04
BTC_CAP = .20
CAP_ATOL = 1e-6

# Numeric CSV columns carry their units. The underlying metrics.csv uses decimals.
PERFORMANCE = [
    ('net_cagr', 'Net CAGR (%)', 100),
    ('annualized_volatility', 'Annualized volatility (%)', 100),
    ('annualized_sharpe', 'Annualized Sharpe', 1),
    ('daily_var95', 'Daily 95% VaR (%)', 100),
    ('monthly_var95', 'Monthly 95% VaR (%)', 100),
    ('max_drawdown', 'Max drawdown (%)', 100),
    ('annualized_transaction_cost', 'Annualized transaction costs (%)', 100),
]
ALLOCATIONS = [
    ('average_equity_weight', 'Avg Equity weight (%)', 100),
    ('average_bond_weight', 'Avg Bond weight (%)', 100),
    ('average_btc_weight', 'Avg BTC weight (%)', 100),
    ('average_monthly_turnover', 'Avg monthly turnover (%)', 100),
    ('btc_cap_binding_frequency', 'BTC cap binding frequency (%)', 100),
]


def daily_risk_free(annual_rate, periods=252):
    if (isinstance(periods, bool) or not isinstance(periods, (int, np.integer)) or periods < 1
            or not np.isscalar(annual_rate) or not np.isfinite(annual_rate) or annual_rate <= -1):
        raise ValueError('Require a finite annual rate greater than -1 and positive integer periods')
    return float(np.expm1(np.log1p(annual_rate)/periods))


def annualized_excess_sharpe(returns, annual_rate=.04, periods=252):
    daily_rf = daily_risk_free(annual_rate, periods)
    values = np.asarray(returns, float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError('Require finite decimal daily returns greater than -1')
    if len(values) < 2 or np.ptp(values) == 0:
        return None
    excess = values-daily_rf
    return float(np.sqrt(periods)*excess.mean()/excess.std(ddof=1))


def return_vector(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all()
            or (values <= -1).any()):
        raise ValueError('Require nonempty finite simple returns greater than -1')
    return values


def geometric_annualized_return(values):
    values = return_vector(values)
    return float(np.expm1(np.log1p(values).sum() * PERIODS / len(values)))


def maximum_drawdown(values):
    """Signed peak-to-trough loss, including the peak before the first return."""
    values = return_vector(values)
    wealth = np.r_[1., np.cumprod(1 + values)]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1))


def empirical_var95(values):
    """95th percentile of realized losses at the supplied observation horizon."""
    return float(np.quantile(-return_vector(values), .95, method='linear'))


def regime_episodes(daily):
    """Reset the wealth peak at every contiguous bull/bear episode boundary."""
    daily = daily.sort_values('date').copy()
    if (daily.empty or daily.date.duplicated().any()
            or not daily.common_regime.isin(['bull', 'bear']).all()):
        raise ValueError('Episodes require unique dates and bull/bear labels')
    daily['episode_id'] = daily.common_regime.ne(daily.common_regime.shift()).cumsum()
    rows = []
    for episode_id, block in daily.groupby('episode_id', sort=False):
        returns = return_vector(block.net_return)
        rows.append({'episode_id': int(episode_id), 'common_regime': block.common_regime.iloc[0],
            'start_date': block.date.iloc[0], 'end_date': block.date.iloc[-1],
            'first_origin': block.origin.iloc[0], 'last_origin': block.origin.iloc[-1],
            'days': len(block), 'months': block.origin.nunique(),
            'net_total_return': float(np.expm1(np.log1p(returns).sum())),
            'max_drawdown': maximum_drawdown(returns)})
    return pd.DataFrame(rows)


def allocation_statistics(monthly, *, btc_cap=BTC_CAP):
    weights = monthly[['Equity', 'Bond', 'BTC']].to_numpy(float)
    if (len(weights) == 0 or not np.isfinite(weights).all()
            or (weights < -1e-8).any() or (weights[:, 2] > btc_cap + CAP_ATOL).any()):
        raise ValueError('Invalid executed target weights')
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-8, rtol=0)
    turnover = monthly.one_way_turnover.to_numpy(float)
    if not np.isfinite(turnover).all() or (turnover < 0).any() or (turnover > 1+1e-8).any():
        raise ValueError('Invalid one-way monthly turnover')
    binding = np.isclose(weights[:, 2], btc_cap, atol=CAP_ATOL, rtol=0)
    return {'average_equity_weight': float(weights[:, 0].mean()),
        'average_bond_weight': float(weights[:, 1].mean()),
        'average_btc_weight': float(weights[:, 2].mean()),
        'average_monthly_turnover': float(turnover.mean()),
        'btc_cap_binding_frequency': float(binding.mean()),
        'btc_cap_binding_months': int(binding.sum())}


def group_statistics(daily, monthly, *, episode_drawdown=None, btc_cap=BTC_CAP):
    returns = return_vector(daily.net_return)
    fees = monthly.fee_fraction.to_numpy(float)
    if not np.isfinite(fees).all() or (fees < 0).any() or (fees >= 1).any():
        raise ValueError('Invalid exact transaction fee fractions')
    full_months = monthly.loc[monthly.complete_month, 'net_return'].to_numpy(float)
    sd = float(returns.std(ddof=1)) if len(returns) > 1 else None
    return {'days': len(daily), 'months': len(monthly), 'complete_months': len(full_months),
        'net_total_return': float(np.expm1(np.log1p(returns).sum())),
        'net_cagr': geometric_annualized_return(returns),
        'annualized_volatility': None if sd is None else sd*np.sqrt(PERIODS),
        'annualized_sharpe': annualized_excess_sharpe(returns, ANNUAL_RF),
        'daily_var95': empirical_var95(returns),
        'monthly_var95': empirical_var95(full_months) if len(full_months) else None,
        'max_drawdown': maximum_drawdown(returns) if episode_drawdown is None else float(episode_drawdown),
        'sum_fee_fractions': float(fees.sum()),
        'annualized_transaction_cost': float(fees.sum()*PERIODS/len(returns)),
        **allocation_statistics(monthly, btc_cap=btc_cap)}


def presentation_table(metrics, columns):
    frame = metrics[['forecast', 'scenarios', 'regime']].rename(columns={
        'forecast': 'Forecasts', 'scenarios': 'Scenarios', 'regime': 'Regime'}).copy()
    for key, label, scale in columns:
        frame[label] = metrics[key]*scale
    return frame


def markdown_table(frame):
    lines = ['| ' + ' | '.join(frame.columns) + ' |',
             '| ' + ' | '.join('---' if i < 3 else '---:' for i in range(len(frame.columns))) + ' |']
    for row in frame.itertuples(index=False, name=None):
        cells = [str(value) if i < 3 else ('undefined' if pd.isna(value) else
                 f'{value:.3f}' if 'Sharpe' in frame.columns[i] or 'costs' in frame.columns[i]
                 else f'{value:.2f}') for i, value in enumerate(row)]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'
