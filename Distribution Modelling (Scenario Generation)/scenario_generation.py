"""Standalone original FHS GARCH, FHS Hybrid, DCC GARCH and DCC Hybrid.

All forecasts use strictly prior five-year observations and original daily
arithmetic drifts for every asset, including Bond. Hybrid Bond variance is
EWMA0.94. GARCH marginals use the original skew-t likelihood; DCC uses Gaussian
QML with empirical whitened innovations, not Gaussian simulated shocks.

States continue across incremental calendar months. The original draw schedule
and float32 monthly-cache convention are preserved. No previous research outputs,
archive files, project path helper or existing scenario cache is required.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from arch import arch_model

ASSETS = ("Equity", "Bond", "BTC")
MODELS = ("FHS GARCH", "FHS Hybrid", "DCC GARCH", "DCC Hybrid")
MARGINAL_DIST = "skewt"
VOL_CAP_MULTIPLE = 1.5
MIN_VAR = 1e-10
DCC_MAX_PERSISTENCE = .998
DCC_MAX_ITER = 250
EWMA_LAMBDA = .94

@dataclass(frozen=True)
class ForecastConfig:
    rolling_years: int = 5
    min_observations: int = 750
    n_scenarios: int = 10000
    seed: int = 42

    def __post_init__(self):
        for name in ("rolling_years", "min_observations", "n_scenarios"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

def _day(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is not None or result != result.normalize():
        raise ValueError("Dates must be valid timezone-naive daily dates")
    return result


def validate_periods(origin, periods):
    origin = _day(origin)
    blocks = tuple(pd.DatetimeIndex(block) for block in periods)
    if not blocks or any(block.empty for block in blocks):
        raise ValueError("Supply at least one nonempty explicit future trading period")
    for block in blocks:
        if (block.hasnans or block.tz is not None or block.has_duplicates or
                not block.is_monotonic_increasing or not block.equals(block.normalize())):
            raise ValueError("Trading periods must contain increasing unique naive daily dates")
    all_dates = blocks[0].append(list(blocks[1:]))
    if (all_dates.has_duplicates or not all_dates.is_monotonic_increasing or all_dates[0] != origin):
        raise ValueError("Periods must be ordered, nonoverlapping, and start at the forecast origin")
    return blocks


def monthly_periods(origin, trading_calendar, horizon=3):
    """Partition an explicit known calendar into complete planned monthly blocks.

    The caller supplies the authoritative calendar, including the complete final
    planning month. This function does not infer exchange holidays from returns.
    """
    origin = _day(origin)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("Planning horizon must be a positive integer")
    dates = pd.DatetimeIndex(trading_calendar)
    if (dates.empty or dates.hasnans or dates.tz is not None or dates.has_duplicates or
            not dates.is_monotonic_increasing or not dates.equals(dates.normalize())):
        raise ValueError("Supply an increasing unique daily trading calendar")
    month = origin.to_period("M")
    blocks = tuple(dates[dates.to_period("M") == month+k] for k in range(horizon))
    if any(block.empty for block in blocks) or blocks[0][0] != origin:
        raise ValueError("Origin must start the first calendar month and every planned month must be supplied")
    return validate_periods(origin, blocks)


def history_before_origin(history, origin, config):
    """Filter chronology first; future return values are never used or validated."""
    origin = _day(origin)
    if not isinstance(history, pd.DataFrame) or set(history.columns) != set(ASSETS):
        raise ValueError("History must have exactly Equity, Bond and BTC columns")
    index = history.index
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.tz is not None or
            index.has_duplicates or not index.is_monotonic_increasing or not index.equals(index.normalize())):
        raise ValueError("History must have increasing unique naive daily dates")
    # Exclude the entire origin date: its return is not known when forecasting.
    window = history.loc[(index < origin) & (index >= origin-pd.DateOffset(years=config.rolling_years)), list(ASSETS)].copy()
    if len(window) < config.min_observations:
        raise ValueError(f"Need at least {config.min_observations} strictly prior fitting observations")
    values = window.to_numpy(float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Prior history must contain finite simple returns greater than -1")
    return window


def fit_garch(window_pct):
    """Fit one GARCH model per asset; return states and n x 3 innovations."""
    states = []
    standardized = []
    for asset in window_pct.columns:
        series = window_pct[asset]
        model = arch_model(series, mean="Constant", vol="GARCH", p=1, o=0,
                           q=1, dist=MARGINAL_DIST)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(disp="off", show_warning=False)
        if fit.convergence_flag != 0:
            warnings.warn(f"GARCH fit for {asset} at {series.index[-1].date()} "
                          f"did not converge (code={fit.convergence_flag})")
        par = fit.params
        z_raw = fit.std_resid.to_numpy(dtype=float)
        zm = float(np.mean(z_raw[np.isfinite(z_raw)]))
        zs = float(np.std(z_raw[np.isfinite(z_raw)], ddof=1))
        if zs < 1e-9 or not np.isfinite(zs):
            raise ValueError(f"Bad GARCH standardized residuals: {asset}")
        z = (z_raw-zm)/zs
        standardized.append(z)
        last_variance = float(fit.conditional_volatility.iloc[-1]**2)
        last_resid = float(fit.resid.iloc[-1])
        s = {
            "asset":asset, "mu":float(par["mu"]),
            "scenario_mu":float(series.mean()),  # same drift convention as old FHS
            "omega":float(par["omega"]),
            "alpha":float(par["alpha[1]"]), "beta":float(par["beta[1]"]),
            "zmean":zm, "zstd":zs,
            "next_var":max(MIN_VAR,float(par["omega"] + par["alpha[1]"]*last_resid**2
                                         + par["beta[1]"]*last_variance)),
            "vol_cap":(np.inf if VOL_CAP_MULTIPLE is None else
                       float((VOL_CAP_MULTIPLE*fit.conditional_volatility.max())**2)),
            "persistence":float(par["alpha[1]"]+par["beta[1]"]),
            "convergence":int(fit.convergence_flag)
        }
        states.append(s)
    Z = np.column_stack(standardized)
    if not np.isfinite(Z).all():
        raise ValueError("Nonfinite standardized residuals, inspect GARCH fits")
    return states, Z


def dcc_correlation(Q):
    d = np.sqrt(np.maximum(np.diagonal(Q, axis1=-2, axis2=-1), 1e-15))
    return Q/(d[..., :, None]*d[..., None, :])


def dcc_parameters(Z):
    """Gaussian DCC(1,1) QML fit to centered/scaled GARCH residuals.

    DCC equations: Q_t=(1-a-b)S + a z_{t-1}z_{t-1}' + b Q_{t-1};
                   R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2).
    For 3 assets use analytic inverse of 3x3 correlation matrix to speed QML.
    """
    S = np.cov(Z, rowvar=False, ddof=1)
    if S.shape != (3, 3):
        raise ValueError("This version explicitly implements three assets")
    if np.linalg.eigvalsh(S).min() < 1e-9:
        raise ValueError("Sample standardized-residual covariance not SPD")
    Z = np.asarray(Z, dtype=float)
    zz = Z[:-1]
    observations = Z[1:]
    S0 = np.array(S, dtype=float)
    def loss(ab):
        a, b = map(float, ab)
        if min(a,b) < 0 or a+b > DCC_MAX_PERSISTENCE:
            return 1e12
        Q = S0.copy()
        total = 0.0
        q_constant = (1-a-b)*S0
        for i, z in enumerate(observations):
            Q = q_constant + a*np.outer(zz[i], zz[i]) + b*Q
            R = dcc_correlation(Q)
            x,y,h = R[0,1],R[0,2],R[1,2]
            det = 1 + 2*x*y*h - x*x-y*y-h*h
            if det <= 1e-12 or not np.isfinite(det):
                return 1e12
            u,v,w = z
            numerator = ((1-h*h)*u*u + (1-y*y)*v*v + (1-x*x)*w*w
                         +2*(y*h-x)*u*v + 2*(x*h-y)*u*w + 2*(x*y-h)*v*w)
            total += np.log(det) + numerator/det
        return float(total / len(observations))

    cons = [{"type":"ineq", "fun":lambda ab: DCC_MAX_PERSISTENCE-ab[0]-ab[1]}]
    bounds = ((0,0.35), (0,DCC_MAX_PERSISTENCE))
    solutions = []
    for x0 in ((0.025,0.94), (0.08,0.85), (0.01,0.75)):
        result = minimize(loss, x0=np.array(x0), method="SLSQP", bounds=bounds,
                          constraints=cons,
                          options={"maxiter":DCC_MAX_ITER, "ftol":1e-8})
        if (result.success and np.isfinite(result.fun) and
                result.x.min() >= -1e-9 and result.x.sum() <= DCC_MAX_PERSISTENCE+1e-8):
            solutions.append(result)
            break
    if not solutions:
        raise RuntimeError("DCC optimization failed; inspect residual data and numerical settings")
    best = min(solutions,key=lambda r:r.fun)
    a,b = [float(x) for x in best.x]
    if a+b > 0.995:
        warnings.warn(f"DCC persistence near constrained boundary: a+b={a+b:.6f}")
    # R[k] is the forecast correlation for observation k, based on prior returns.
    Q = S0.copy()
    R_filtered = np.empty((len(Z),3,3))
    for k in range(len(Z)):
        R_filtered[k] = dcc_correlation(Q)
        Q = ((1-a-b)*S0 + a*np.outer(Z[k],Z[k]) + b*Q)
    # Q after the last observed shock is the FIRST out-of-sample Q forecast.
    return {"a":a,"b":b,"S":S0,"Q_start":Q,"R_training":R_filtered,
            "qll":float(best.fun), "message":str(best.message)}


def whitened_empirical_shocks(Z, S):
    """Preserve observed multivariate tail draws while removing linear covariance.

    Sample rows of e = Z @ inv(chol(S)).T, Cov(e)=I; during simulation
    set z_t=chol(R_t)@e_t, so Cov(z_t | history)=R_t.
    """
    L = np.linalg.cholesky(S)
    E = np.linalg.solve(L, Z.T).T
    return E


def ewma_bond(window_pct):
    """Bond EWMA conditional variance, filtered one-day-ahead (percent^2)."""
    x = window_pct["Bond"].to_numpy(float)
    mean = x.mean()
    eps = x-mean
    var = np.empty(len(x), dtype=float)
    var[0] = max(float(np.var(x[:min(len(x), 60)], ddof=1)), 1e-10)
    for t in range(1, len(x)):
        var[t] = EWMA_LAMBDA*var[t-1] + (1-EWMA_LAMBDA)*eps[t-1]**2
    raw = eps / np.sqrt(np.maximum(var,1e-10))
    z = (raw-raw.mean())/raw.std(ddof=1)
    next_v = EWMA_LAMBDA*var[-1] + (1-EWMA_LAMBDA)*eps[-1]**2
    cap = (VOL_CAP_MULTIPLE * np.sqrt(var.max()))**2 if VOL_CAP_MULTIPLE is not None else np.inf
    return dict(mu=float(mean), next_var=float(next_v), vol_cap=float(cap)),z


def simulate_periods(states, pool, indices, lengths, *, fit=None, save_daily=False):
    """Simulate all-GARCH FHS, or DCC-FHS when a fitted DCC state is supplied."""
    lengths = np.asarray(lengths)
    indices, pool = np.asarray(indices), np.asarray(pool, float)
    if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer) or
            not len(lengths) or (lengths < 1).any() or indices.ndim != 2 or
            not np.issubdtype(indices.dtype, np.integer) or indices.shape[0] < 1 or
            indices.shape[1] != int(lengths.sum()) or pool.ndim != 2 or pool.shape[1] != 3 or
            not np.isfinite(pool).all() or not len(pool) or indices.min() < 0 or indices.max() >= len(pool)):
        raise ValueError('Invalid all-GARCH empirical pool, indices or period lengths')
    if len(states) != 3:
        raise ValueError('Three marginal states are required')
    means = np.array([s['scenario_mu'] for s in states])
    initial = np.array([s['next_var'] for s in states])
    caps = np.array([s['vol_cap'] for s in states])
    omega, alpha, beta = [np.array([s[key] for s in states]) for key in ('omega', 'alpha', 'beta')]
    if (not np.isfinite(np.r_[means, initial, omega, alpha, beta]).all() or (initial <= 0).any()
            or np.isnan(caps).any() or (caps <= 0).any() or (np.r_[omega, alpha, beta] < 0).any()):
        raise ValueError('Invalid all-GARCH marginal states')
    n, days = indices.shape
    variance = np.broadcast_to(initial, (n, 3)).copy()
    if fit is not None:
        a, b, S = float(fit['a']), float(fit['b']), np.asarray(fit['S'], float)
        if min(a, b) < 0 or a+b >= 1:
            raise ValueError('Nonstationary DCC parameters')
        np.linalg.cholesky(S)
        np.linalg.cholesky(fit['Q_start'])
        q = np.broadcast_to(fit['Q_start'], (n, 3, 3)).copy()
    result = np.empty((n, len(lengths), 3))
    daily = np.empty((n, days, 3)) if save_daily else None
    wealth = np.ones((n, 3))
    boundaries = set(np.cumsum(lengths).tolist())
    position = clips = cap_hits = 0
    for day in range(days):
        innovation = pool[indices[:, day]]
        if fit is not None:
            diagonal = np.sqrt(np.diagonal(q, axis1=-2, axis2=-1))
            correlation = q/(diagonal[:, :, None]*diagonal[:, None, :])
            innovation = np.einsum('nij,nj->ni', np.linalg.cholesky(correlation), innovation)
        shocks = np.sqrt(variance)*innovation
        returns = (means+shocks)/100.
        clips += int((returns < -.99).sum())
        returns = np.maximum(returns, -.99)
        wealth *= 1+returns
        if daily is not None:
            daily[:, day] = returns
        variance = omega+alpha*shocks**2+beta*variance
        cap_hits += int((variance > caps).sum())
        variance = np.maximum(np.minimum(variance, caps), 1e-10)
        if fit is not None:
            q = (1-a-b)*S[None]+a*innovation[:, :, None]*innovation[:, None, :]+b*q
        if day+1 in boundaries:
            result[:, position] = wealth-1
            position += 1
            wealth.fill(1.)
    if not np.isfinite(result).all() or (result <= -1).any():
        raise ValueError('Invalid compounded scenario returns')
    return result, daily, {'daily_clips': clips, 'variance_cap_share': cap_hits/(n*days*3)}


@dataclass(frozen=True)
class ScenarioForecast:
    origin: pd.Timestamp
    periods: tuple
    returns: dict
    holding_days: dict
    metadata: dict
    assets: tuple = ASSETS


def original_indices(pool_size, lengths, origin_index, *, n_scenarios=10000, seed=42):
    if (isinstance(origin_index, bool) or not isinstance(origin_index, (int, np.integer)) or origin_index < 0):
        raise ValueError('Origin index must be a nonnegative chronological integer')
    rng = np.random.default_rng(seed+1_000_003*origin_index)
    # Drawing each period separately preserves the exact old first-month draws.
    return np.concatenate([rng.integers(pool_size, size=(n_scenarios, length), dtype=np.int32)
                           for length in lengths], axis=1)


def simulate_models(states, z, ewma, mixed_z, dcc, hybrid_dcc, indices, lengths):
    """Original volatility/dependence rules, with states carried across months."""
    hybrid_states = copy.deepcopy(states)
    hybrid_states[1].update(scenario_mu=ewma['mu'], next_var=ewma['next_var'], vol_cap=ewma['vol_cap'],
                            omega=0., alpha=1-EWMA_LAMBDA, beta=EWMA_LAMBDA)
    innovations = whitened_empirical_shocks(z, dcc['S'])
    mixed_innovations = whitened_empirical_shocks(mixed_z, hybrid_dcc['S'])
    specifications = (
        ('FHS GARCH', states, z, None), ('FHS Hybrid', hybrid_states, mixed_z, None),
        ('DCC GARCH', states, innovations, dcc), ('DCC Hybrid', hybrid_states, mixed_innovations, hybrid_dcc))
    cubes, diagnostics = {}, {}
    for name, marginal, pool, fit in specifications:
        result, _, diagnostic = simulate_periods(marginal, pool, indices, lengths, fit=fit)
        cubes[name] = result
        diagnostics[name] = diagnostic
    return cubes, diagnostics


def generate_scenarios(history, origin, periods, origin_index, *, n_scenarios=10000, seed=42):
    """Fit strictly prior data; supply each model's unshifted native scenarios."""
    periods = validate_periods(origin, periods)
    config = ForecastConfig(n_scenarios=n_scenarios, seed=seed)
    window = history_before_origin(history, origin, config)
    states, z = fit_garch(window*100.)
    dcc = dcc_parameters(z)
    ewma, bond_z = ewma_bond(window*100.)
    mixed_z = z.copy()
    mixed_z[:, 1] = bond_z
    hybrid_dcc = dcc_parameters(mixed_z)
    lengths = [len(period) for period in periods]
    indices = original_indices(len(z), lengths, origin_index, n_scenarios=n_scenarios, seed=seed)
    cubes, diagnostics = simulate_models(states, z, ewma, mixed_z, dcc, hybrid_dcc, indices, lengths)
    # The original monthly cache stores float32. Retain that precision convention
    # for all stages, then expose float64 arrays to the optimizer as before.
    cubes = {name: values.astype(np.float32).astype(float) for name, values in cubes.items()}
    model_days = {name: lengths.copy() for name in MODELS}
    metadata = {'origin': str(pd.Timestamp(origin).date()), 'history_start': str(window.index[0].date()),
        'history_end': str(window.index[-1].date()), 'history_observations': len(window),
        'history_sha256': hashlib.sha256(pd.util.hash_pandas_object(window, index=True).values.tobytes()).hexdigest(),
        'origin_index': int(origin_index), 'seed': int(seed+1_000_003*origin_index),
        'draw_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
        'period_dates': [[str(day.date()) for day in period] for period in periods],
        'model_holding_days': model_days, 'bond_drift_decimal': float(ewma['mu']/100.),
        'bond_drift_convention': 'Original prior-five-year arithmetic daily mean; no prior252 override',
        'marginal_states': states, 'ewma': ewma,
        'dcc': {key: dcc[key] for key in ('a', 'b', 'S', 'Q_start')},
        'hybrid_dcc': {key: hybrid_dcc[key] for key in ('a', 'b', 'S', 'Q_start')},
        'simulation_diagnostics': diagnostics,
        'mean_convention': 'Native sample mean of each model and incremental stage; no external drift-based optimizer means',
        'precision': 'Original float32 monthly cache precision, converted to float64 for optimization'}
    return ScenarioForecast(pd.Timestamp(origin), periods, cubes, model_days, metadata)
