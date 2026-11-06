"""Three aligned hybrid forecasts with incremental multi-period scenarios.

All fitting precedes the origin. Future periods come from an explicitly supplied
trading calendar, never from future returns. Marginal and DCC states continue
across periods; model parameters and forecast drifts stay fixed within a plan.
There is no automatic dataset loading, backtest, or model selection.
"""
# FROZEN RESEARCH SPECIFICATION
# Primary scenario generator: bundle.returns["HybridDCC"]. HybridFHS remains the
# simple benchmark; MomentControlledWeightedHybridDCC is the documented challenger.
# All three stay available here so earlier comparisons remain reproducible.
#
# - Rolling five-calendar-year history, using only dates strictly before origin.
# - Equity/BTC GARCH(1,1); Bond EWMA with decay 0.94. Equity/BTC simulation drifts
#   are prior-five-year arithmetic daily means; Bond drift uses exactly the last
#   252 prior aligned observations. Historical Bond filtering stays unchanged.
# - HybridDCC uniformly resamples complete empirical whitened innovation rows,
#   then applies path-specific DCC correlations. Innovation draws are empirical;
#   the Gaussian DCC estimation criterion does not impose Gaussian scenario draws.
# - Research defaults: 10,000 paths, seed 42, and three calendar-month periods.
#   Daily returns compound into incremental monthly simple returns. Volatility
#   and correlation states continue across months; fitted parameters/drifts stay
#   fixed until the next actual forecast origin.
# - Future trading dates must be supplied from a calendar known at the origin.
#   Monthly refitting/reoptimization and first-month-only execution belong to the
#   calling backtest/optimizer. This module supplies forecasts, not trading rules.
# - CVaR confidence, lambda, BTC caps, return floors, costs and the choice of
#   summed monthly versus terminal-wealth CVaR are separate optimizer settings.
#
# "Frozen" documents the selected research specification. The reusable interface
# still supports explicit overrides and H=1 controls; defaults are not hard locks.
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_MODELLING_ROOT = Path(__file__).resolve().parents[1]
if str(_MODELLING_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODELLING_ROOT))
from project_paths import source_path

import numpy as np
import pandas as pd

ASSETS = ("Equity", "Bond", "BTC")
MODELS = ("HybridFHS", "HybridDCC", "MomentControlledWeightedHybridDCC")
MOMENT_CONTROL = "weighted mean and full covariance matched to uniform reference pool"


@dataclass(frozen=True)
class ForecastConfig:
    # Selected shared forecast settings; min_observations is a fitting safeguard.
    rolling_years: int = 5
    min_observations: int = 750
    bond_mean_observations: int = 252
    # Applies only to the weighted challenger, not the primary uniform HybridDCC.
    weight_half_life: float = 126.
    n_scenarios: int = 10_000
    seed: int = 42
    save_daily_returns: bool = False

    def __post_init__(self):
        for name in ("rolling_years", "min_observations", "bond_mean_observations", "n_scenarios"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.bond_mean_observations != 252:
            raise ValueError("This three-model specification fixes Bond drift at 252 prior observations")
        if self.min_observations < self.bond_mean_observations:
            raise ValueError("The fitting minimum must cover the Bond mean window")
        if not np.isfinite(self.weight_half_life) or self.weight_half_life <= 0:
            raise ValueError("Weight half-life must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "weight_half_life", float(self.weight_half_life))
        if not isinstance(self.save_daily_returns, bool):
            raise ValueError("save_daily_returns must be boolean")


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


def age_weights(size, half_life=126.):
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Innovation pool size must be a positive integer")
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("Half-life must be finite and positive")
    values = np.exp2(-np.arange(size-1, -1, -1, dtype=float)/half_life)
    return values/values.sum()


def probability_moments(pool, probabilities):
    pool, probabilities = np.asarray(pool, float), np.asarray(probabilities, float)
    if (pool.ndim != 2 or pool.shape[1] != 3 or len(pool) < 4 or
            probabilities.shape != (len(pool),) or not np.isfinite(pool).all() or
            not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("Require a finite three-asset innovation pool and normalized probabilities")
    mean = probabilities @ pool
    centered = pool-mean
    covariance = centered.T @ (probabilities[:, None]*centered)
    return mean, (covariance+covariance.T)/2


def _spd_power(matrix, exponent):
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if eigenvalues.min() <= 1e-12*max(1., eigenvalues.max()):
        raise ValueError("Moment matching requires a well-conditioned positive-definite covariance; no repair is applied")
    return (vectors*eigenvalues**exponent) @ vectors.T


def match_weighted_moments(pool, probabilities):
    """Match weighted first/second moments to the uniform reference distribution.

    For row vectors: e* = (e-m_w) C_w^(-1/2) C_ref^(1/2) + m_ref.
    Population/probability covariances are used consistently, not mixed ddofs.
    This full-covariance definition supersedes the earlier variance-only idea.
    """
    pool = np.asarray(pool, float)
    weighted_mean, weighted_covariance = probability_moments(pool, probabilities)
    reference_mean, reference_covariance = probability_moments(pool, np.full(len(pool), 1/len(pool)))
    transform = _spd_power(weighted_covariance, -.5) @ _spd_power(reference_covariance, .5)
    adjusted = (pool-weighted_mean) @ transform + reference_mean
    mean, covariance = probability_moments(adjusted, probabilities)
    mean_error = float(np.max(np.abs(mean-reference_mean)))
    covariance_error = float(np.max(np.abs(covariance-reference_covariance)))
    if mean_error > 1e-9 or covariance_error > 1e-9*max(1., np.abs(reference_covariance).max()):
        raise AssertionError("Weighted innovation moment control failed")
    return adjusted, {
        "definition": MOMENT_CONTROL, "reference_mean": reference_mean.tolist(),
        "reference_covariance": reference_covariance.tolist(),
        "native_weighted_mean": weighted_mean.tolist(), "native_weighted_covariance": weighted_covariance.tolist(),
        "controlled_weighted_mean": mean.tolist(), "controlled_weighted_covariance": covariance.tolist(),
        "transform": transform.tolist(), "max_mean_error": mean_error,
        "max_covariance_error": covariance_error, "effective_pool_size": float(1/np.sum(np.asarray(probabilities)**2))}


@lru_cache(maxsize=1)
def _generator():
    path = source_path("DCC-GARCH EWMA GARCH.py")
    spec = importlib.util.spec_from_file_location("multiperiod_shared_hybrid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FittedHybrid:
    origin: pd.Timestamp
    states: tuple
    standardized_residuals: np.ndarray
    innovations: np.ndarray
    dcc_fit: dict
    ewma: dict
    config: ForecastConfig
    metadata: dict


def fit_models(history, origin, config=None):
    """Fit shared marginal/DCC states once for all three models at one origin."""
    config = config or ForecastConfig()
    origin = _day(origin)
    window = history_before_origin(history, origin, config)
    generator = _generator()
    states, z = generator.dcc.fit_garch(window*100.)
    ewma, bond_z = generator.ewma_bond(window*100.)
    ewma = dict(ewma)
    filter_mean = float(ewma["mu"])
    # Align simulation drift across all three models. This replaces only Bond's
    # forecast location; bond_z and the EWMA variance retain the original filter.
    bond_daily_mean = float(window.Bond.tail(252).mean()*100.)
    ewma["mu"] = bond_daily_mean
    z = np.asarray(z, float).copy()
    z[:, 1] = bond_z
    fit = generator.dcc.dcc_parameters(z)
    # Whiten the hybrid residual rows using the fitted static target S, then
    # recolor sampled vectors with evolving path-specific R during simulation.
    innovations = generator.dcc.whitened_empirical_shocks(z, fit["S"])
    history_hash = hashlib.sha256(pd.util.hash_pandas_object(window, index=True).values.tobytes()).hexdigest()
    metadata = {"origin": str(origin.date()), "history_start": str(window.index[0].date()),
        "history_end": str(window.index[-1].date()), "history_observations": len(window),
        "history_sha256": history_hash, "bond_daily_mean_decimal": bond_daily_mean/100.,
        "bond_mean_observations": 252, "bond_mean_last_date": str(window.index[-1].date()),
        "bond_filter_daily_mean_decimal": filter_mean/100.,
        "mean_convention": "Bond drift only uses prior252; original five-year residual/variance filtering is retained",
        "equity_btc_mean_convention": "Prior five-year arithmetic daily mean",
        "ewma_lambda": float(generator.EWMA_LAMBDA), "dcc_a": float(fit["a"]), "dcc_b": float(fit["b"]),
        "fit_messages": {"dcc": str(fit.get("message", "")),
                         "garch_convergence": [int(s.get("convergence", 0)) for s in states]},
        "marginal_states": copy.deepcopy(states), "ewma_state": copy.deepcopy(ewma)}
    return FittedHybrid(origin, tuple(copy.deepcopy(states)), z, innovations, fit, ewma, config, metadata)


def _indices(size, lengths, config):
    # Drawing each full period separately preserves H=1/H>1 first-period paths.
    rng, jitter_rng = np.random.default_rng(config.seed), np.random.default_rng(config.seed+9_000_001)
    weights = age_weights(size, config.weight_half_life)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    uniform, weighted = [], []
    for length in lengths:
        # One row index selects the complete three-asset shock vector. Uniform
        # draws are with replacement across days/paths, rather than block draws.
        indices = rng.integers(size, size=(config.n_scenarios, length), dtype=np.int32)
        uniforms = (indices+jitter_rng.random(indices.shape))/size
        uniform.append(indices)
        weighted.append(np.searchsorted(cdf, uniforms, side="right").astype(np.int32))
    return np.concatenate(uniform, axis=1), np.concatenate(weighted, axis=1), weights


def simulate_periods(fitted, pool, indices, lengths, *, dcc=True, save_daily=False):
    """Simulate coherent paths; reset only period P&L, never variance/DCC states."""
    indices, pool = np.asarray(indices), np.asarray(pool, float)
    lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or not len(lengths) or not np.issubdtype(lengths.dtype, np.integer) or
            (lengths < 1).any() or indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer) or
            indices.shape[0] < 1 or indices.shape[1] != int(lengths.sum()) or
            pool.ndim != 2 or pool.shape[1] != 3 or not len(pool) or not np.isfinite(pool).all() or
            indices.min() < 0 or indices.max() >= len(pool)):
        raise ValueError("Invalid empirical pool, sampled indices, or period lengths")
    states, ewma, fit = fitted.states, fitted.ewma, fitted.dcc_fit
    if len(states) != 3:
        raise ValueError("Require three marginal states")
    means = np.array([s["scenario_mu"] for s in states], float)
    initial = np.array([s["next_var"] for s in states], float)
    caps = np.array([s["vol_cap"] for s in states], float)
    omega = np.array([s["omega"] for s in states], float)
    alpha = np.array([s["alpha"] for s in states], float)
    beta = np.array([s["beta"] for s in states], float)
    initial[1], caps[1], means[1] = ewma["next_var"], ewma["vol_cap"], ewma["mu"]
    # Bond replaces the generic marginal recursion: v_next = .06*shock^2 + .94*v.
    omega[1], alpha[1], beta[1] = 0., 1-.94, .94
    if (not np.isfinite(np.r_[means, initial, omega, alpha, beta]).all() or
            (initial <= 0).any() or np.isnan(caps).any() or (caps <= 0).any() or
            (np.r_[omega, alpha, beta] < 0).any()):
        raise ValueError("Invalid shared marginal state")
    n, days = indices.shape
    variance = np.broadcast_to(initial, (n, 3)).copy()
    q = np.broadcast_to(fit["Q_start"], (n, 3, 3)).astype(float).copy() if dcc else None
    if dcc:
        covariance = np.asarray(fit["S"], float)
        a, b = float(fit["a"]), float(fit["b"])
        if (covariance.shape != (3, 3) or not np.isfinite(covariance).all() or
                not np.allclose(covariance, covariance.T) or min(a, b) < 0 or a+b >= 1 or
                not np.isfinite(q).all() or not np.allclose(q, q.transpose(0, 2, 1))):
            raise ValueError("Invalid DCC state")
        np.linalg.cholesky(covariance)
        np.linalg.cholesky(q)
    result = np.empty((n, len(lengths), 3))
    daily = np.empty((n, days, 3)) if save_daily else None
    period_wealth, position, clips, cap_hits = np.ones((n, 3)), 0, 0, 0
    boundaries = set(np.cumsum(lengths).tolist())
    for day in range(days):
        innovations = pool[indices[:, day]]
        if dcc:
            diagonal = np.sqrt(np.diagonal(q, axis1=-2, axis2=-1))
            correlation = q/(diagonal[:, :, None]*diagonal[:, None, :])
            innovations = np.einsum("nij,nj->ni", np.linalg.cholesky(correlation), innovations)
        shocks = np.sqrt(variance)*innovations
        returns = (means+shocks)/100.
        # Preserve the existing return-support and variance safeguards; their
        # activation counts are returned for assessment, not silently omitted.
        clips += int((returns < -.99).sum())
        returns = np.maximum(returns, -.99)
        if daily is not None:
            daily[:, day] = returns
        period_wealth *= 1+returns
        variance = omega+alpha*shocks**2+beta*variance
        cap_hits += int((variance > caps).sum())
        variance = np.maximum(np.minimum(variance, caps), 1e-10)
        if dcc:
            q = (1-a-b)*covariance[None, :, :]+a*innovations[:, :, None]*innovations[:, None, :]+b*q
        if day+1 in boundaries:
            result[:, position] = period_wealth-1
            position += 1
            # Reset only the monthly return accumulator. Carry variance and Q
            # forward so each row remains one coherent multi-month scenario.
            period_wealth.fill(1.)
    if not np.isfinite(result).all() or (result <= -1).any():
        raise ValueError("Compounded scenario paths overflowed or lost valid simple-return support")
    return result, daily, {"daily_clips": clips, "variance_cap_share": cap_hits/(n*days*3)}


@dataclass(frozen=True)
class ForecastBundle:
    origin: pd.Timestamp
    periods: tuple
    returns: dict
    expected_returns: np.ndarray
    metadata: dict
    daily_returns: dict | None = None
    assets: tuple = ASSETS

    def __post_init__(self):
        periods = validate_periods(self.origin, self.periods)
        if tuple(self.assets) != ASSETS or set(self.returns) != set(MODELS):
            raise ValueError("Forecast bundle must contain the three named models in Equity/Bond/BTC order")
        arrays = {}
        shape = None
        for model, values in self.returns.items():
            values = np.array(values, dtype=float, copy=True)
            if (values.ndim != 3 or min(values.shape) < 1 or values.shape[1:] != (len(periods), 3) or
                    not np.isfinite(values).all() or (values <= -1).any()):
                raise ValueError(f"Invalid incremental scenario cube: {model}")
            if shape is not None and values.shape != shape:
                raise ValueError("All three models must have identical scenario dimensions")
            shape = values.shape
            values.setflags(write=False)
            arrays[model] = values
        expected = np.array(self.expected_returns, float, copy=True)
        if expected.shape != (len(periods), 3) or not np.isfinite(expected).all() or (expected <= -1).any():
            raise ValueError("Common expected returns must have shape (period, asset) and valid support")
        expected.setflags(write=False)
        daily_arrays = None
        if self.daily_returns is not None:
            daily_arrays = {}
            if set(self.daily_returns) != set(MODELS):
                raise ValueError("Daily paths must contain all three models")
            for model, values in self.daily_returns.items():
                values = np.array(values, float, copy=True)
                if (values.shape != (shape[0], sum(map(len, periods)), 3) or
                        not np.isfinite(values).all() or (values <= -1).any()):
                    raise ValueError("Daily paths are invalid or misaligned")
                offset = 0
                for k, period in enumerate(periods):
                    actual = np.prod(1+values[:, offset:offset+len(period)], axis=1)-1
                    if not np.allclose(actual, arrays[model][:, k], atol=1e-12, rtol=1e-10):
                        raise ValueError("Daily and period scenarios do not compound consistently")
                    offset += len(period)
                values.setflags(write=False)
                daily_arrays[model] = values
        object.__setattr__(self, "origin", _day(self.origin))
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "returns", arrays)
        object.__setattr__(self, "expected_returns", expected)
        object.__setattr__(self, "daily_returns", daily_arrays)
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    def save(self, path):
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Forecast output must end in .npz")
        arrays = {**self.returns, "expected_returns": self.expected_returns}
        if self.daily_returns is not None:
            arrays.update({f"daily__{model}": values for model, values in self.daily_returns.items()})
        manifest = {"schema": 1, "origin": str(self.origin.date()), "assets": list(self.assets),
            "periods": [[str(day.date()) for day in period] for period in self.periods],
            "metadata": self.metadata,
            "array_sha256": {name: hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()
                             for name, value in arrays.items()}}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, manifest=np.array(json.dumps(manifest, allow_nan=False)), **arrays)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            manifest = json.loads(str(stored["manifest"]))
            if manifest["schema"] != 1 or tuple(manifest["assets"]) != ASSETS:
                raise ValueError("Unsupported forecast schema/assets")
            arrays = {name: stored[name].astype(float) for name in manifest["array_sha256"]}
            for name, checksum in manifest["array_sha256"].items():
                if hashlib.sha256(arrays[name].tobytes()).hexdigest() != checksum:
                    raise ValueError(f"Forecast array checksum failed: {name}")
            daily_names = [f"daily__{model}" for model in MODELS]
            if any(name in arrays for name in daily_names) and not all(name in arrays for name in daily_names):
                raise ValueError("Forecast archive contains incomplete daily paths")
            daily = {model: arrays[f"daily__{model}"] for model in MODELS} if all(name in arrays for name in daily_names) else None
            return cls(pd.Timestamp(manifest["origin"]), tuple(pd.DatetimeIndex(p) for p in manifest["periods"]),
                       {model: arrays[model] for model in MODELS}, arrays["expected_returns"],
                       manifest["metadata"], daily)


def forecast_from_fit(fitted, periods):
    periods = validate_periods(fitted.origin, periods)
    lengths = [len(period) for period in periods]
    uniform, weighted, probabilities = _indices(len(fitted.innovations), lengths, fitted.config)
    # Retained challenger: 126-observation half-life by default, with weighted
    # mean AND full covariance matched to the uniform reference innovation pool.
    # HybridDCC itself uses the original unweighted pool and uniform indices.
    controlled, moments = match_weighted_moments(fitted.innovations, probabilities)
    pools = (fitted.standardized_residuals, fitted.innovations, controlled)
    result, daily, diagnostics = {}, {}, {}
    for model, pool, indices in zip(MODELS, pools, (uniform, uniform, weighted)):
        values, paths, audit = simulate_periods(fitted, pool, indices, lengths,
            dcc=model != "HybridFHS", save_daily=fitted.config.save_daily_returns)
        result[model], diagnostics[model] = values, audit
        if paths is not None:
            daily[model] = paths
    mean = np.array([state["scenario_mu"] for state in fitted.states], float)/100.
    mean[1] = fitted.ewma["mu"]/100.
    # Shared optimizer return forecasts compound the fixed daily drift over each
    # stage's trading days. Native scenario sample means remain separate; no
    # post-hoc shifting/rescaling forces the three scenario cubes to agree.
    expected = np.expm1(np.asarray(lengths)[:, None]*np.log1p(mean))
    # Provenance hashes include comments as well as code. Documentation edits
    # change future fingerprints; historical saved manifests retain their hashes.
    sources = [Path(__file__), source_path("DCC-GARCH Gaussian.py"), source_path("DCC-GARCH EWMA GARCH.py")]
    metadata = {**copy.deepcopy(fitted.metadata), "config": asdict(fitted.config),
        "models": list(MODELS), "assets": list(ASSETS), "period_days": lengths,
        "weighted_moment_control": moments, "simulation_diagnostics": diagnostics,
        "common_daily_drift_decimal": mean.tolist(),
        "expected_return_convention": "Common fixed daily drifts compounded by stage days; Monte Carlo means and clipping effects are reported separately",
        "scenario_means": {model: values.mean(axis=0).tolist() for model, values in result.items()},
        "calendar_convention": "Explicit caller-supplied known trading dates; no future returns used",
        "within_plan_fitting": "Fixed current fit and drift; simulated variance/DCC states continue across all stages",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "uniform_draw_sha256": hashlib.sha256(uniform.tobytes()).hexdigest(),
        "weighted_draw_sha256": hashlib.sha256(weighted.tobytes()).hexdigest()}
    metadata["forecast_fingerprint"] = hashlib.sha256(json.dumps(
        {"metadata": metadata, "periods": [[str(d.date()) for d in p] for p in periods]},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    return ForecastBundle(fitted.origin, periods, result, expected, metadata, daily or None)


def forecast_models(history, origin, periods, config=None):
    return forecast_from_fit(fit_models(history, origin, config), periods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", nargs="+", required=True, help="Explicit prior-return CSV files; no datasets are loaded automatically")
    parser.add_argument("--origin", required=True, help="Current rebalance date YYYY-MM-DD")
    parser.add_argument("--calendar", required=True, help="CSV whose first column is the known trading calendar, including full planning months")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--scenarios", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    history = pd.concat([pd.read_csv(path, index_col=0, parse_dates=True) for path in args.history]).sort_index()
    calendar = pd.to_datetime(pd.read_csv(args.calendar).iloc[:, 0], errors="raise")
    periods = monthly_periods(args.origin, calendar, args.horizon)
    bundle = forecast_models(history, args.origin, periods,
        ForecastConfig(n_scenarios=args.scenarios, seed=args.seed))
    bundle.save(args.output)
    print(f"Saved {len(MODELS)} forecast cubes of shape {next(iter(bundle.returns.values())).shape} to {args.output}")


if __name__ == "__main__":
    main()
