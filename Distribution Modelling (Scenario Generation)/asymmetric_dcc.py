"""Scalar asymmetric DCC, retaining the existing DCC likelihood/filter timing.

Q_next = (1-a-b) S - g N + a z z' + b Q + g n n', n = min(z, 0).
The coefficients a, b, and g multiply the outer products directly; they are
not the square-root diagonal-matrix parameters used in the general model.
S uses the existing project's ddof=1 covariance convention; N is mean(n n').

References: Cappiello, Engle and Sheppard, *Asymmetric Dynamics in the
Correlations of Global Equity and Bond Returns*, scalar restriction of eq. 3:
https://web-static.stern.nyu.edu/rengle/ECB10.06.pdf
The sufficient positive-definite-intercept restriction a+b+delta*g < 1 is
also given explicitly in ECB Working Paper 1546, equations 20--24:
https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1546.pdf

No eigenvalue clipping, diagonal floors, or matrix repairs are applied.
"""
from __future__ import annotations

# Support direct execution after organizing the source folders.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_MODELLING_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents
                      if (p / "project_paths.py").is_file())
if str(_MODELLING_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_MODELLING_ROOT))
from project_paths import (PROJECT_ROOT, MODELLING_ROOT, source_path,
                           cache_source_bytes, cache_source_name)


import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

MAX_PERSISTENCE = 0.998
MAX_ITER = 350
_PENALTY = 1e12


def _check_matrix(q, name="Q"):
    q = np.asarray(q, dtype=float)
    if q.shape[-2:] != (3, 3) or not np.isfinite(q).all():
        raise ValueError(f"{name} must contain finite 3 x 3 matrices")
    if not np.allclose(q, np.swapaxes(q, -1, -2), rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(q)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite; no repair is applied") from exc
    return q


def correlation(q):
    """Normalize one Q or a batch of Q matrices; reject invalid matrices."""
    q = _check_matrix(q)
    scale = np.sqrt(np.diagonal(q, axis1=-2, axis2=-1))
    return q / (scale[..., :, None] * scale[..., None, :])


def next_q(q, z, fit):
    """Update Q after a shock, with broadcasting over scenario paths."""
    q = _check_matrix(q)
    z = np.asarray(z, dtype=float)
    if z.shape[-1:] != (3,) or not np.isfinite(z).all():
        raise ValueError("z must contain finite three-asset shock vectors")
    a, b, g = (float(fit[key]) for key in ("a", "b", "g"))
    if not np.isfinite([a, b, g]).all() or min(a, b, g) < 0:
        raise ValueError("ADCC coefficients must be finite and nonnegative")
    if "delta" in fit and a + b + float(fit["delta"]) * g > MAX_PERSISTENCE:
        raise ValueError("ADCC persistence exceeds the sufficient SPD bound")
    constant = _check_matrix(fit["C"], "ADCC intercept")
    n = np.minimum(z, 0.)
    return (constant + a * z[..., :, None] * z[..., None, :] + b * q
            + g * n[..., :, None] * n[..., None, :])


def _moments(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 3 or len(z) < 4 or not np.isfinite(z).all():
        raise ValueError("ADCC requires at least four finite three-asset observations")
    s = _check_matrix(np.cov(z, rowvar=False, ddof=1), "Sample covariance")
    n = np.minimum(z, 0.)
    negative = np.einsum("ti,tj->ij", n, n) / len(n)
    values, vectors = np.linalg.eigh(s)
    inverse_root = (vectors * (1 / np.sqrt(values))) @ vectors.T
    delta = float(np.linalg.eigvalsh(inverse_root @ negative @ inverse_root)[-1])
    return z, s, negative, delta


def _intercept(parameters, s, negative, delta):
    a, b, g = np.asarray(parameters, dtype=float)
    if (not np.isfinite([a, b, g]).all() or min(a, b, g) < 0
            or a + b + delta * g > MAX_PERSISTENCE):
        return None
    constant = (1 - a - b) * s - g * negative
    if np.linalg.eigvalsh(constant)[0] <= 0:
        return None
    return constant


def _q_after_shocks(outer, negative_outer, s, constant, parameters):
    """All Q updates via an exact linear filter, including the last shock."""
    a, b, g = parameters
    forcing = constant + a * outer + g * negative_outer
    after, _ = lfilter([1.], [1., -b], forcing, axis=0,
                       zi=(b * s)[None, :, :])
    return after


def _loss_factory(z, s, negative, delta):
    outer = z[:, :, None] * z[:, None, :]
    n = np.minimum(z, 0.)
    negative_outer = n[:, :, None] * n[:, None, :]
    u, v, w = z[1:].T

    def loss(parameters):
        constant = _intercept(parameters, s, negative, delta)
        if constant is None:
            return _PENALTY
        q = _q_after_shocks(outer[:-1], negative_outer[:-1], s, constant, parameters)
        diagonal = np.diagonal(q, axis1=-2, axis2=-1)
        if not np.isfinite(q).all() or np.any(diagonal <= 0):
            return _PENALTY
        scale = np.sqrt(diagonal)
        x = q[:, 0, 1] / (scale[:, 0] * scale[:, 1])
        y = q[:, 0, 2] / (scale[:, 0] * scale[:, 2])
        h = q[:, 1, 2] / (scale[:, 1] * scale[:, 2])
        det = 1 + 2 * x * y * h - x * x - y * y - h * h
        if not np.isfinite(det).all() or np.any(det <= 1e-12):
            return _PENALTY
        numerator = ((1 - h * h) * u * u + (1 - y * y) * v * v
                     + (1 - x * x) * w * w + 2 * (y * h - x) * u * v
                     + 2 * (x * h - y) * u * w + 2 * (x * y - h) * v * w)
        return float(np.mean(np.log(det) + numerator / det))

    return loss, outer, negative_outer


def _optimizer_record(result, start, stage):
    return {"stage": stage, "start": [float(x) for x in start],
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "iterations": int(getattr(result, "nit", 0)),
            "objective": float(result.fun), "parameters": [float(x) for x in result.x]}


def fit_adcc(z, dcc_fit=None):
    """Fit ADCC QML, keeping the exact g=0 DCC candidate as a safeguard.

    The likelihood evaluates z[1:] after updating from the preceding shock,
    just as dcc_parameters does. R_training[0] instead uses S, and Q_start
    includes the final observed shock for the first out-of-sample forecast.
    Successful feasible optimizer solutions compete against the exact nested
    candidate. Failed solutions are never reported as successful fits.
    """
    z, s, negative, delta = _moments(z)
    loss, outer, negative_outer = _loss_factory(z, s, negative, delta)
    details = []
    if dcc_fit is None:
        candidates = []
        for start in ((.025, .94), (.08, .85), (.01, .75)):
            result = minimize(lambda ab: loss([ab[0], ab[1], 0.]), start,
                              method="SLSQP", bounds=((0., .35), (0., MAX_PERSISTENCE)),
                              constraints=[{"type": "ineq", "fun": lambda ab:
                                            MAX_PERSISTENCE - ab[0] - ab[1]}],
                              options={"maxiter": MAX_ITER, "ftol": 1e-9})
            details.append(_optimizer_record(result, start, "nested_dcc"))
            parameters = np.r_[result.x, 0.]
            value = loss(parameters)
            if result.success and value < _PENALTY:
                candidates.append((value, parameters))
        if not candidates:
            raise RuntimeError("Nested DCC optimization failed; ADCC has no valid fallback")
        nested_qll, nested = min(candidates, key=lambda pair: pair[0])
    else:
        nested = np.array([float(dcc_fit["a"]), float(dcc_fit["b"]), 0.])
        nested_qll = loss(nested)
        if nested_qll >= _PENALTY or not np.isfinite(nested_qll):
            raise ValueError("Supplied nested DCC candidate is infeasible for these residuals")
        if "S" in dcc_fit and not np.allclose(dcc_fit["S"], s, rtol=1e-8, atol=1e-10):
            raise ValueError("Supplied DCC fit must use the same residual sample covariance")

    # Interior starts keep the asymmetry optimizer away from the g=0 boundary.
    # Shrink a,b first when the nested fit is already close to the constraint.
    shrunk = nested.copy()
    shrunk[:2] *= .94
    slack = MAX_PERSISTENCE - shrunk[:2].sum()
    asymmetric_start = shrunk.copy()
    asymmetric_start[2] = min(.03, .35 * slack / delta) if delta > 0 else .03
    starts = [asymmetric_start, np.array([.04, .85, .04 / max(delta, 1.)]),
              np.array([.015, .65, .12 / max(delta, 1.)])]
    optimized = []
    for start in starts:
        result = minimize(loss, start, method="SLSQP",
                          bounds=((0., .35), (0., MAX_PERSISTENCE),
                                  (0., MAX_PERSISTENCE / delta if delta > 0 else 1.)),
                          constraints=[{"type": "ineq", "fun": lambda p:
                                        MAX_PERSISTENCE - p[0] - p[1] - delta * p[2]}],
                          options={"maxiter": MAX_ITER, "ftol": 1e-9})
        details.append(_optimizer_record(result, start, "adcc"))
        value = loss(result.x)
        if result.success and np.isfinite(value) and value < _PENALTY:
            optimized.append((value, np.array(result.x, dtype=float), result))

    best_qll, best_parameters = nested_qll, nested.copy()
    selected_result = None
    if optimized:
        value, parameters, result = min(optimized, key=lambda candidate: candidate[0])
        if value <= nested_qll:
            best_qll, best_parameters, selected_result = value, parameters, result
            status = "optimized_adcc" if parameters[2] > 1e-8 else "optimized_g_zero"
        else:
            status = "nested_retained_no_improvement"
    else:
        status = "nested_fallback_all_adcc_optimizations_failed"

    a, b, g = map(float, best_parameters)
    constant = _intercept(best_parameters, s, negative, delta)
    if constant is None:
        raise RuntimeError("Selected ADCC parameters have an invalid intercept")
    after = _q_after_shocks(outer, negative_outer, s, constant, best_parameters)
    before = np.concatenate((s[None, :, :], after[:-1]), axis=0)
    persistence = a + b + delta * g
    minimum_eigenvalue = float(np.linalg.eigvalsh(constant)[0])
    return {"a": a, "b": b, "g": g, "S": s, "N": negative, "C": constant,
            "Q_start": after[-1].copy(), "R_training": correlation(before),
            "qll": float(best_qll), "delta": delta, "persistence": persistence,
            "fit_status": status, "optimizer_success": selected_result is not None,
            "optimizer_status": (int(selected_result.status) if selected_result is not None else None),
            "message": (str(selected_result.message) if selected_result is not None else status),
            "nested_qll": float(nested_qll), "nested_a": float(nested[0]),
            "nested_b": float(nested[1]), "qll_improvement": float(nested_qll - best_qll),
            "intercept_min_eigenvalue": minimum_eigenvalue,
            "near_boundary": bool(persistence >= MAX_PERSISTENCE - .003),
            "optimizer_details": details}
