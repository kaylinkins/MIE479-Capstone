"""Black-Litterman expected returns for Equity, Bond and BTC.

All calculations use DAILY arithmetic simple returns, in asset-column order.
Annual input rates are arithmetic annualized means, divided by 252 (not CAGR).
Uses analytical NONLINEAR Ledoit-Wolf shrinkage via nonlinshrink.
No downloads, fitting, or file writes happen when this module is imported.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def validate_returns(frame, name="returns"):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame.")
    if not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError(f"{name} must have unique asset columns and dates.")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError(f"{name} needs a valid DatetimeIndex.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError(f"{name} contains invalid simple returns or missing values.")
    return frame.sort_index()


def nonlinear_covariance(train):
    """Estimate daily covariance using TRAINING observations only."""
    train = validate_returns(train, "train")
    if len(train) < max(12, train.shape[1] + 2):
        raise ValueError("Too few observations for nonlinear shrinkage.")
    import nonlinshrink as nls
    sigma = np.asarray(nls.shrink_cov(train.to_numpy(dtype=float)), dtype=float)
    sigma = (sigma + sigma.T) / 2  # remove floating point asymmetry only
    if sigma.shape != (train.shape[1], train.shape[1]) or not np.isfinite(sigma).all():
        raise ValueError("Nonlinear covariance estimate is invalid.")
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("Nonlinear covariance must be positive definite; inspect data.")
    return pd.DataFrame(sigma, index=train.columns, columns=train.columns)


def posterior_excess_returns(pi, sigma, P, q, omega, tau=0.05):
    """Stable equivalent of the precision/inverse formula in the report.

    mu = pi + tau*Sigma*P' solve(P*tau*Sigma*P' + Omega, q - P*pi)
    Inputs here are DAILY EXCESS returns; omega is in daily-return squared units.
    A zero-row P returns the equilibrium prior unchanged.
    """
    pi = np.asarray(pi, dtype=float)
    sigma, P = np.asarray(sigma, dtype=float), np.asarray(P, dtype=float)
    q, omega = np.asarray(q, dtype=float), np.asarray(omega, dtype=float)
    n = len(pi)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive.")
    if sigma.shape != (n, n) or P.ndim != 2 or P.shape[1] != n:
        raise ValueError("Sigma/P dimensions do not match the prior.")
    k = P.shape[0]
    if q.shape != (k,) or omega.shape != (k, k):
        raise ValueError("Q/Omega dimensions do not match P.")
    if not all(np.isfinite(x).all() for x in (pi, sigma, P, q, omega)):
        raise ValueError("BL inputs must be finite.")
    if not np.allclose(sigma, sigma.T) or np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("Sigma must be symmetric positive definite.")
    if not k:
        return pi.copy()
    if not np.allclose(omega, omega.T) or np.linalg.eigvalsh(omega).min() < 0:
        raise ValueError("Omega must be symmetric positive semidefinite.")
    prior_cov = tau * sigma
    return pi + prior_cov @ P.T @ np.linalg.solve(
        P @ prior_cov @ P.T + omega, q - P @ pi
    )


def fit_black_litterman(train, config):
    """Return labelled means, covariance, views and settings for CVaR.

    benchmark_weights: documented asset-class weights, OR provide market_values
    measured on one consistent date/currency/universe, which are normalized.
    risk_aversion: positive lambda; fixed explicitly, not estimated from test data.
    views: list of {name, assets: {asset: coefficient}, annual_return, confidence}.
    View returns are TOTAL returns. Subtract rf * sum(P row) before BL:
    absolute view P=[1,0,0] subtracts rf; relative [1,-1,0] subtracts zero.
    Confidence 0 drops a view. Positive confidences must be below 1.
    """
    train = validate_returns(train, "train")
    assets = list(train.columns)
    n = len(assets)
    sigma_df = nonlinear_covariance(train)
    sigma = sigma_df.to_numpy()
    if ("market_values" in config) == ("benchmark_weights" in config):
        raise ValueError("Provide exactly one of market_values or benchmark_weights.")
    inputs = config.get("market_values", config.get("benchmark_weights"))
    if set(inputs) != set(assets):
        raise ValueError("Benchmark inputs must match return asset names exactly.")
    b = np.array([inputs[a] for a in assets], dtype=float)
    if not np.isfinite(b).all() or (b < 0).any() or b.sum() <= 0:
        raise ValueError("Benchmark values must be finite, nonnegative, with positive sum.")
    if "benchmark_weights" in config and not np.isclose(b.sum(), 1.0):
        raise ValueError("benchmark_weights must sum to one.")
    b = b / b.sum()
    lam, tau = float(config["risk_aversion"]), float(config.get("tau", 0.05))
    rf = float(config.get("risk_free_annual", 0.0)) / TRADING_DAYS
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(rf):
        raise ValueError("risk_aversion must be positive and risk-free rate finite.")
    pi = lam * sigma @ b  # implied equilibrium EXCESS returns
    rows, q_total, confidence, names = [], [], [], []
    for j, view in enumerate(config.get("views", [])):
        coeffs = view["assets"]
        if set(coeffs) - set(assets):
            raise ValueError(f"Unknown asset in view {j}: {coeffs}")
        p = np.array([coeffs.get(a, 0.0) for a in assets], dtype=float)
        c = float(view["confidence"])
        forecast = float(view["annual_return"]) / TRADING_DAYS
        if not np.isfinite(c) or not 0 <= c < 1:
            raise ValueError("Confidence must be in [0,1); use e.g. 0.99 rather than 1.")
        if not np.isfinite(p).all() or not np.any(p) or not np.isfinite(forecast):
            raise ValueError("Each view needs finite nonzero coefficients and a finite forecast.")
        if c == 0:
            continue  # no information from zero-confidence views
        rows.append(p)
        q_total.append(forecast)
        confidence.append(c)
        names.append(f"{j+1}: {view.get('name', 'view')}")
    P = np.asarray(rows, dtype=float).reshape(-1, n)
    q_total = np.asarray(q_total)
    q = q_total - rf * P.sum(axis=1)
    # Idzorek-style confidence-to-uncertainty mapping (diagonal independent views).
    # 50% -> view variance equals projected prior-mean variance.
    c = np.asarray(confidence)
    omega = np.diag(((1-c)/c) * np.diag(P @ (tau * sigma) @ P.T))
    mu_excess = posterior_excess_returns(pi, sigma, P, q, omega, tau)
    estimates = pd.DataFrame({
        "benchmark_weight": b,
        "prior_excess_daily": pi,
        "prior_total_daily": pi + rf,
        "bl_excess_daily": mu_excess,
        "bl_total_daily": mu_excess + rf,
        "bl_total_annual_arithmetic": (mu_excess + rf) * TRADING_DAYS,
        "historical_total_annual_arithmetic": train.mean().to_numpy() * TRADING_DAYS,
    }, index=assets)
    view_table = pd.DataFrame({"Q_total_daily": q_total, "Q_excess_daily": q,
                               "confidence": c}, index=names)
    return {"estimates": estimates, "covariance_daily": sigma_df,
            "P": pd.DataFrame(P, index=names, columns=assets),
            "views": view_table, "omega_daily": pd.DataFrame(omega, index=names, columns=names),
            "metadata": {"mode": config.get("mode", "unspecified"),
                         "benchmark_note": config.get("benchmark_note", ""),
                         "covariance_method": "Ledoit-Wolf analytical nonlinear shrinkage (nonlinshrink 0.7)",
                         "train_start": str(train.index.min().date()),
                         "train_end": str(train.index.max().date()),
                         "train_rows": len(train), "trading_days": TRADING_DAYS,
                         "risk_aversion": lam, "tau": tau,
                         "risk_free_annual_arithmetic": rf * TRADING_DAYS,
                         "scenario_policy": "Historical raw returns; BL used in expected-return constraint only"}}


def save_black_litterman(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, filename in {
        "estimates": "bl_expected_returns.csv", "covariance_daily": "bl_covariance_daily.csv",
        "P": "bl_P.csv", "views": "bl_views.csv", "omega_daily": "bl_omega_daily.csv",
    }.items():
        result[key].to_csv(directory / filename)
    (directory / "bl_metadata.json").write_text(json.dumps(result["metadata"], indent=2), encoding="utf-8")


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base)
    parser.add_argument("--config", type=Path, default=base / "bl_config_demo.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = pd.read_csv(args.data_dir / "portfolio_train.csv", index_col=0, parse_dates=True)
    result = fit_black_litterman(train, config)
    dest = args.output_dir or args.data_dir / "bl_outputs"
    save_black_litterman(result, dest)
    print("MODE:", config.get("mode", "unspecified"))
    print(result["estimates"].to_string())
    print("Saved BL outputs to", dest)


if __name__ == "__main__":
    main()
