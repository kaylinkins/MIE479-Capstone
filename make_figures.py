"""Report figures for hmm_regime_forecasting_mvo_cvar.py (WP 2.1).

Run the HMM forecasting script first. This script ONLY reads existing CSVs;
it does not import or refit the HMM, generate CVaR paths, or modify forecasts.

Usage (place next to the HMM script and portfolio CSVs):
    python make_figures.py
    python make_figures.py --period test

Files written in hmm_outputs/ (or hmm_outputs_test/):
    fig1_regimes_training.png           In-sample training regimes / asset growth
    fig2_regime_probability.png         Walk-forward validation/test regimes
    fig3_forecast_horizons.png          Three-month bear-probability forecasts
    fig4_regime_characteristics.png     Training-state mean/volatility/correlation
    fig5_mvo_forecast_moments.png       Forward compounded means / volatilities
    fig6_mvo_cvar_consistency.png       Analytical vs simulated moment differences

Definitions:
  * "Bear" means the higher-EQUITY-variance regime, not necessarily a
    negative-return regime.
  * Training filtered state probabilities use parameters fitted using the
    entire training period (in-sample parameters, not truly OOS).
  * Validation/test filtered probabilities are walk-forward filtered, with
    model parameters re-estimated at the start of each month. The filtered
    probability for a given trading day uses that day's observed returns.
  * Forecast bear probability for a month is the AVERAGE of daily state
    probabilities over that month's actual exchange trading sessions.
  * Monthly means/covariances are for COMPOUNDED monthly simple returns.
  * "Consistency" plots show Monte Carlo sampling discrepancies, not
    empirical model-forecast accuracy.

Dependencies: numpy, pandas, matplotlib. No hmmlearn or exchange_calendars needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSETS = ("Equity", "Bond", "BTC")
ASSET_COLORS = {"Equity": "#2A78D6", "Bond": "#1BAF7A", "BTC": "#EB6834"}
HORIZON_COLORS = {1: "#2A78D6", 2: "#EB6834", 3: "#1BAF7A"}
BULL_COLOR = "#2A78D6"
BEAR_COLOR = "#EB6834"
BEAR_SHADE = "#A9ACB2"


def read_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if not set(ASSETS).issubset(frame.columns):
        raise ValueError(f"{path.name}: required asset columns {ASSETS}")
    frame = frame.loc[:, list(ASSETS)].sort_index()
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"{path.name}: missing or empty returns")
    return frame


def load_outputs(root: Path, period: str):
    output = root / ("hmm_outputs" if period == "validation" else "hmm_outputs_test")
    if not output.exists():
        raise FileNotFoundError(
            f"{output} not found. Run hmm_regime_forecasting_mvo_cvar.py "
            f"with PERIOD='{period}' first."
        )
    train = read_returns(root / "portfolio_train.csv")
    oos = read_returns(root / ("portfolio_validation.csv" if period == "validation"
                               else "portfolio_test.csv"))
    daily = pd.read_csv(output / "daily_bear_probability.csv", parse_dates=["date"])
    if not {"date", "p_bear"}.issubset(daily.columns):
        raise ValueError("daily_bear_probability.csv missing date or p_bear")
    daily = daily.set_index("date").sort_index()
    probs = pd.read_csv(output / "regime_probabilities.csv", parse_dates=["rebalance_date"])
    summary = pd.read_csv(output / "regime_summary.csv", index_col=0)
    means = pd.read_csv(output / "expected_returns.csv", parse_dates=["rebalance_date"])
    covs = pd.read_csv(output / "covariances.csv", parse_dates=["rebalance_date"])
    diagnostic = pd.read_csv(output / "moment_consistency.csv",
                             parse_dates=["rebalance_date"])
    required_probs = {"rebalance_date", "p_bear_today"} | {
        f"p_bear_month{k}" for k in (1, 2, 3)
    }
    if not required_probs.issubset(probs.columns):
        raise ValueError(f"regime_probabilities.csv missing {required_probs - set(probs.columns)}")
    if not {"bull", "bear"}.issubset(summary.columns):
        raise ValueError("regime_summary.csv missing bull or bear column")
    if not {"rebalance_date", "month_ahead", *ASSETS}.issubset(means.columns):
        raise ValueError("expected_returns.csv missing forecast keys or assets")
    if not {"rebalance_date", "month_ahead", "asset", *ASSETS}.issubset(covs.columns):
        raise ValueError("covariances.csv missing covariance keys or assets")
    if not {"rebalance_date", "month_ahead", "max_mean_z_score",
            "max_abs_cov_difference"}.issubset(diagnostic.columns):
        raise ValueError("moment_consistency.csv has an unexpected format")

    train_probs = daily["p_bear"].reindex(train.index)
    oos_probs = daily["p_bear"].reindex(oos.index)
    if train_probs.isna().any():
        raise ValueError("Some training dates lack filtered bear probabilities")
    if oos_probs.isna().any():
        raise ValueError("Some validation/test dates lack filtered bear probabilities")
    if not daily["p_bear"].between(0, 1).all():
        raise ValueError("Filtered regime probabilities must be between 0 and 1")
    if "source" in daily.columns:
        expected_source = "out_of_sample_walk_forward"
        wrong = daily["source"].reindex(oos.index).ne(expected_source)
        if wrong.any():
            raise ValueError("OOS plotted probabilities are not walk-forward filtered")
    return output, train, oos, train_probs, oos_probs, probs, summary, means, covs, diagnostic


def setup_style():
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def shade_bear(ax, dates, probabilities):
    """Shade p(bear)>0.5 without altering data/axis scales (incl. log axes)."""
    ax.fill_between(dates, 0, 1,
                    where=(np.asarray(probabilities) > 0.5),
                    transform=ax.get_xaxis_transform(),
                    step="mid", facecolor=BEAR_SHADE, alpha=0.29,
                    edgecolor="none", zorder=0)


def save(fig, output: Path, filename: str):
    fig.savefig(output / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def plot_period_regimes(frame, bear, output, *, training: bool, period: str):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.1), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})
    for ax, asset in zip(axes[:3], ASSETS):
        growth = (1 + frame[asset]).cumprod()
        ax.plot(frame.index, growth, linewidth=1.3, color=ASSET_COLORS[asset])
        shade_bear(ax, frame.index, bear)
        ax.set_ylabel("Growth of $1")
        ax.set_title(asset, loc="left", fontweight="bold")
        if asset == "BTC":
            ax.set_yscale("log")
        ax.margins(x=0)
    axes[0].legend(handles=[Patch(facecolor=BEAR_SHADE, alpha=0.5,
                                  label="High-volatility state: P(bear) > 0.5")],
                   loc="upper left", fontsize=8)

    axes[3].plot(bear.index, bear.values, color=BEAR_COLOR, linewidth=1.1)
    axes[3].fill_between(bear.index, 0, bear.values,
                         color=BEAR_COLOR, alpha=0.15)
    axes[3].axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("P(bear)")
    axes[3].set_title("Filtered probability of high-volatility regime", loc="left")
    format_time_axis(axes[-1])
    if training:
        title = "Fig 1. Regimes in training data"
        subtitle = ("Daily filtered probabilities; parameters estimated using "
                    "the full training sample (in-sample fit)")
        filename = "fig1_regimes_training.png"
    else:
        title = f"Fig 2. Regimes in {period} data"
        subtitle = ("Parameters refitted monthly using past data only; each "
                    "day's filtered probability uses returns through that day")
        filename = "fig2_regime_probability.png"
    fig.suptitle(title, fontsize=13, weight="bold", x=0.08, y=0.995, ha="left")
    fig.text(0.08, 0.955, subtitle, fontsize=8.6, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.0)
    save(fig, output, filename)


def plot_forecast_probabilities(probs, output):
    probs = probs.sort_values("rebalance_date")
    fig, ax = plt.subplots(figsize=(10, 4))
    for k in (1, 2, 3):
        ax.plot(probs["rebalance_date"], probs[f"p_bear_month{k}"],
                label=f"Month {k}", color=HORIZON_COLORS[k],
                linewidth=1.4, marker="o", markersize=2.5)
    ax.plot(probs["rebalance_date"], probs["p_bear_today"],
            color="#555555", alpha=0.60, linewidth=1.1,
            linestyle="--", label="As of prior trading session")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability of high-volatility state")
    ax.set_title("Fig 3. Regime probability forecasts at each rebalance",
                 loc="left", fontweight="bold", pad=16)
    ax.text(0, 1.01, "Monthly values average daily probabilities over actual trading sessions",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    format_time_axis(ax)
    fig.tight_layout()
    save(fig, output, "fig3_forecast_horizons.png")


def summary_row(summary, asset, category):
    if category == "mean":
        candidates = (f"{asset} return (ann. arithmetic)", f"{asset} return (ann.)")
    else:
        candidates = (f"{asset} volatility (ann.)",)
    for candidate in candidates:
        if candidate in summary.index:
            return summary.loc[candidate, ["bull", "bear"]].to_numpy(dtype=float)
    raise ValueError(f"regime_summary.csv missing row for {asset} {category}")


def plot_regime_characteristics(summary, output):
    means = np.array([summary_row(summary, asset, "mean") for asset in ASSETS])
    vols = np.array([summary_row(summary, asset, "vol") for asset in ASSETS])
    pairs = ["Corr Equity-Bond", "Corr Equity-BTC", "Corr Bond-BTC"]
    if not set(pairs).issubset(summary.index):
        raise ValueError("regime_summary.csv missing correlation rows")
    corr = summary.loc[pairs, ["bull", "bear"]].to_numpy(dtype=float)
    panel_data = [
        ("Arithmetic mean return (annualized, %)", means * 100, list(ASSETS)),
        ("Volatility (annualized, %)", vols * 100, list(ASSETS)),
        ("Within-regime correlation", corr, ["Eq–Bond", "Eq–BTC", "Bond–BTC"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 3.7))
    x = np.arange(3)
    for ax, (title, values, labels) in zip(axes, panel_data):
        ax.bar(x - 0.19, values[:, 0], 0.38, color=BULL_COLOR, label="Bull / low-vol")
        ax.bar(x + 0.19, values[:, 1], 0.38, color=BEAR_COLOR, label="Bear / high-vol")
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left")
        if title.startswith("Within"):
            ax.set_ylim(-1, 1)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig 4. Fitted regime characteristics (training sample)",
                 x=0.04, ha="left", weight="bold", fontsize=13)
    fig.text(0.04, 0.90,
             "High-volatility state is defined by higher equity variance, not the sign of its mean return.",
             color="#555555", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.6)
    save(fig, output, "fig4_regime_characteristics.png")


def plot_mvo_moments(means, covs, output):
    """Compounded conditional monthly means and SDs for each asset/horizon."""
    # Diagonal of each (forecast date, horizon) covariance matrix.
    diag_rows = []
    for asset in ASSETS:
        these = covs.loc[covs["asset"] == asset,
                          ["rebalance_date", "month_ahead", asset]].copy()
        these = these.rename(columns={asset: "variance"})
        these["asset"] = asset
        diag_rows.append(these)
    diag = pd.concat(diag_rows, ignore_index=True)
    if diag.duplicated(["rebalance_date", "month_ahead", "asset"]).any():
        raise ValueError("covariances.csv has duplicate diagonal entries")

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for row, asset in enumerate(ASSETS):
        mm = means[["rebalance_date", "month_ahead", asset]].copy()
        cc = diag[diag["asset"] == asset]
        series = mm.merge(cc, on=["rebalance_date", "month_ahead"],
                          validate="one_to_one", how="left").sort_values("rebalance_date")
        if series["variance"].isna().any():
            raise ValueError(f"Missing {asset} forecast variances")
        if (series["variance"] < -1e-10).any():
            raise ValueError(f"Negative {asset} forecast variance")
        series["vol"] = np.sqrt(np.maximum(series["variance"], 0))
        for k in (1, 2, 3):
            h = series[series["month_ahead"] == k]
            axes[row, 0].plot(h["rebalance_date"], 100 * h[asset],
                              label=f"Month {k}", color=HORIZON_COLORS[k], linewidth=1.45)
            axes[row, 1].plot(h["rebalance_date"], 100 * h["vol"],
                              label=f"Month {k}", color=HORIZON_COLORS[k], linewidth=1.45)
        axes[row, 0].axhline(0, linewidth=0.7, color="#777777")
        axes[row, 0].set_ylabel(f"{asset}\nReturn (%)")
        axes[row, 1].set_ylabel(f"{asset}\nVolatility (%)")
        format_time_axis(axes[row, 0])
        format_time_axis(axes[row, 1])
    axes[0, 0].set_title("Conditional expected compounded monthly return",
                         loc="left", weight="bold")
    axes[0, 1].set_title("Conditional standard deviation of compounded monthly return",
                         loc="left", weight="bold")
    axes[0, 1].legend(ncol=3, fontsize=8)
    fig.suptitle("Fig 5. Regime-aware forecasts for multi-period MVO",
                 x=0.06, y=0.992, ha="left", fontsize=13, weight="bold")
    fig.text(0.06, 0.936,
             "Analytical HMM moments with daily regime transitions and each month's actual trading-day count",
             fontsize=8.6, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=1.5)
    save(fig, output, "fig5_mvo_forecast_moments.png")


def plot_moment_consistency(diagnostic, output):
    """Sampling error check: CVaR scenario moments vs exact MVO moments."""
    diag = diagnostic.sort_values("rebalance_date")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for k in (1, 2, 3):
        h = diag[diag["month_ahead"] == k]
        if h.empty:
            continue
        axes[0].plot(h["rebalance_date"], h["max_mean_z_score"],
                     label=f"Month {k}", color=HORIZON_COLORS[k],
                     linewidth=1.2, marker="o", markersize=2.1)
        axes[1].plot(h["rebalance_date"], h["max_abs_cov_difference"] * 10000,
                     label=f"Month {k}", color=HORIZON_COLORS[k],
                     linewidth=1.2, marker="o", markersize=2.1)
    axes[0].set_title("Largest absolute expected-return discrepancy, standardized by MC SE",
                      loc="left")
    axes[0].set_ylabel("Max |z-score|")
    axes[1].set_title("Largest absolute covariance discrepancy", loc="left")
    axes[1].set_ylabel("Percentage-points²")
    axes[1].legend(ncol=3)
    format_time_axis(axes[1])
    fig.suptitle("Fig 6. MVO analytical moments vs CVaR scenario moments",
                 x=0.07, y=0.992, ha="left", fontsize=13, weight="bold")
    fig.text(0.07, 0.930,
             "Diagnostic of Monte Carlo sampling error, NOT out-of-sample forecasting accuracy",
             fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.87), h_pad=1.5)
    save(fig, output, "fig6_mvo_cvar_consistency.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=("validation", "test"), default="validation",
                        help="Choose the forecast run's output folder (default: validation)")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing portfolio CSVs and HMM output folder")
    args = parser.parse_args()
    root = args.data_dir.expanduser().resolve()
    (output, train, oos, p_train, p_oos, probs, summary,
     means, covs, diagnostic) = load_outputs(root, args.period)
    setup_style()
    print(f"Creating figures in {output}:")
    plot_period_regimes(train, p_train, output, training=True, period=args.period)
    plot_period_regimes(oos, p_oos, output, training=False, period=args.period)
    plot_forecast_probabilities(probs, output)
    plot_regime_characteristics(summary, output)
    plot_mvo_moments(means, covs, output)
    plot_moment_consistency(diagnostic, output)
    print("Saved 6 figures. Forecast files were not modified.")


if __name__ == "__main__":
    main()
