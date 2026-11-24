"""Figures for matched-return CVaR and max-Sharpe MVO on validation data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


FORECASTS = {"HMM": "hmm", "Jump Model": "jump"}
CVAR = "Matched-return CVaR"
MVO = "Max-Sharpe MVO"
HEURISTIC = "3-asset heuristic"
ASSETS = ("Equity", "Bond", "BTC")
ASSET_COLORS = ("#256b91", "#b7cbd5", "#cf821a")
STRATEGY_LINES = {CVAR: ("#176d8a", 2.1), MVO: ("#b95623", 1.3), HEURISTIC: ("#675094", 1.5)}


def format_dates(ax, *, compact=False):
    ax.xaxis.set_major_locator(MonthLocator(bymonth=(3, 9)))
    ax.xaxis.set_major_formatter(DateFormatter("%b\n%Y" if compact else "%b %Y"))
    ax.spines[["top", "right"]].set_visible(False)


def save_paths(paths, output, *, ylabel, formatter, reference):
    fig = Figure(figsize=(11, 4), facecolor="white")
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.88, wspace=0.13)

    for ax, (forecast, frame) in zip(axes, paths.items()):
        ax.axhline(reference, color="#9ca8ae", linewidth=0.9, linestyle="-")
        for strategy in frame.columns:
            color, width = STRATEGY_LINES[strategy]
            ax.plot(frame.index, frame[strategy], color=color, linestyle="-", linewidth=width, label=strategy)
        ax.set_title(f"{forecast} forecasts", fontsize=11, fontweight="bold")
        ax.set_xlim(frame.index[0], frame.index[-1])
        ax.grid(axis="y", color="#e1e7ea", linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        format_dates(ax)

    axes[0].set_ylabel(ylabel)
    axes[0].yaxis.set_major_formatter(formatter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.savefig(output, dpi=220)


def save_allocations(allocations, probabilities, end, output):
    fig = Figure(figsize=(15, 5.4), facecolor="white")
    axes = fig.subplots(2, 4, sharex=True, sharey="row",
                        gridspec_kw={"height_ratios": (2, 1)})
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.85, hspace=0.22, wspace=0.13)

    for i, (forecast, by_strategy) in enumerate(allocations.items()):
        left = axes[0, 2 * i].get_position().x0
        right = axes[0, 2 * i + 1].get_position().x1
        fig.text((left + right) / 2, 0.95, f"{forecast} forecasts", ha="center", fontweight="bold")
        for j, (strategy, weights) in enumerate(by_strategy.items()):
            top, bottom = axes[:, 2 * i + j]
            # Extend the final rebalance through the last validation session.
            plot_dates = weights.index.append(pd.DatetimeIndex([end]))
            plot_weights = np.vstack((weights.to_numpy(), weights.iloc[-1]))
            probability = probabilities[forecast]
            top.stackplot(plot_dates, *plot_weights.T, colors=ASSET_COLORS,
                          labels=("Equity", "Bond", "Bitcoin"), step="post", linewidth=0)
            top.set_title(strategy, fontsize=10)
            top.set_ylim(0, 1)
            top.set_xlim(plot_dates[0], end)
            bottom.axhline(0.5, color="#7a898f", linewidth=0.9, linestyle="-")
            bottom.step(plot_dates, np.r_[probability, probability.iloc[-1]], where="post",
                        color="#4b416b", linewidth=1.7, linestyle="-")
            bottom.set_ylim(0, 1)
            for ax in (top, bottom):
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_axisbelow("line")
                ax.grid(axis="y", color="#536974", linewidth=0.8, alpha=0.34, linestyle="-")
                format_dates(ax, compact=True)

    axes[0, 0].set_ylabel("Rebalance weight")
    axes[1, 0].set_ylabel("Bear probability")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.savefig(output, dpi=220)


def save_comparison_figures(cvar_daily, cvar_monthly, mvo_daily, mvo_monthly, *, initial_date, output,
                            heuristic_daily=None):
    """Plot the same validation days and executed rebalances for each paired strategy."""
    cvar_daily, cvar_monthly = [
        frame[(frame.policy == "matched_return") & (frame.horizon == 3) & (frame.scenarios == "DCC GARCH")]
        for frame in (cvar_daily, cvar_monthly)
    ]
    mvo_model, mvo_policy = "Regime-aware MVO MPO", "Max Sharpe MVO"
    mvo_monthly = mvo_monthly[
        (mvo_monthly["Model"] == mvo_model) & (mvo_monthly["Policy"] == mvo_policy)
        & (mvo_monthly["H"] == 3) & (mvo_monthly["BTC cap"] == "20%")
        & (mvo_monthly["Risk model"] == "DCC GARCH")
    ]
    wealth, allocations, probabilities = {}, {}, {}
    validation_dates = None
    for forecast, experiment in FORECASTS.items():
        cvar_returns = cvar_daily[cvar_daily.experiment == experiment].set_index("date").net_return.sort_index()
        mvo_returns = mvo_daily[f"{mvo_model} | {forecast} | {mvo_policy} | H=3 | cap 20%"]
        if (cvar_returns.empty or cvar_returns.index.has_duplicates
                or not cvar_returns.index.equals(mvo_returns.index) or initial_date >= cvar_returns.index[0]):
            raise ValueError(f"{forecast}: CVaR and MVO must cover the same validation sessions")
        if validation_dates is None:
            validation_dates = cvar_returns.index
        elif not cvar_returns.index.equals(validation_dates):
            raise ValueError("HMM and Jump Model must cover the same validation sessions")

        cvar_weights = cvar_monthly[cvar_monthly.experiment == experiment].set_index("origin").sort_index()
        mvo_weights = mvo_monthly[mvo_monthly.Forecasts == forecast].set_index("rebalance_date").sort_index()
        mvo_weights.index = pd.to_datetime(mvo_weights.index)
        rebalances = validation_dates[~validation_dates.to_period("M").duplicated()]
        if not cvar_weights.index.equals(mvo_weights.index) or not cvar_weights.index.equals(rebalances):
            raise ValueError(f"{forecast}: CVaR and MVO must cover the same validation rebalances")
        p_bear = mvo_weights.p_bear_month1_end
        if not p_bear.between(0, 1).all():
            raise ValueError(f"{forecast}: invalid month-end bear probabilities")

        daily = pd.DataFrame({CVAR: cvar_returns, MVO: mvo_returns})
        if heuristic_daily is not None:
            heuristic_returns = heuristic_daily[forecast]
            if not heuristic_returns.index.equals(validation_dates) or not np.isfinite(heuristic_returns).all():
                raise ValueError(f"{forecast}: heuristic returns must cover the same validation sessions")
            daily[HEURISTIC] = heuristic_returns
        initial = pd.DataFrame(0.0, index=[initial_date], columns=daily.columns)
        wealth[forecast] = pd.concat([initial, daily]).add(1).cumprod()
        allocations[forecast] = {CVAR: cvar_weights[list(ASSETS)], MVO: mvo_weights[list(ASSETS)]}
        probabilities[forecast] = p_bear

    drawdowns = {forecast: frame.div(frame.cummax()).sub(1) for forecast, frame in wealth.items()}
    save_paths(wealth, output / "figure_1_cumulative_wealth.png",
               ylabel="Portfolio wealth",
               formatter=StrMethodFormatter("${x:.2f}"), reference=1)
    save_allocations(allocations, probabilities, validation_dates[-1],
                     output / "figure_2_portfolio_allocations.png")
    save_paths(drawdowns, output / "figure_3_drawdowns.png",
               ylabel="Drawdown",
               formatter=PercentFormatter(1, decimals=0), reference=0)
