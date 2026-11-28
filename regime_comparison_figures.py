"""Validation figures for CVaR, MVO and the three-asset heuristic."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


FORECASTS = {"HMM": "hmm", "Jump Model": "jump"}"""Validation figures for CVaR, MVO and the three-asset heuristic."""
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


def add_figure_heading(fig, title, subtitle):
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", va="top", fontsize=10, color="#41555e")


def save_paths(paths, output, *, ylabel, formatter, reference, styles=STRATEGY_LINES, legend_columns=None,
               title=None, subtitle=None):
    height = 4.4 if legend_columns else 4
    fig = Figure(figsize=(11, height + 0.6 if title else height), facecolor="white")
    axes = fig.subplots(1, len(paths), sharex=True, sharey=True, squeeze=False)[0]
    show_forecast_titles = any(paths) and (not title or len(paths) > 1)
    plot_top = 0.83 if title else (0.88 if show_forecast_titles else 0.96)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23 if legend_columns else 0.20,
                        top=plot_top, wspace=0.13)
    if title:
        add_figure_heading(fig, title, subtitle)

    for ax, (forecast, frame) in zip(axes, paths.items()):
        ax.axhline(reference, color="#9ca8ae", linewidth=0.9, linestyle="-")
        for strategy in frame.columns:
            color, width = styles[strategy]
            ax.plot(frame.index, frame[strategy], color=color, linestyle="-", linewidth=width, label=strategy)
        if forecast and show_forecast_titles:
            ax.set_title(f"{forecast} forecasts", fontsize=11, fontweight="bold")
        ax.set_xlim(frame.index[0], frame.index[-1])
        ax.grid(axis="y", color="#e1e7ea", linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        format_dates(ax)

    axes[0].set_ylabel(ylabel)
    axes[0].yaxis.set_major_formatter(formatter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=legend_columns or len(labels), frameon=False)
    fig.savefig(output, dpi=220)


def save_allocations(allocations, probabilities, end, output, *, width=None, title=None, subtitle=None):
    strategies = len(next(iter(allocations.values())))
    columns = len(allocations) * strategies
    fig = Figure(figsize=(width or (5 * strategies + 5), 6 if title else 5.4), facecolor="white")
    axes = fig.subplots(2, columns, sharex=True, sharey="row",
                        gridspec_kw={"height_ratios": (2, 1)}, squeeze=False)
    left_margin = max(0.075, 0.86 / fig.get_figwidth())
    show_forecasts = not title or len(allocations) > 1
    plot_top = 0.8 if title else 0.85
    fig.subplots_adjust(left=left_margin, right=0.98, bottom=0.17, top=plot_top, hspace=0.36, wspace=0.13)
    if title:
        add_figure_heading(fig, title, subtitle)

    for i, (forecast, by_strategy) in enumerate(allocations.items()):
        first = strategies * i
        if show_forecasts:
            left = axes[0, first].get_position().x0
            right = axes[0, first + strategies - 1].get_position().x1
            fig.text((left + right) / 2, 0.855 if title else 0.95, f"{forecast} forecasts",
                     ha="center", fontweight="bold")
        for j, (strategy, weights) in enumerate(by_strategy.items()):
            top, bottom = axes[:, first + j]
            # Extend the final rebalance through the last session.
            plot_dates = weights.index.append(pd.DatetimeIndex([end]))
            plot_weights = np.vstack((weights.to_numpy(), weights.iloc[-1]))
            probability, threshold = probabilities[forecast][strategy]
            top.stackplot(plot_dates, *plot_weights.T, colors=ASSET_COLORS,
                          labels=("Equity", "Bond", "Bitcoin"), step="post", linewidth=0)
            if not title or strategies > 1:
                top.set_title(strategy, fontsize=10)
            top.set_ylim(0, 1)
            top.set_xlim(plot_dates[0], end)
            if strategy == HEURISTIC:
                signal_label = "As-of bear state" if forecast == "Jump Model" else "As-of bear probability"
            else:
                signal_label = "Month-end bear probability"
            bottom.set_title(signal_label, fontsize=9)
            bottom.axhline(threshold, color="#7a898f", linewidth=0.9, linestyle="-")
            bottom.step(plot_dates, np.r_[probability, probability.iloc[-1]], where="post",
                        color="#4b416b", linewidth=1.7, linestyle="-")
            bottom.set_ylim(-0.025, 1.025)
            for ax in (top, bottom):
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_axisbelow("line")
                ax.grid(axis="y", color="#536974", linewidth=0.8, alpha=0.34, linestyle="-")
                format_dates(ax, compact=True)

    axes[0, 0].set_ylabel("Rebalance weight")
    axes[1, 0].set_ylabel("Bear signal")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.savefig(output, dpi=220)


def save_testing_paths(runs, *, initial_date, output, forecast=None, subtitle=None):
    """Draw wealth and drawdown from the same aligned daily net returns."""
    returns = pd.DataFrame({label: daily.set_index('date').net_return
                            for label, (daily, _) in runs.items()})
    if returns.empty or returns.isna().any().any():
        raise ValueError('All plotted strategies must share the same testing sessions')
    initial = pd.DataFrame(0., index=[initial_date], columns=returns.columns)
    wealth = pd.concat([initial, returns]).add(1).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1)
    colors = ('#176d8a', '#b95623', '#675094', '#367b53', '#a8415e')
    styles = {label: (color, 1.6) for label, color in zip(runs, colors)}
    wealth_file, drawdown_file = ('figure_1_cumulative_wealth.png', 'figure_3_drawdowns.png') if forecast else (
        'comparison_cumulative_wealth.png', 'comparison_drawdowns.png')
    appearance = dict(styles=styles, legend_columns=None if forecast else 3,
                      subtitle=subtitle or "Test-period strategy comparison")
    save_paths({forecast: wealth}, output / wealth_file,
               ylabel='Portfolio wealth', formatter=StrMethodFormatter('${x:.2f}'), reference=1,
               title='Cumulative Portfolio Wealth', **appearance)
    save_paths({forecast: drawdown}, output / drawdown_file,
               ylabel='Drawdown', formatter=PercentFormatter(1), reference=0,
               title='Portfolio Drawdowns', **appearance)
    return wealth.index[-1]


def save_testing_figures(runs, *, initial_date, output, forecast, heuristic=False, parameter=None):
    """Draw the three figures for a model or a parameter sweep."""
    subtitle = HEURISTIC if heuristic else output.name
    if parameter:
        subtitle = f"{output.parent.name} · {parameter}"
    end = save_testing_paths(runs, initial_date=initial_date, output=output, forecast=forecast, subtitle=subtitle)

    allocations, probabilities = {}, {}
    for label, (_, monthly) in runs.items():
        group, strategy = (label, HEURISTIC) if heuristic else (forecast, label)
        monthly = monthly.set_index('origin')
        allocations.setdefault(group, {})[strategy] = monthly[list(ASSETS)]
        probabilities.setdefault(group, {})[strategy] = (monthly.bear_signal, monthly.threshold.iloc[0])
    width = max(9, 3.25 * len(runs) + 2)
    save_allocations(allocations, probabilities, end,
                     output / 'figure_2_portfolio_allocations.png', width=width,
                     title='Portfolio Allocations Over Time', subtitle=subtitle)


def save_comparison_figures(cvar_daily, cvar_monthly, mvo_daily, mvo_monthly, *, initial_date, output,
                            heuristic=None):
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
        allocations[forecast] = {CVAR: cvar_weights[list(ASSETS)], MVO: mvo_weights[list(ASSETS)]}
        probabilities[forecast] = {CVAR: (p_bear, 0.5), MVO: (p_bear, 0.5)}
        if heuristic is not None:
            heuristic_returns, weights, bear_signal, threshold = heuristic[forecast]
            if not heuristic_returns.index.equals(validation_dates) or not np.isfinite(heuristic_returns).all():
                raise ValueError(f"{forecast}: heuristic returns must cover the same validation sessions")
            if not weights.index.equals(rebalances) or not bear_signal.index.equals(rebalances):
                raise ValueError(f"{forecast}: heuristic weights and signals must cover the same rebalances")
            daily[HEURISTIC] = heuristic_returns
            allocations[forecast][HEURISTIC] = weights[list(ASSETS)]
            probabilities[forecast][HEURISTIC] = bear_signal, threshold
        initial = pd.DataFrame(0.0, index=[initial_date], columns=daily.columns)
        wealth[forecast] = pd.concat([initial, daily]).add(1).cumprod()

    drawdowns = {forecast: frame.div(frame.cummax()).sub(1) for forecast, frame in wealth.items()}
    save_paths(wealth, output / "figure_1_cumulative_wealth.png",
               ylabel="Portfolio wealth",
               formatter=StrMethodFormatter("${x:.2f}"), reference=1)
    save_allocations(allocations, probabilities, validation_dates[-1],
                     output / "figure_2_portfolio_allocations.png")
    save_paths(drawdowns, output / "figure_3_drawdowns.png",
               ylabel="Drawdown",
               formatter=PercentFormatter(1, decimals=0), reference=0)

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


def add_figure_heading(fig, title, subtitle):
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", va="top", fontsize=10, color="#41555e")


def save_paths(paths, output, *, ylabel, formatter, reference, styles=STRATEGY_LINES, legend_columns=None,
               title=None, subtitle=None):
    height = 4.4 if legend_columns else 4
    fig = Figure(figsize=(11, height + 0.6 if title else height), facecolor="white")
    axes = fig.subplots(1, len(paths), sharex=True, sharey=True, squeeze=False)[0]
    show_forecast_titles = any(paths) and (not title or len(paths) > 1)
    plot_top = 0.83 if title else (0.88 if show_forecast_titles else 0.96)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23 if legend_columns else 0.20,
                        top=plot_top, wspace=0.13)
    if title:
        add_figure_heading(fig, title, subtitle)

    for ax, (forecast, frame) in zip(axes, paths.items()):
        ax.axhline(reference, color="#9ca8ae", linewidth=0.9, linestyle="-")
        for strategy in frame.columns:
            color, width = styles[strategy]
            ax.plot(frame.index, frame[strategy], color=color, linestyle="-", linewidth=width, label=strategy)
        if forecast and show_forecast_titles:
            ax.set_title(f"{forecast} forecasts", fontsize=11, fontweight="bold")
        ax.set_xlim(frame.index[0], frame.index[-1])
        ax.grid(axis="y", color="#e1e7ea", linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        format_dates(ax)

    axes[0].set_ylabel(ylabel)
    axes[0].yaxis.set_major_formatter(formatter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=legend_columns or len(labels), frameon=False)
    fig.savefig(output, dpi=220)


def save_allocations(allocations, probabilities, end, output, *, width=None, title=None, subtitle=None):
    strategies = len(next(iter(allocations.values())))
    columns = len(allocations) * strategies
    fig = Figure(figsize=(width or (5 * strategies + 5), 6 if title else 5.4), facecolor="white")
    axes = fig.subplots(2, columns, sharex=True, sharey="row",
                        gridspec_kw={"height_ratios": (2, 1)}, squeeze=False)
    left_margin = max(0.075, 0.86 / fig.get_figwidth())
    show_forecasts = not title or len(allocations) > 1
    plot_top = 0.8 if title else 0.85
    fig.subplots_adjust(left=left_margin, right=0.98, bottom=0.17, top=plot_top, hspace=0.36, wspace=0.13)
    if title:
        add_figure_heading(fig, title, subtitle)

    for i, (forecast, by_strategy) in enumerate(allocations.items()):
        first = strategies * i
        if show_forecasts:
            left = axes[0, first].get_position().x0
            right = axes[0, first + strategies - 1].get_position().x1
            fig.text((left + right) / 2, 0.855 if title else 0.95, f"{forecast} forecasts",
                     ha="center", fontweight="bold")
        for j, (strategy, weights) in enumerate(by_strategy.items()):
            top, bottom = axes[:, first + j]
            # Extend the final rebalance through the last session.
            plot_dates = weights.index.append(pd.DatetimeIndex([end]))
            plot_weights = np.vstack((weights.to_numpy(), weights.iloc[-1]))
            probability, threshold = probabilities[forecast][strategy]
            top.stackplot(plot_dates, *plot_weights.T, colors=ASSET_COLORS,
                          labels=("Equity", "Bond", "Bitcoin"), step="post", linewidth=0)
            if not title or strategies > 1:
                top.set_title(strategy, fontsize=10)
            top.set_ylim(0, 1)
            top.set_xlim(plot_dates[0], end)
            if strategy == HEURISTIC:
                signal_label = "As-of bear state" if forecast == "Jump Model" else "As-of bear probability"
            else:
                signal_label = "Month-end bear probability"
            bottom.set_title(signal_label, fontsize=9)
            bottom.axhline(threshold, color="#7a898f", linewidth=0.9, linestyle="-")
            bottom.step(plot_dates, np.r_[probability, probability.iloc[-1]], where="post",
                        color="#4b416b", linewidth=1.7, linestyle="-")
            bottom.set_ylim(-0.025, 1.025)
            for ax in (top, bottom):
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_axisbelow("line")
                ax.grid(axis="y", color="#536974", linewidth=0.8, alpha=0.34, linestyle="-")
                format_dates(ax, compact=True)

    axes[0, 0].set_ylabel("Rebalance weight")
    axes[1, 0].set_ylabel("Bear signal")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.savefig(output, dpi=220)


def save_testing_paths(runs, *, initial_date, output, forecast=None, subtitle=None):
    """Draw wealth and drawdown from the same aligned daily net returns."""
    returns = pd.DataFrame({label: daily.set_index('date').net_return
                            for label, (daily, _) in runs.items()})
    if returns.empty or returns.isna().any().any():
        raise ValueError('All plotted strategies must share the same testing sessions')
    initial = pd.DataFrame(0., index=[initial_date], columns=returns.columns)
    wealth = pd.concat([initial, returns]).add(1).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1)
    colors = ('#176d8a', '#b95623', '#675094', '#367b53', '#a8415e')
    styles = {label: (color, 1.6) for label, color in zip(runs, colors)}
    wealth_file, drawdown_file = ('figure_1_cumulative_wealth.png', 'figure_3_drawdowns.png') if forecast else (
        'comparison_cumulative_wealth.png', 'comparison_drawdowns.png')
    appearance = dict(styles=styles, legend_columns=None if forecast else 3,
                      subtitle=subtitle or "Test-period strategy comparison")
    save_paths({forecast: wealth}, output / wealth_file,
               ylabel='Portfolio wealth', formatter=StrMethodFormatter('${x:.2f}'), reference=1,
               title='Cumulative Portfolio Wealth', **appearance)
    save_paths({forecast: drawdown}, output / drawdown_file,
               ylabel='Drawdown', formatter=PercentFormatter(1), reference=0,
               title='Portfolio Drawdowns', **appearance)
    return wealth.index[-1]


def save_testing_figures(runs, *, initial_date, output, forecast, heuristic=False, parameter=None):
    """Draw the three figures for a model or a parameter sweep."""
    subtitle = HEURISTIC if heuristic else output.name
    if parameter:
        subtitle = f"{output.parent.name} · {parameter}"
    end = save_testing_paths(runs, initial_date=initial_date, output=output, forecast=forecast, subtitle=subtitle)

    allocations, probabilities = {}, {}
    for label, (_, monthly) in runs.items():
        group, strategy = (label, HEURISTIC) if heuristic else (forecast, label)
        monthly = monthly.set_index('origin')
        allocations.setdefault(group, {})[strategy] = monthly[list(ASSETS)]
        probabilities.setdefault(group, {})[strategy] = (monthly.bear_signal, monthly.threshold.iloc[0])
    width = max(9, 3.25 * len(runs) + 2)
    save_allocations(allocations, probabilities, end,
                     output / 'figure_2_portfolio_allocations.png', width=width,
                     title='Portfolio Allocations Over Time', subtitle=subtitle)


def save_comparison_figures(cvar_daily, cvar_monthly, mvo_daily, mvo_monthly, *, initial_date, output,
                            heuristic=None):
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
        allocations[forecast] = {CVAR: cvar_weights[list(ASSETS)], MVO: mvo_weights[list(ASSETS)]}
        probabilities[forecast] = {CVAR: (p_bear, 0.5), MVO: (p_bear, 0.5)}
        if heuristic is not None:
            heuristic_returns, weights, bear_signal, threshold = heuristic[forecast]
            if not heuristic_returns.index.equals(validation_dates) or not np.isfinite(heuristic_returns).all():
                raise ValueError(f"{forecast}: heuristic returns must cover the same validation sessions")
            if not weights.index.equals(rebalances) or not bear_signal.index.equals(rebalances):
                raise ValueError(f"{forecast}: heuristic weights and signals must cover the same rebalances")
            daily[HEURISTIC] = heuristic_returns
            allocations[forecast][HEURISTIC] = weights[list(ASSETS)]
            probabilities[forecast][HEURISTIC] = bear_signal, threshold
        initial = pd.DataFrame(0.0, index=[initial_date], columns=daily.columns)
        wealth[forecast] = pd.concat([initial, daily]).add(1).cumprod()

    drawdowns = {forecast: frame.div(frame.cummax()).sub(1) for forecast, frame in wealth.items()}
    save_paths(wealth, output / "figure_1_cumulative_wealth.png",
               ylabel="Portfolio wealth",
               formatter=StrMethodFormatter("${x:.2f}"), reference=1)
    save_allocations(allocations, probabilities, validation_dates[-1],
                     output / "figure_2_portfolio_allocations.png")
    save_paths(drawdowns, output / "figure_3_drawdowns.png",
               ylabel="Drawdown",
               formatter=PercentFormatter(1, decimals=0), reference=0)
