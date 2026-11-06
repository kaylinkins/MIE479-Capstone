
# ============================================================
# CAPSTONE PROJECT: EXPLORATORY DATA ANALYSIS
# Equities, Fixed Income, and Bitcoin
# ============================================================

#This file runs some descriptive statistics and visualizations on the daily simple returns of three asset classes: equities, fixed income, and Bitcoin. It saves all tables and figures to a folder for later reference.
# These stats and figures are useful for informing our approach in our reports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "datasets"
output = PROJECT_ROOT / "outputs" / "descriptive"
output.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# STEP 1: LOAD AND PREPARE DATA
# ------------------------------------------------------------

# Read the CSV containing daily simple returns.
# The first column contains dates, so use it as the index.
returns = pd.read_csv(
    DATA_DIR / "portfolio_daily_returns.csv",
    index_col=0,
    parse_dates=True
)

# Sort the rows chronologically, from earliest to latest.
returns = returns.sort_index()

# Keep only our three asset classes in a consistent order.
assets = ["Equity", "Bond", "BTC"]
returns = returns[assets]

# Ensure every return is numeric.
returns = returns.apply(pd.to_numeric, errors="coerce")

# Remove any rows with missing returns.
returns = returns.dropna()

# Verify that enough observations exist for rolling analysis.
if len(returns) < 61:
    raise ValueError(
        "At least 61 return observations are required."
    )

# Check that all simple returns are mathematically valid.
# A simple return cannot be less than -100%.
if (returns < -1).any().any():
    raise ValueError("Invalid returns below -100 found.")

# Approximately 252 equity-market trading days per year.
TRADING_DAYS = 252

# Window for rolling volatility and correlation analysis.
ROLLING_WINDOW = 60

# Confidence level for Value at Risk and CVaR.
ALPHA = 0.95

# Create a folder where all figures and tables are saved.
# Figures and tables use the output folder defined above.

# Use a consistent plotting style.
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25
})

# Create a function to save and display each figure.
def save_figure(fig, filename):

    # Ensure labels fit without overlapping.
    fig.tight_layout()

    # Save the figure at high resolution.
    fig.savefig(
        output / filename,
        dpi=300,
        bbox_inches="tight"
    )

    # Display the figure in Python.
    plt.show()

    # Close the figure to free memory.
    plt.close(fig)


# Create a function to display DataFrames as tables.
def plot_table(df, title, filename):

    # Choose figure height based on number of table rows.
    height = max(2.5, 0.42 * len(df) + 1.4)

    fig, ax = plt.subplots(
        figsize=(11, height)
    )

    # Hide the axes because this is a table, not a graph.
    ax.axis("off")

    # Give the table a descriptive title.
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    # Create the visual table from the DataFrame.
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        rowLoc="left",
        loc="center",
        bbox=[0, 0, 1, 0.92]
    )

    # Set a readable font size.
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    # Format table borders, header and alternating rows.
    for (row, col), cell in table.get_celld().items():

        cell.set_edgecolor("#DDDDDD")

        if row == 0:
            cell.set_facecolor("#DCE8F3")
            cell.set_text_props(weight="bold")

        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    # Save the visual table as a PNG.
    fig.savefig(
        output / filename,
        dpi=300,
        bbox_inches="tight"
    )

    # Display the table.
    plt.show()
    plt.close(fig)


# ------------------------------------------------------------
# STEP 2: CALCULATE DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

# Store all calculated statistics in a dictionary.
summary = {}

# Loop through equity, bond and Bitcoin returns separately.
for asset in assets:

    # Extract one asset's daily return observations.
    r = returns[asset]

    # Calculate mean daily simple return.
    daily_mean = r.mean()

    # Annualize arithmetic mean by multiplying by 252.
    annual_mean = daily_mean * TRADING_DAYS

    # Calculate sample daily standard deviation.
    daily_vol = r.std()

    # Annualize volatility using square-root-of-time.
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS)

    # Calculate distribution skewness.
    skewness = r.skew()

    # Calculate excess kurtosis.
    # Normal distributions have excess kurtosis of zero.
    kurtosis = r.kurt()

    # Identify worst and best observed daily returns.
    minimum = r.min()
    maximum = r.max()

    # Calculate the 5th and 95th return percentiles.
    p5 = r.quantile(0.05)
    p95 = r.quantile(0.95)

    # Daily 95% historical VaR, expressed as a loss.
    var95 = -r.quantile(1 - ALPHA)

    # Historical CVaR: average loss among observations
    # at or below the 5th-percentile return threshold.
    tail_returns = r[r <= r.quantile(1 - ALPHA)]
    cvar95 = -tail_returns.mean()

    # Construct the wealth of $1 invested in the asset.
    wealth = (1 + r).cumprod()

    # Track the highest wealth achieved to each date.
    peak = wealth.cummax()

    # Calculate the drawdown relative to the prior peak.
    drawdown = wealth / peak - 1

    # Worst peak-to-trough decline.
    maximum_drawdown = drawdown.min()

    # Jarque-Bera test for normality.
    jb_result = stats.jarque_bera(r)

    # Store all statistics for this asset.
    summary[asset] = {
        "Observations": len(r),
        "Mean Daily Return (%)": daily_mean * 100,
        "Annualized Mean Return (%)": annual_mean * 100,
        "Daily Volatility (%)": daily_vol * 100,
        "Annualized Volatility (%)": annual_vol * 100,
        "Minimum Daily Return (%)": minimum * 100,
        "Maximum Daily Return (%)": maximum * 100,
        "Skewness": skewness,
        "Excess Kurtosis": kurtosis,
        "5th Percentile Return (%)": p5 * 100,
        "95th Percentile Return (%)": p95 * 100,
        "95% Historical VaR (%)": var95 * 100,
        "95% Historical CVaR (%)": cvar95 * 100,
        "Maximum Drawdown (%)": maximum_drawdown * 100,
        "Jarque-Bera Statistic": jb_result.statistic,
        "Jarque-Bera P-value": jb_result.pvalue
    }


# Convert the dictionary into a table.
# Rows are statistics, columns are the three assets.
summary_df = pd.DataFrame(summary)

# Print numerical results in the terminal.
print("\nTABLE 1: DESCRIPTIVE STATISTICS")
print(summary_df.round(4).to_string())

# Save the full-precision results as a CSV.
summary_df.to_csv(output / "table1_summary_statistics.csv")

# Format the numbers for display in a figure.
summary_display = summary_df.apply(
    lambda col: col.map(lambda x: f"{x:,.4f}")
)

# Display the number of observations as integers.
summary_display.loc["Observations"] = (
    summary_df.loc["Observations"]
    .map(lambda x: f"{int(x):,}")
)

# Generate and save Table 1 as an image.
plot_table(
    summary_display,
    "Table 1: Descriptive Statistics",
    "table1_summary_statistics.png"
)


# ------------------------------------------------------------
# STEP 3: CORRELATION AND COVARIANCE TABLES
# ------------------------------------------------------------

# Calculate the Pearson correlation matrix.
correlation = returns.corr()

# Calculate the daily sample covariance matrix.
covariance = returns.cov()

# Print the two matrices in the terminal.
print("\nTABLE 2: CORRELATION MATRIX")
print(correlation.round(4))

print("\nTABLE 3: DAILY COVARIANCE MATRIX")
print(covariance.round(8))

# Save numerical matrices for future modelling.
correlation.to_csv(output / "table2_correlation.csv")
covariance.to_csv(output / "table3_covariance.csv")

# Create formatted copies for visual tables.
correlation_display = correlation.apply(
    lambda col: col.map(lambda x: f"{x:.4f}")
)

covariance_display = covariance.apply(
    lambda col: col.map(lambda x: f"{x:.8f}")
)

# Plot and save the correlation matrix table.
plot_table(
    correlation_display,
    "Table 2: Pearson Correlation Matrix",
    "table2_correlation.png"
)

# Plot and save the covariance matrix table.
plot_table(
    covariance_display,
    "Table 3: Daily Return Covariance Matrix",
    "table3_covariance.png"
)


# ------------------------------------------------------------
# FIGURE 1: CUMULATIVE PERFORMANCE
# ------------------------------------------------------------

# Calculate growth of $1 invested in each asset.
#
# Example:
# A return of 5% changes wealth from $1 to $1.05.
# A subsequent return of -2% changes it to $1.029.
wealth = np.log((1 + returns).cumprod())

# Plot all three assets on the same chart.
fig, ax = plt.subplots()

for asset in assets:
    ax.plot(
        wealth.index,
        wealth[asset],
        label=asset,
        linewidth=1.5
    )

ax.set_title("Figure 1: Growth of $1 Invested")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()

save_figure(fig, "figure1_cumulative_returns.png")


# ------------------------------------------------------------
# FIGURE 2: RETURN HISTOGRAMS AND NORMAL CURVES
# ------------------------------------------------------------

# Create three panels, one for each asset class.
fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5)
)

for i, asset in enumerate(assets):

    # Convert returns from decimals into percentages.
    r = returns[asset].dropna() * 100

    # Plot the empirical histogram.
    axes[i].hist(
        r,
        bins=60,
        density=True,
        alpha=0.65,
        label="Observed returns"
    )

    # Estimate mean and standard deviation of returns.
    mu = r.mean()
    sigma = r.std()

    # Generate x-values spanning the return distribution.
    x = np.linspace(
        r.min(),
        r.max(),
        500
    )

    # Calculate the normal probability density function
    # using the observed mean and standard deviation.
    normal_pdf = stats.norm.pdf(
        x,
        loc=mu,
        scale=sigma
    )

    # Overlay the fitted normal curve.
    axes[i].plot(
        x,
        normal_pdf,
        label="Fitted normal",
        linewidth=2
    )

    axes[i].set_title(asset)
    axes[i].set_xlabel("Daily Return (%)")
    axes[i].set_ylabel("Density")
    axes[i].legend(fontsize=8)

fig.suptitle(
    "Figure 2: Return Distributions vs Normality",
    fontsize=14
)

save_figure(fig, "figure2_return_histograms.png")


# ------------------------------------------------------------
# FIGURE 3: NORMAL Q-Q PLOTS
# ------------------------------------------------------------

# Q-Q plots compare empirical return quantiles against
# theoretical quantiles of a normal distribution.
#
# If the points follow the reference line, the distribution
# is approximately normal.
#
# Large deviations at the ends indicate unusual tail
# behaviour relative to a normal distribution.

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5)
)

for i, asset in enumerate(assets):

    # Express daily returns as percentages.
    r = returns[asset].dropna() * 100

    # Construct the theoretical normal Q-Q plot.
    stats.probplot(
        r,
        dist="norm",
        plot=axes[i]
    )

    axes[i].set_title(asset)
    axes[i].set_xlabel("Theoretical Quantiles")
    axes[i].set_ylabel("Sample Quantiles (%)")

fig.suptitle(
    "Figure 3: Normal Q-Q Plots",
    fontsize=14
)

save_figure(fig, "figure3_qq_plots.png")


# ------------------------------------------------------------
# FIGURE 4: DAILY RETURNS OVER TIME
# ------------------------------------------------------------

# Separate panels make it easier to see each asset's
# volatility because BTC is usually much more volatile.
fig, axes = plt.subplots(
    3, 1,
    figsize=(13, 10),
    sharex=True
)

for i, asset in enumerate(assets):

    # Plot daily percentage returns through time.
    axes[i].plot(
        returns.index,
        returns[asset] * 100,
        linewidth=0.65
    )

    # Horizontal line representing zero return.
    axes[i].axhline(
        y=0,
        color="black",
        linewidth=0.7
    )

    axes[i].set_title(asset)
    axes[i].set_ylabel("Daily Return (%)")

axes[-1].set_xlabel("Date")

fig.suptitle(
    "Figure 4: Daily Asset Returns Over Time",
    fontsize=14
)

save_figure(fig, "figure4_daily_returns.png")

# Calculate cumulative wealth for each asset.
wealth = (1 + returns).cumprod()

# Calculate number of calendar years.
years = (
    returns.index[-1] - returns.index[0]
).days / 365.25

# Calculate compound annual growth rates.
cagr = wealth.iloc[-1] ** (1 / years) - 1

print("Compound Annual Growth Rates:")
print((cagr * 100).round(2))


# ------------------------------------------------------------
# FIGURE 5: ROLLING ANNUALIZED VOLATILITY
# ------------------------------------------------------------

# rolling(60) uses the current observation and the
# previous 59 common trading-date observations.
#
# std() calculates the sample standard deviation
# within each rolling window.
#
# Multiply by sqrt(252) for conventional annualization.
#
# Multiply by 100 to express volatility as a percentage.
rolling_vol = (
    returns.rolling(ROLLING_WINDOW).std()
    * np.sqrt(TRADING_DAYS)
    * 100
)

# Plot each asset's rolling volatility.
fig, ax = plt.subplots()

for asset in assets:
    ax.plot(
        rolling_vol.index,
        rolling_vol[asset],
        label=asset
    )

ax.set_title(
    "Figure 5: 60-Observation Rolling Annualized Volatility"
)
ax.set_xlabel("Date")
ax.set_ylabel("Annualized Volatility (%)")
ax.legend()

save_figure(fig, "figure5_rolling_volatility.png")


# ------------------------------------------------------------
# FIGURE 6: ROLLING CORRELATIONS
# ------------------------------------------------------------

# Calculate 60-observation rolling correlations
# between each pair of asset classes.

# Bitcoin versus equities.
btc_equity = (
    returns["BTC"].rolling(ROLLING_WINDOW)
    .corr(returns["Equity"])
)

# Bitcoin versus bonds.
btc_bond = (
    returns["BTC"].rolling(ROLLING_WINDOW)
    .corr(returns["Bond"])
)

# Equities versus bonds.
equity_bond = (
    returns["Equity"].rolling(ROLLING_WINDOW)
    .corr(returns["Bond"])
)

fig, ax = plt.subplots()

ax.plot(
    btc_equity.index,
    btc_equity,
    label="BTC–Equity"
)

ax.plot(
    btc_bond.index,
    btc_bond,
    label="BTC–Bond"
)

ax.plot(
    equity_bond.index,
    equity_bond,
    label="Equity–Bond"
)

# Show zero correlation as a reference line.
ax.axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=0.8
)

ax.set_title(
    "Figure 6: 60-Observation Rolling Correlations"
)
ax.set_xlabel("Date")
ax.set_ylabel("Pearson Correlation")
ax.set_ylim(-1, 1)
ax.legend()

save_figure(fig, "figure6_rolling_correlations.png")


# ------------------------------------------------------------
# FIGURE 7: HISTORICAL DRAWDOWNS
# ------------------------------------------------------------

# Calculate each asset's highest cumulative wealth to date.
running_peak = wealth.cummax()

# Calculate drawdowns as a percentage of prior peak wealth.
drawdowns = (wealth / running_peak - 1) * 100

fig, axes = plt.subplots(
    3, 1,
    figsize=(13, 10),
    sharex=True
)

for i, asset in enumerate(assets):

    axes[i].fill_between(
        drawdowns.index,
        drawdowns[asset],
        0,
        alpha=0.5
    )

    axes[i].set_title(asset)
    axes[i].set_ylabel("Drawdown (%)")

axes[-1].set_xlabel("Date")

fig.suptitle(
    "Figure 7: Historical Asset Drawdowns",
    fontsize=14
)

save_figure(fig, "figure7_drawdowns.png")


# ------------------------------------------------------------
# FIGURE 8: AUTOCORRELATION OF SQUARED RETURNS
# ------------------------------------------------------------

# Squared returns can serve as a simple proxy for
# the magnitude of market volatility.
#
# If squared returns are correlated over time,
# that is consistent with volatility clustering.
#
# This can motivate investigating GARCH models.

MAX_LAG = 30
ACF_CONFIDENCE = 0.95

# Shuffle observed magnitudes to preserve heavy tails while removing
# temporal ordering. Under the null, observations must be exchangeable.
# The 95th percentile of max(abs(ACF)) gives simultaneous reference
# bounds across lags 1-30 within each asset, not across all three assets.
# These are null reference bands, not intervals around the true ACF.
ACF_PERMUTATIONS = 10_000
ACF_RANDOM_SEED = 20261010
acf_results = []


def lagged_pearson_correlations(values, max_lag):
    """Compute the same lag-specific Pearson correlations as Series.autocorr."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= max_lag or not np.isfinite(values).all() or np.std(values) == 0:
        raise ValueError("ACF needs finite, nonconstant data longer than max_lag.")
    # Center globally for numerical stability; segment means are accounted
    # for below. FFT supplies all lagged cross-products efficiently.
    values = (values - values.mean()) / values.std()
    lags = np.arange(1, max_lag + 1)
    counts = n - lags
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(values, n=fft_size)
    products = np.fft.irfft(spectrum * spectrum.conj(), n=fft_size)[lags]
    sums = np.concatenate(([0.0], np.cumsum(values)))
    squares = np.concatenate(([0.0], np.cumsum(values ** 2)))
    left_sum, right_sum = sums[counts], sums[n] - sums[lags]
    left_ss = squares[counts] - left_sum ** 2 / counts
    right_ss = squares[n] - squares[lags] - right_sum ** 2 / counts
    if (left_ss <= 0).any() or (right_ss <= 0).any():
        raise ValueError("A lagged segment has zero variance.")
    return (products - left_sum * right_sum / counts) / np.sqrt(left_ss * right_ss)


def permutation_acf_band(values, rng):
    maxima = np.empty(ACF_PERMUTATIONS)
    for permutation in range(ACF_PERMUTATIONS):
        shuffled = rng.permutation(values)
        maxima[permutation] = np.max(np.abs(lagged_pearson_correlations(shuffled, MAX_LAG)))
    bound = np.quantile(maxima, ACF_CONFIDENCE, method="higher")
    return bound, maxima

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 10),
    sharey=True
)

for i, asset in enumerate(assets):

    # Square the daily return observations.
    squared_returns = returns[asset] ** 2

    # Calculate correlation between squared returns
    # separated by each lag from 1 to 30.
    autocorrelations = lagged_pearson_correlations(squared_returns.to_numpy(), MAX_LAG)
    rng = np.random.default_rng(ACF_RANDOM_SEED + i)
    acf_bound, permutation_maxima = permutation_acf_band(squared_returns.to_numpy(), rng)
    observed_max = np.max(np.abs(autocorrelations))
    global_p = (1 + np.count_nonzero(permutation_maxima >= observed_max)) / (ACF_PERMUTATIONS + 1)
    print(f"{asset}: 95% simultaneous permutation bounds +/- {acf_bound:.4f}; "
          f"global permutation p-value = {global_p:.6f}")
    for lag, correlation in enumerate(autocorrelations, start=1):
        acf_results.append({
            "Asset": asset, "Lag": lag, "Autocorrelation": correlation,
            "Lower bound": -acf_bound, "Upper bound": acf_bound,
            "Outside simultaneous band": abs(correlation) > acf_bound,
            "Global permutation p-value": global_p,
            "Permutations": ACF_PERMUTATIONS, "Seed": ACF_RANDOM_SEED + i,
            "Observations": len(squared_returns), "Confidence": ACF_CONFIDENCE,
        })

    # Plot correlation at each lag.
    axes[i].bar(
        range(1, MAX_LAG + 1),
        autocorrelations
    )

    axes[i].axhspan(
        -acf_bound, acf_bound,
        color="orange", alpha=0.18,
        label=f"95% simultaneous permutation band (+/- {acf_bound:.3f})",
    )
    for bound in (-acf_bound, acf_bound):
        axes[i].axhline(bound, color="darkorange", linestyle="--", linewidth=1)
    axes[i].legend(loc="upper right", fontsize=8)

    # Show zero correlation.
    axes[i].axhline(
        0,
        color="black",
        linewidth=0.8
    )

    axes[i].set_title(asset)
    axes[i].set_ylabel("Autocorrelation")
    axes[i].set_xlabel("Lag (observations)")

# Shared limits include all bars and both confidence boundaries;
# Matplotlib adds a common margin automatically through sharey=True.

fig.suptitle(
    "Figure 8: Autocorrelation of Squared Returns",
    fontsize=14
)

pd.DataFrame(acf_results).to_csv(output / "figure8_squared_return_acf.csv", index=False)
save_figure(fig, "figure8_squared_return_acf.png")


# ------------------------------------------------------------
# FINAL STEP: CONFIRM OUTPUTS
# ------------------------------------------------------------

print("\nANALYSIS COMPLETE")

print("\nSample period:")
print(returns.index.min(), "to", returns.index.max())

print("\nNumber of aligned observations:")
print(len(returns))

print("\nAll tables and figures saved to:")
print(output.resolve())
