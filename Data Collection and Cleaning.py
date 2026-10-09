
# ============================================================
# CAPSTONE PROJECT: PORTFOLIO DATA COLLECTION AND PREPARATION
# ============================================================
#
#This file extracts daily price data for three asset classes (equities, fixed income, and cryptocurrency) from Yahoo Finance and Coin Metrics. It then aligns the three datasets by date, calculates daily simple returns, and saves the clean prices and returns to CSV files.
# Objective:
# Download daily price data for three asset classes:
#   1. Equities: S&P 500 Total Return Index
#   2. Fixed Income: AGG (aggregate bond market ETF)
#   3. Cryptocurrency: Bitcoin (BTC)
#
# Then:
#   - Align the three datasets by date
#   - Handle differences in trading days
#   - Calculate daily simple returns
#   - Save the clean prices and returns to CSV files
#
# These datasets will eventually be used for:
#   - Mean-Variance Optimization (MVO)
#   - Risk Parity
#   - CVaR Optimization
#   - Ledoit-Wolf covariance estimation
#   - Black-Litterman expected return estimation
#   - GARCH modelling and MPC
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT PYTHON LIBRARIES
# ------------------------------------------------------------

# yfinance downloads historical financial data from Yahoo Finance.
# We will use it to obtain the equity and bond price series.
import yfinance as yf

# pandas is used to organize, clean, manipulate and calculate
# statistics from time-series data.
# A pandas DataFrame is essentially a table with rows and columns.
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
import json


# ------------------------------------------------------------
# STEP 2: DEFINE THE HISTORICAL PERIOD
# ------------------------------------------------------------

# Specify the first date we want to include in our dataset.
# We start in 2015 to obtain a relatively long history of
# equity, bond and cryptocurrency market returns.
start = "2015-01-01"

# Exclude today's potentially incomplete market observations.
# Yahoo's end date is exclusive; use Toronto's current date.
end = datetime.now(ZoneInfo("America/Toronto")).date().isoformat()
output = Path(__file__).resolve().parent


# ------------------------------------------------------------
# STEP 3: DOWNLOAD EQUITY DATA
# ------------------------------------------------------------

# Download historical observations for the S&P 500 Total
# Return Index, whose Yahoo Finance ticker is ^SP500TR.
#
# The Total Return Index includes reinvested dividends,
# unlike the ordinary S&P 500 price index (^GSPC).
#
# Parameters:
#   start: beginning of the historical period
#   end: end of the historical period (exclusive)
#   interval="1d": request daily observations
#   auto_adjust=True: request adjusted price data
#   progress=False: hide the download progress bar
equity_data = yf.download(
    "^SP500TR",
    start=start,
    end=end,
    interval="1d",
    auto_adjust=True,
    progress=False
)

# Verify that the download returned observations.
# If the DataFrame is empty, the ticker may not have
# downloadable historical data for the requested period.
if equity_data.empty:
    raise ValueError("No S&P 500 data was downloaded.")

# Extract the "Close" price column.
#
# yfinance can return a DataFrame with multiple column levels.
# Selecting ["Close"] retrieves the closing-price information.
# .iloc[:, 0] extracts the first column as a pandas Series.
#
# A Series is a single column indexed by date.
equity = equity_data["Close"].iloc[:, 0]

# Rename this Series "Equity".
# This will become its column name in our combined dataset.
equity = equity.rename("Equity")


# ------------------------------------------------------------
# STEP 4: DOWNLOAD FIXED-INCOME DATA
# ------------------------------------------------------------

# Download daily data for AGG.
#
# AGG is the iShares Core U.S. Aggregate Bond ETF.
# It tracks a broad U.S. investment-grade bond index,
# making it an investable proxy for the fixed-income market.
#
# With auto_adjust=True, the Close prices are adjusted
# for distributions and splits. This lets us approximate
# the investor's total return rather than price return alone.
bond_data = yf.download(
    "AGG",
    start=start,
    end=end,
    interval="1d",
    auto_adjust=True,
    progress=False
)

# Check whether the bond download succeeded.
if bond_data.empty:
    raise ValueError("No AGG bond data was downloaded.")

# Extract AGG's adjusted closing prices.
bond = bond_data["Close"].iloc[:, 0]

# Rename this Series "Bond".
bond = bond.rename("Bond")


# ------------------------------------------------------------
# STEP 5: DOWNLOAD BITCOIN DATA
# ------------------------------------------------------------

# The GitHub archive can lag behind the live API. Keep the same
# Coin Metrics PriceUSD series, fetched directly with pagination.
url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?" + urlencode({
    "assets": "btc", "metrics": "PriceUSD", "frequency": "1d",
    "start_time": start, "end_time": end,
    "page_size": 1000, "paging_from": "start",
})
btc_records = []
while url:
    with urlopen(url, timeout=60) as response:
        page = json.load(response)
    btc_records.extend(page["data"])
    url = page.get("next_page_url")
if not btc_records:
    raise ValueError("No Bitcoin observations were returned by Coin Metrics.")
btc_data = pd.DataFrame(btc_records)

# Convert the "time" column from text into datetime values.
#
# utc=True ensures the timestamps are interpreted in UTC
# (Coordinated Universal Time).
#
# This is important because time-series datasets must use
# compatible date/time formats before they can be aligned.
btc_data["time"] = pd.to_datetime(
    btc_data["time"],
    utc=True
)

# Set the date/time column as the DataFrame's index.
#
# Before:
#       time          PriceUSD
#       2015-01-01    ...
#
# After:
#       Index         PriceUSD
#       2015-01-01    ...
#
# Having dates as the index allows pandas to automatically
# match observations from different assets by date.
btc_data = btc_data.set_index("time")

# Extract Bitcoin's USD price column.
#
# astype(float) ensures that prices are stored as numerical
# values suitable for calculations.
btc = pd.to_numeric(btc_data["PriceUSD"], errors="coerce")

# Remove timezone information from the Bitcoin index.
#
# Yahoo Finance data generally uses timezone-naive dates.
# We want all three datasets to use compatible date indexes.
#
# tz_localize(None) removes the UTC timezone label.
# normalize() sets the time component to midnight while
# preserving the calendar date.
btc.index = btc.index.tz_localize(None).normalize()

# Name this Series "BTC" for the combined DataFrame.
btc = btc.rename("BTC")


# ------------------------------------------------------------
# STEP 6: COMBINE THE THREE PRICE SERIES
# ------------------------------------------------------------

# Combine Equity, Bond and BTC into one DataFrame.
#
# axis=1 means combine the Series as COLUMNS.
#
# pandas automatically aligns observations by their date index.
#
# Initially, the result may look like:
#
# Date          Equity     Bond       BTC
# Friday        6000       100        90000
# Saturday      NaN        NaN        92000
# Sunday        NaN        NaN        94000
# Monday        6060       100.5      95000
#
# NaN represents a missing value.
prices = pd.concat(
    [equity, bond, btc],
    axis=1
)

# Remove timezone information from the combined index.
#
# This ensures that the final index has a consistent
# datetime representation.
prices.index = prices.index.tz_localize(None)

# Restrict observations to the requested historical period.
#
# .loc selects rows using the date index.
# Note that this slice includes its endpoint if present,
# whereas yfinance's original end date was exclusive.
prices = prices.sort_index()
prices = prices.loc[(prices.index >= start) & (prices.index < end)]

# Report coverage BEFORE dropping incomplete rows so the limiting
# source is visible instead of silently truncating the portfolio.
coverage = pd.DataFrame({
    asset: {
        "First valid date": prices[asset].first_valid_index(),
        "Last valid date": prices[asset].last_valid_index(),
        "Observations": prices[asset].count(),
    }
    for asset in prices.columns
}).T
print("\nSOURCE COVERAGE BEFORE DATE ALIGNMENT:")
print(coverage.to_string())

# Remove rows containing missing values in ANY column.
#
# This is particularly important because:
#   - BTC trades seven days a week.
#   - Equity and bond markets generally trade Monday-Friday.
#   - Equity/bond markets also close on certain holidays.
#
# dropna() removes dates when any of the three assets
# lacks a price observation.
#
# Importantly, we are removing weekend PRICE observations
# before calculating returns.
#
# Bitcoin's weekend price changes will therefore be included
# in the return between Friday and Monday.
prices = prices.dropna()

# Confirm that the combined dataset contains observations.
if prices.empty:
    raise ValueError("No common dates were found for the assets.")
if not (prices > 0).all().all() or not prices.index.is_unique:
    raise ValueError("Prices must be positive with unique dates.")
staleness = (pd.Timestamp(end) - prices.index.max()).days
if staleness > 7:
    raise ValueError(
        f"Latest common date {prices.index.max().date()} is {staleness} days old. "
        "Check source coverage above. Existing output files were not overwritten."
    )


# ------------------------------------------------------------
# STEP 7: CALCULATE DAILY PORTFOLIO ASSET RETURNS
# ------------------------------------------------------------

# Convert price levels into simple percentage returns.
#
# The mathematical formula is:
#
#          P_t - P_(t-1)
# r_t = -----------------
#              P_(t-1)
#
# This is equivalent to:
#
# r_t = P_t / P_(t-1) - 1
#
# pandas pct_change() performs this calculation for every
# asset and every consecutive observation in the DataFrame.
#
# Example:
#   Monday BTC price = 95,000
#   Previous observation (Friday) = 90,000
#
#   Monday BTC return = 95,000 / 90,000 - 1
#                     = 0.05556 = 5.556%
#
# Because weekend dates were already removed,
# this incorporates Bitcoin's cumulative weekend movement.
returns = prices.pct_change(fill_method=None)

# The first row of returns is NaN because there is no
# previous observation from which to calculate a return.
#
# Remove that first missing row.
returns = returns.dropna()


# ------------------------------------------------------------
# STEP 8: SAVE THE PROCESSED DATASETS
# ------------------------------------------------------------

# Save the aligned price dataset as a CSV file.
#
# CSV files can be opened in Excel or read into Python.
#
# The file will contain:
#   Date | Equity | Bond | BTC
prices.to_csv(output / "portfolio_prices.csv")

# Save the calculated returns to a second CSV file.
#
# This is the most important file for the initial
# portfolio optimization models.
#
# The file will contain:
#   Date | Equity Return | Bond Return | BTC Return
returns.to_csv(output / "portfolio_daily_returns.csv")
coverage.to_csv(output / "source_coverage.csv")


# ------------------------------------------------------------
# STEP 9: VERIFY THE RESULTS
# ------------------------------------------------------------

# Print the first five rows of daily returns.
# This lets us visually inspect the dataset.
print("FIRST FIVE RETURN OBSERVATIONS:")
print(returns.head())

# Print the number of rows and columns.
#
# shape returns a tuple:
#   (number of observations, number of assets)
#
# For example: (2800, 3)
print("\nDATASET DIMENSIONS:")
print(returns.shape)

# Print the first and last dates in the cleaned dataset.
print("\nHISTORICAL PERIOD:")
print("Start:", returns.index.min())
print("End:", returns.index.max())

# Count missing values for each asset.
#
# Ideally, all values should equal zero because we
# already applied dropna() to the price dataset.
print("\nMISSING VALUES:")
print(returns.isna().sum())

# Calculate the correlation between the three return series.
#
# corr() produces a 3x3 correlation matrix.
# This gives a first indication of the relationships
# between equity, bond and Bitcoin returns.
print("\nRETURN CORRELATIONS:")
print(returns.corr())

# Print a confirmation after saving the datasets.
print("\nCSV files saved successfully.")
