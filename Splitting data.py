
import pandas as pd

# This code splits our daily portfolio returns dataset
# (May 2015– May 2026) into three chronological periods:
#
# 1. TRAINING:   2015–2018
# 2. VALIDATION: 2019–2022
# 3. TESTING:    2023–2026
#
# PURPOSE OF EACH DATASET:
#
# TRAINING:
# - Estimate initial model parameters (mean, covariance, etc.).
# - Construct baseline MVO, risk parity, and CVaR models.
# - Fit initial Ledoit-Wolf, Black-Litterman, and GARCH models.
#
# VALIDATION:
# - Evaluate preliminary out-of-sample performance.
# - Compare baseline and enhanced models.
# - Tune model parameters, portfolio constraints, and MPC settings.
# - Select and finalize modelling choices.
#
# TESTING:
# - Evaluate finalized models on out-of-sample observations.
# - Compare single-period and multi-period portfolio performance.
# - Calculate final performance and downside-risk metrics.
# - Do not use final testing performance to tune models.
#
# IMPORTANT:
# When performing walk-forward backtesting, models can be
# re-estimated at each decision date using all permitted
# historical observations available up to that date.
# ============================================================


# ------------------------------------------------------------
# STEP 1: LOAD THE RETURN DATASET
# ------------------------------------------------------------

# Load daily equity, bond, and Bitcoin returns.
# The first column contains dates, which become the index.
# parse_dates=True converts dates into datetime objects.
returns = pd.read_csv(
    "portfolio_daily_returns.csv",
    index_col=0,
    parse_dates=True
)

# Sort observations from oldest to newest.
returns = returns.sort_index()


# ------------------------------------------------------------
# STEP 2: SPLIT THE DATASET CHRONOLOGICALLY
# ------------------------------------------------------------

# TRAINING DATA: January 2015 – December 2018
# Used for initial parameter estimation and model fitting.
train = returns.loc["2015-01-01":"2018-12-31"]


# VALIDATION DATA: January 2019 – December 2022
# Used for model development, comparison, and tuning.
validation = returns.loc["2019-01-01":"2022-12-31"]


# FINAL TESTING DATA: January 2023 – December 2026
# Used only for final out-of-sample performance evaluation.
# The dataset ends at the latest available observation.
test = returns.loc["2023-01-01":"2026-12-31"]


# ------------------------------------------------------------
# STEP 3: VERIFY THE SPLIT
# ------------------------------------------------------------

# Print the number of daily observations in each period.
print("Training observations:", len(train))
print("Validation observations:", len(validation))
print("Testing observations:", len(test))


# Print the actual start and end dates for each dataset.
print("\nTRAINING PERIOD:")
print(train.index.min(), "to", train.index.max())

print("\nVALIDATION PERIOD:")
print(validation.index.min(), "to", validation.index.max())

print("\nFINAL TESTING PERIOD:")
print(test.index.min(), "to", test.index.max())


# ------------------------------------------------------------
# STEP 4: SAVE EACH DATASET
# ------------------------------------------------------------

# Save training observations for initial model fitting.
train.to_csv("portfolio_train.csv")

# Save validation observations for model development.
validation.to_csv("portfolio_validation.csv")

# Save final testing observations for final evaluation.
test.to_csv("portfolio_test.csv")

print("\nTraining, validation, and testing files saved.")
