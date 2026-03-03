# -*- coding: utf-8 -*-
"""
Data Preprocessing
==================
Loads and cleans KCC pest attack data, handles missing values, aggregates to
monthly State-level time series, constructs SIR state variables (S, I, R),
and saves the processed output for downstream phases.

Output: kcc_codebase/processed_data/kcc_monthly_sir.csv
Columns: date, state, S, I, R, rainfall, harvest_area,
         S_norm, I_norm, R_norm
"""

import os
import logging
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "Normalised_KCC_Data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "kcc_monthly_sir.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Step 1.1 – Load and Clean Raw Data
# ===========================================================================
def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, parse dates, standardise string columns."""
    log.info("Loading raw data from %s ...", path)
    df = pd.read_csv(path, parse_dates=["CreatedOn"], low_memory=False)
    log.info("Raw shape: %s", df.shape)

    # Ensure Year / Month columns are present (they already exist, but derive
    # them from the parsed date to guarantee consistency)
    df["Year"] = df["CreatedOn"].dt.year
    df["Month"] = df["CreatedOn"].dt.month

    # Standardise string columns: lowercase + strip whitespace
    for col in ["State Name", "Dist Name", "Pest", "Crop"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # Safety check: drop rows where Count is null
    before = len(df)
    df.dropna(subset=["Count"], inplace=True)
    if len(df) < before:
        log.warning("Dropped %d rows with null Count.", before - len(df))

    log.info("After cleaning: %s rows, %d unique states",
             len(df), df["State Name"].nunique())
    return df


# ===========================================================================
# Step 1.2 – Handle Missing Values
# ===========================================================================
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing Rainfall and Harvest Area values:
      - Rainfall (MM) : monthly–district median → monthly–state median → 0
      - Harvest Area  : district mean over time → state mean over time → 0
    """
    log.info("Imputing missing values ...")
    log.info("  Rainfall missing before: %d / %d",
             df["Rainfall (MM)"].isna().sum(), len(df))
    log.info("  Harvest Area missing before: %d / %d",
             df["Harvest Area"].isna().sum(), len(df))

    # ---------- Rainfall --------------------------------------------------
    # 1) Monthly–district median
    rain_district_monthly = (
        df.groupby(["Year", "Month", "Dist Name"])["Rainfall (MM)"]
        .transform("median")
    )
    df["Rainfall (MM)"] = df["Rainfall (MM)"].fillna(rain_district_monthly)

    # 2) Monthly–state median
    rain_state_monthly = (
        df.groupby(["Year", "Month", "State Name"])["Rainfall (MM)"]
        .transform("median")
    )
    df["Rainfall (MM)"] = df["Rainfall (MM)"].fillna(rain_state_monthly)

    # 3) Fill any remaining with 0
    df["Rainfall (MM)"] = df["Rainfall (MM)"].fillna(0.0)
    log.info("  Rainfall missing after : %d", df["Rainfall (MM)"].isna().sum())

    # ---------- Harvest Area ----------------------------------------------
    # 1) District mean over time
    harv_district = (
        df.groupby("Dist Name")["Harvest Area"].transform("mean")
    )
    df["Harvest Area"] = df["Harvest Area"].fillna(harv_district)

    # 2) State mean over time
    harv_state = (
        df.groupby("State Name")["Harvest Area"].transform("mean")
    )
    df["Harvest Area"] = df["Harvest Area"].fillna(harv_state)

    # 3) Fill any remaining with 0
    df["Harvest Area"] = df["Harvest Area"].fillna(0.0)
    log.info("  Harvest Area missing after: %d",
             df["Harvest Area"].isna().sum())

    return df


# ===========================================================================
# Step 1.3 – Aggregate to Monthly State-Level Time Series
# ===========================================================================
def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to (State × Year-Month) granularity:
      - Count       → sum   (total pest-attacked reports = "I" signal)
      - Rainfall    → mean
      - Harvest Area → mean
    Returns a DataFrame with one row per (state, date) combination.
    """
    log.info("Aggregating to monthly state-level time series ...")

    agg = (
        df.groupby(["Year", "Month", "State Name"])
        .agg(
            I_raw=("Count", "sum"),
            rainfall=("Rainfall (MM)", "mean"),
            harvest_area=("Harvest Area", "mean"),
        )
        .reset_index()
    )

    # Build a proper date column (first day of each month)
    agg["date"] = pd.to_datetime(
        agg["Year"].astype(str) + "-" +
        agg["Month"].astype(str).str.zfill(2) + "-01"
    )
    agg.rename(columns={"State Name": "state"}, inplace=True)
    agg.drop(columns=["Year", "Month"], inplace=True)

    # Sort for clarity
    agg.sort_values(["state", "date"], inplace=True)
    agg.reset_index(drop=True, inplace=True)

    log.info("Aggregated shape: %s", agg.shape)
    log.info("  Date range: %s → %s", agg["date"].min(), agg["date"].max())
    log.info("  States     : %d", agg["state"].nunique())
    log.info("  Time points: %d", agg["date"].nunique())

    return agg


# ===========================================================================
# Step 1.4 – Construct SIR State Variables
# ===========================================================================
def construct_sir(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Derive S, I, R per state per month.

    I(t) = I_raw (pest-attacked crop area counts)
    S(t) = harvest_area - cumulative_I_up_to_t   (approximate susceptible)
    R(t) = harvest_area - S(t) - I(t)            (recovered / ceased)

    Clamp all values to [0, harvest_area] to avoid negative artefacts.
    Also produce normalised versions: S_norm, I_norm, R_norm in [0, 1].
    """
    log.info("Constructing SIR state variables ...")

    records = []
    for state, grp in agg.groupby("state"):
        grp = grp.sort_values("date").copy()

        I = grp["I_raw"].values.astype(float)
        H = grp["harvest_area"].values.astype(float)  # total population proxy

        # Cumulative infections up to (but not including) current month
        cum_I = np.cumsum(I) - I  # = cumulative infected before this month

        S = np.maximum(H - cum_I, 0.0)
        S = np.minimum(S, H)           # S cannot exceed total harvest area

        I_clamped = np.minimum(I, S)   # I cannot exceed current S

        R = np.maximum(H - S - I_clamped, 0.0)

        # Normalise by harvest_area (per state)
        # Avoid divide-by-zero: where H == 0, keep raw proportions as 0
        H_safe = np.where(H > 0, H, 1.0)
        S_norm = S / H_safe
        I_norm = I_clamped / H_safe
        R_norm = R / H_safe

        grp = grp.copy()
        grp["S"] = S
        grp["I"] = I_clamped
        grp["R"] = R
        grp["S_norm"] = S_norm
        grp["I_norm"] = I_norm
        grp["R_norm"] = R_norm
        records.append(grp)

    result = pd.concat(records, ignore_index=True)
    result.sort_values(["state", "date"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    log.info("SIR construction complete. Sample statistics:")
    log.info("  I_norm  – mean: %.4f  max: %.4f",
             result["I_norm"].mean(), result["I_norm"].max())
    log.info("  S_norm  – mean: %.4f  min: %.4f",
             result["S_norm"].mean(), result["S_norm"].min())
    log.info("  R_norm  – mean: %.4f  max: %.4f",
             result["R_norm"].mean(), result["R_norm"].max())

    return result


# ===========================================================================
# Step 1.5 – Save Output
# ===========================================================================
def save_output(df: pd.DataFrame, path: str) -> None:
    """Save the processed monthly SIR DataFrame to CSV."""
    cols = ["date", "state", "S", "I", "R",
            "S_norm", "I_norm", "R_norm",
            "rainfall", "harvest_area", "I_raw"]
    df[cols].to_csv(path, index=False)
    log.info("Saved %d rows to %s", len(df), path)


# ===========================================================================
# Step 1.6 – Quick Validation
# ===========================================================================
def validate(df: pd.DataFrame) -> None:
    """Run sanity checks on the processed output."""
    log.info("Running validation checks ...")

    # Check expected shape:  T × N = ~74 months × 31 states
    T = df["date"].nunique()
    N = df["state"].nunique()
    log.info("  T (months) = %d,  N (states) = %d,  T×N = %d (expected ~%d)",
             T, N, len(df), T * N)

    # Every (date, state) pair should be unique
    dupes = df.duplicated(["date", "state"]).sum()
    assert dupes == 0, f"Duplicate (date, state) pairs found: {dupes}"
    log.info("  No duplicate (date, state) pairs – OK")

    # SIR values should be non-negative
    for col in ["S", "I", "R", "S_norm", "I_norm", "R_norm"]:
        neg = (df[col] < -1e-6).sum()
        assert neg == 0, f"Negative values in {col}: {neg}"
    log.info("  All SIR values non-negative – OK")

    # Normalised values should be in [0, 1]  (with tiny float tolerance)
    for col in ["S_norm", "I_norm", "R_norm"]:
        over = (df[col] > 1.0 + 1e-6).sum()
        assert over == 0, f"Values > 1 in {col}: {over}"
    log.info("  All normalised SIR values in [0, 1] – OK")

    # No missing values in key columns
    for col in ["date", "state", "S", "I", "R", "S_norm", "I_norm", "R_norm"]:
        nulls = df[col].isna().sum()
        assert nulls == 0, f"Null values in column {col}: {nulls}"
    log.info("  No null values in key columns – OK")

    log.info("All validation checks passed.")


# ===========================================================================
# Main
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("Data Preprocessing – START")
    log.info("=" * 60)

    # 1.1 Load & clean
    df = load_and_clean(RAW_DATA_PATH)

    # 1.2 Impute missing values
    df = impute_missing(df)

    # 1.3 Aggregate to monthly state-level
    agg = aggregate_monthly(df)

    # 1.4 Construct SIR variables
    sir = construct_sir(agg)

    # 1.5 Save
    save_output(sir, OUTPUT_PATH)

    # 1.6 Validate
    validate(sir)

    log.info("=" * 60)
    log.info("Data Preprocessing COMPLETE  →  %s", OUTPUT_PATH)
    log.info("=" * 60)

    return sir


if __name__ == "__main__":
    result = main()
    print("\nFinal output preview:")
    print(result.head(10).to_string(index=False))
    print("\nShape:", result.shape)
    print("\nStates covered:")
    print(sorted(result["state"].unique()))
