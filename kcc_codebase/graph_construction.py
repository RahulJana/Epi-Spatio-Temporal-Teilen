# -*- coding: utf-8 -*-
"""
graph_construction.py
=====================
Builds three types of spatial adjacency matrices for the 31 Indian states
present in the KCC dataset:

  2.1  Geographic (binary border adjacency)
  2.2  Distance-based soft adjacency  (Gaussian kernel on haversine distance)
  2.3  Correlation-based adjacency   (Pearson r on I(t) time series)

Output files saved to kcc_codebase/processed_data/:
  - neighbor_adjacency_matrix.csv          (binary border, row-normalised)
  - neighbor_adjacency_matrix_soft.csv     (Gaussian distance kernel)
  - neighbor_adjacency_matrix_corr.csv     (thresholded Pearson correlation)

Usage
-----
    python kcc_codebase/graph_construction.py
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
SIR_PATH = os.path.join(BASE_DIR, "processed_data", "kcc_monthly_sir.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
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

# ---------------------------------------------------------------------------
# State index (alphabetical — matches kcc_monthly_sir.csv)
# ---------------------------------------------------------------------------
STATES = [
    "andhra pradesh",  # 0
    "arunachal pradesh",  # 1
    "assam",  # 2
    "bihar",  # 3
    "chandigarh",  # 4
    "chhattisgarh",  # 5
    "delhi",  # 6
    "goa",  # 7
    "gujarat",  # 8
    "haryana",  # 9
    "himachal pradesh",  # 10
    "jammu and kashmir",  # 11
    "karnataka",         # 12
    "kerala",            # 13
    "madhya pradesh",    # 14
    "maharashtra",       # 15
    "manipur",           # 16
    "meghalaya",         # 17
    "mizoram",           # 18
    "nagaland",          # 19
    "odisha",            # 20
    "puducherry",        # 21
    "punjab",            # 22
    "rajasthan",         # 23
    "sikkim",            # 24
    "tamilnadu",         # 25
    "telangana",         # 26
    "tripura",           # 27
    "uttar pradesh",     # 28
    "uttarakhand",       # 29
    "west bengal",       # 30
]
N = len(STATES)
STATE_IDX = {s: i for i, s in enumerate(STATES)}

# ---------------------------------------------------------------------------
# Geographic border adjacency (hand-coded from India political map)
# ---------------------------------------------------------------------------
BORDER_PAIRS = [
    # andhra pradesh (0)
    (0, 5), (0, 12), (0, 20), (0, 25), (0, 26),
    # arunachal pradesh (1)
    (1, 2), (1, 19),
    # assam (2)
    (2, 16), (2, 17), (2, 18), (2, 19), (2, 27), (2, 30),
    # bihar (3)
    (3, 20), (3, 28), (3, 30),
    # chandigarh (4)
    (4, 9), (4, 10), (4, 22),
    # chhattisgarh (5)
    (5, 14), (5, 15), (5, 20), (5, 26), (5, 28),
    # delhi (6)
    (6, 9), (6, 28),
    # goa (7)
    (7, 12), (7, 15),
    # gujarat (8)
    (8, 14), (8, 15), (8, 23),
    # haryana (9)
    (9, 10), (9, 22), (9, 23), (9, 28),
    # himachal pradesh (10)
    (10, 11), (10, 22), (10, 29),
    # jammu and kashmir (11)
    (11, 22),
    # karnataka (12)
    (12, 13), (12, 15), (12, 25), (12, 26),
    # kerala (13)
    (13, 21), (13, 25),
    # madhya pradesh (14)
    (14, 15), (14, 23), (14, 28),
    # maharashtra (15)
    (15, 26),
    # manipur (16)
    (16, 18), (16, 19),
    # meghalaya (17)
    (17, 30),
    # mizoram (18)
    (18, 27),
    # odisha (20)
    (20, 30),
    # puducherry (21)
    (21, 25),
    # punjab (22)
    (22, 23),
    # rajasthan (23)
    (23, 28),
    # sikkim (24)
    (24, 30),
    # tamilnadu (25)
    # telangana (26): already covered
    # uttar pradesh (28)
    (28, 29),
]


# ===========================================================================
# 2.1 – Geographic Adjacency
# ===========================================================================
def build_geo_adjacency() -> pd.DataFrame:
    """Build binary border adjacency matrix, then row-normalise."""
    log.info("Building geographic border adjacency matrix ...")
    A = np.zeros((N, N), dtype=np.float32)
    for i, j in BORDER_PAIRS:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Self-loops
    np.fill_diagonal(A, 1.0)

    # Row-normalise  (each row sums to 1)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A / row_sums

    log.info("  Edges (excl. self-loops): %d", int(A.sum() - N))
    df = pd.DataFrame(A_norm, index=STATES, columns=STATES)
    return df


# ===========================================================================
# 2.2 – Distance-Based Soft Adjacency
# ===========================================================================
def _haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute pairwise haversine distance (km) between N points."""
    R = 6371.0
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r[:, None] - lats_r[None, :]
    dlon = lons_r[:, None] - lons_r[None, :]
    a = np.sin(dlat / 2) ** 2 + (
        np.cos(lats_r[:, None]) * np.cos(lats_r[None, :]) *
        np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_soft_adjacency(raw_path: str, sigma_km: float = 500.0, threshold: float = 0.1) -> pd.DataFrame:
    """
    Build Gaussian-kernel distance-based adjacency.

    A[i,j] = exp(-d(i,j)^2 / sigma^2)  if > threshold, else 0.
    """
    log.info("Building distance-based soft adjacency (sigma=%.0f km) ...", sigma_km)

    df_raw = pd.read_csv(raw_path, low_memory=False,
                         usecols=["State Name", "lattitude", "longitude"])
    df_raw["State Name"] = df_raw["State Name"].str.lower().str.strip()
    df_raw = df_raw[df_raw["State Name"].isin(STATES)]

    centroids = (
        df_raw.groupby("State Name")[["lattitude", "longitude"]]
        .mean()
        .rename(columns={"lattitude": "Latitude", "longitude": "Longitude"})
        .reindex(STATES)
    )
    # Fill any state with no lat/lon data using rough central India fallback
    centroids.fillna({"Latitude": 22.0, "Longitude": 78.0}, inplace=True)

    lats = centroids["Latitude"].values.astype(float)
    lons = centroids["Longitude"].values.astype(float)

    dist = _haversine_matrix(lats, lons)
    A = np.exp(-(dist ** 2) / (sigma_km ** 2)).astype(np.float32)
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 1.0)

    # Row-normalise
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A / row_sums

    log.info("  Non-zero off-diagonal entries: %d / %d",
             int((A != 0).sum() - N), N * (N - 1))
    df = pd.DataFrame(A_norm, index=STATES, columns=STATES)
    return df


# ===========================================================================
# 2.3 – Correlation-Based Adjacency
# ===========================================================================
def build_corr_adjacency(sir_path: str, threshold: float = 0.5) -> pd.DataFrame:
    """
    Build Pearson-correlation adjacency on I(t) time series.
    Threshold r > threshold → binary adjacency.
    """
    log.info("Building correlation-based adjacency (threshold=%.2f) ...", threshold)

    df = pd.read_csv(sir_path, parse_dates=["date"])
    pivot = df.pivot(index="date", columns="state", values="I_norm")[STATES]

    corr = pivot.corr().values.astype(np.float32)
    A = (corr > threshold).astype(np.float32)
    np.fill_diagonal(A, 1.0)

    # Row-normalise
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A / row_sums

    log.info("  Correlated pairs (r>%.2f, excl. self): %d",
             threshold, int(A.sum() - N))
    df_out = pd.DataFrame(A_norm, index=STATES, columns=STATES)
    return df_out


# ===========================================================================
# Main
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("PHASE 2 – Graph Construction")
    log.info("=" * 60)

    # 2.1 Geographic
    geo = build_geo_adjacency()
    geo_path = os.path.join(OUTPUT_DIR, "neighbor_adjacency_matrix.csv")
    geo.to_csv(geo_path)
    log.info("Saved geographic adjacency → %s", geo_path)

    # 2.2 Soft distance
    soft = build_soft_adjacency(RAW_DATA_PATH)
    soft_path = os.path.join(OUTPUT_DIR, "neighbor_adjacency_matrix_soft.csv")
    soft.to_csv(soft_path)
    log.info("Saved soft adjacency       → %s", soft_path)

    # 2.3 Correlation
    corr = build_corr_adjacency(SIR_PATH)
    corr_path = os.path.join(OUTPUT_DIR, "neighbor_adjacency_matrix_corr.csv")
    corr.to_csv(corr_path)
    log.info("Saved correlation adjacency→ %s", corr_path)

    log.info("Phase 2 complete.")
    return geo, soft, corr


if __name__ == "__main__":
    geo, soft, corr = main()
    print("\nGeographic adjacency (first 5×5):")
    print(geo.iloc[:5, :5].round(3).to_string())
