# -*- coding: utf-8 -*-
"""
dataset_builder.py
==================
Builds PyTorch DataLoaders for the KCC SIR dataset.

Steps
-----
3.1  Load processed SIR CSV; expand to full monthly calendar (2013-01 → 2020-07)
     filling data-absent months with I=0 and forward-filled S/R/rain/harvest_area.
3.2  Build sliding windows  (obs_len + pre_len months, stride 1).
3.3  Temporal train/val/test split – windows whose prediction horizon falls in:
       train: pre_len months end ≤ 2017-09  (avoids 2017-Q4 gap & 2019 gap)
       val  : prediction months in 2018
       test : prediction months in 2020
3.4  Construct PyTorch Datasets & DataLoaders yielding (x_SIR, yd, yi).
3.5  Build auxdata dict (prov_pop, imax, imin, date_list) and save as JSON.

Tensor conventions (matching existing Train.py)
-----------------------------------------------
x_SIR : (B, window_size, N, 3)  – [S_norm, I_norm, R_norm] per state
yd    : same as x_SIR (monthly data; used when params['daily']=True → not used)
yi    : same as x_SIR (the label when params['daily']=False)

Shape: B=batch, window_size=obs_len+pre_len, N=31 states, F=3 features.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
# State list (must match kcc_monthly_sir.csv column order)
# ---------------------------------------------------------------------------
STATES = sorted([
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chandigarh",
    "chhattisgarh", "delhi", "goa", "gujarat", "haryana", "himachal pradesh",
    "jammu and kashmir", "karnataka", "kerala", "madhya pradesh", "maharashtra",
    "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "puducherry",
    "punjab", "rajasthan", "sikkim", "tamilnadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal",
])
N_STATES = len(STATES)  # 31


# ===========================================================================
# 3.1 – Load & build complete monthly panel
# ===========================================================================
def load_complete_panel(sir_path: str) -> tuple[np.ndarray, pd.DatetimeIndex, list]:
    """
    Returns
    -------
    tensor : np.ndarray  shape (T, N, 3) – [S_norm, I_norm, R_norm]
    dates  : pd.DatetimeIndex  length T – monthly timestamps
    states : list of str       length N
    """
    log.info("Loading SIR CSV from %s ...", sir_path)
    df = pd.read_csv(sir_path, parse_dates=["date"])

    # Align states list with what's in the file
    states_in_file = sorted(df["state"].unique().tolist())
    if states_in_file != STATES:
        log.warning("State list mismatch – using file's states.")
    states = states_in_file
    n = len(states)

    # Full monthly date range (including months that had zero KCC reports)
    date_min = df["date"].min()
    date_max = df["date"].max()
    full_dates = pd.date_range(date_min, date_max, freq="MS")
    T = len(full_dates)
    log.info("  Full monthly range: %s → %s  (%d months)",
             date_min.date(), date_max.date(), T)

    # Build complete (T, N, 3) array
    # Pivot per feature then reindex to full date range
    features = ["S_norm", "I_norm", "R_norm"]
    arrays = []
    for feat in features:
        pivot = df.pivot(index="date", columns="state", values=feat)[states]
        pivot = pivot.reindex(full_dates)
        if feat == "I_norm":
            pivot.fillna(0.0, inplace=True)   # no attack → 0
        else:
            # Forward-fill then back-fill; then zero
            pivot.ffill(inplace=True)
            pivot.bfill(inplace=True)
            pivot.fillna(0.0, inplace=True)
        arrays.append(pivot.values.astype(np.float32))   # (T, N)

    # Also collect harvest_area for auxdata (per state mean)
    ha_pivot = df.pivot(index="date", columns="state",
                        values="harvest_area")[states]
    harvest_area_mean = ha_pivot.mean(
        axis=0).values.astype(np.float32)   # (N,)

    tensor = np.stack(arrays, axis=-1)   # (T, N, 3)
    log.info("  Full data tensor shape: %s  (T, N, F)", tensor.shape)

    return tensor, full_dates, states, harvest_area_mean


# ===========================================================================
# 3.2 – Build sliding windows
# ===========================================================================
def build_windows(tensor: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Parameters
    ----------
    tensor      : (T, N, F)
    window_size : obs_len + pre_len
    stride      : step between consecutive windows

    Returns
    -------
    windows : (n_windows, window_size, N, F)
    """
    T = tensor.shape[0]
    indices = list(range(0, T - window_size + 1, stride))
    windows = np.stack([tensor[i: i + window_size] for i in indices], axis=0)
    log.info("  Sliding windows: %d  (window_size=%d, stride=%d)",
             len(windows), window_size, stride)
    return windows, indices


# ===========================================================================
# 3.3 – Train / Val / Test split (temporal, no shuffle)
# ===========================================================================
def split_windows(
    windows: np.ndarray,
    start_indices: list,
    full_dates: pd.DatetimeIndex,
    obs_len: int,
    pre_len: int,
) -> dict:
    """
    Split windows by the date range of their prediction horizon.

    Train : pred horizon ends before 2018-01
    Val   : pred horizon starts in 2018
    Test  : pred horizon starts in 2020 and does NOT fall in data gap (Apr-May 2020)

    Note: Gap months that appear in the **observation** window are allowed;
    they have been zero-filled earlier and are treated as valid (no-attack) signal.
    Only prediction targets that land on known data-absent months are excluded.
    """
    # Months with zero KCC reports (entirely absent from raw data)
    pred_gaps = set(pd.date_range("2019-01-01", "2019-12-01", freq="MS")) | {
        pd.Timestamp(
            "2017-10-01"), pd.Timestamp("2017-11-01"), pd.Timestamp("2017-12-01"),
        pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01"),
    }

    train_idx, val_idx, test_idx = [], [], []

    for w_pos, start_i in enumerate(start_indices):
        pred_months = set(
            full_dates[start_i + obs_len: start_i + obs_len + pre_len])

        # Skip if any prediction month lands in a known data-absent period
        if pred_months & pred_gaps:
            continue

        pred_start_year = min(m.year for m in pred_months)
        pred_end_year = max(m.year for m in pred_months)
        pred_end_month = max(m.month for m in pred_months)

        if pred_end_year < 2018:
            train_idx.append(w_pos)
        elif pred_start_year == 2018 and pred_end_year == 2018:
            val_idx.append(w_pos)
        elif pred_start_year == 2020:
            test_idx.append(w_pos)

    log.info("  Split (gap-aware): train=%d  val=%d  test=%d",
             len(train_idx), len(val_idx), len(test_idx))

    splits = {}
    for name, idxs in [("training", train_idx), ("validation", val_idx), ("test", test_idx)]:
        splits[name] = windows[idxs] if idxs else np.empty(
            (0,) + windows.shape[1:], dtype=windows.dtype
        )
    return splits


# ===========================================================================
# 3.4 – PyTorch Dataset & DataLoaders
# ===========================================================================
class SIRWindowDataset(Dataset):
    """
    Wraps a (n_windows, window_size, N, F) numpy array into a Dataset.
    Each item returns (x_SIR, yd, yi) with identical tensors (monthly data).
    """

    def __init__(self, windows: np.ndarray, device: str = "cpu"):
        # windows: (n_windows, window_size, N, F=3)
        self.data = torch.as_tensor(
            windows, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]      # (window_size, N, 3)  [S_norm, I_norm, R_norm]
        # The model predicts only I (shape: B, T, N, 1).
        # yd / yi must therefore be I-only to avoid shape-mismatch in loss/metrics.
        yi = x[:, :, 1:2]       # (window_size, N, 1)  – I_norm channel only
        return x, yi, yi        # (x_SIR, yd, yi)


def build_dataloaders(
    splits: dict,
    batch_size: int = 16,
    device: str = "cpu",
) -> dict:
    """Return dict of { 'training': DataLoader, 'validation': ..., 'test': ... }."""
    loaders = {}
    for split_name, arr in splits.items():
        if len(arr) == 0:
            log.warning(
                "  Split '%s' is empty – DataLoader will also be empty.", split_name)
        dataset = SIRWindowDataset(arr, device=device)
        shuffle = split_name == "training"
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
        log.info("  DataLoader '%s': %d windows, batch_size=%d",
                 split_name, len(dataset), batch_size)
    return loaders


# ===========================================================================
# 3.5 – Build and save auxdata
# ===========================================================================
def build_auxdata(
    splits: dict,
    full_dates: pd.DatetimeIndex,
    start_indices: list,
    harvest_area_mean: np.ndarray,
    obs_len: int,
    output_dir: str,
    params: dict,
) -> dict:
    """
    Compute and save auxdata JSON.

    auxdata = {
        'prov_pop': [mean harvest_area per state],  # (N,) – used for denorm
        'imax'    : [1.0] * N,                      # I_norm already in [0,1]
        'imin'    : [0.0] * N,
        'date_list': [str(d) for d in full_dates],
    }
    """
    auxdata = {
        "prov_pop": harvest_area_mean.tolist(),
        "imax": [1.0] * N_STATES,
        "imin": [0.0] * N_STATES,
        "date_list": [str(d.date()) for d in full_dates],
    }

    fname = "_".join(filter(None, [
        params.get("data_type", "kcc"),
        params.get("normalize", "norm"),
        str(params.get("obs_len", obs_len)),
        str(params.get("pre_len", 3)),
        "auxdata.json",
    ]))
    path = os.path.join(output_dir, fname)
    with open(path, "w") as f:
        json.dump(auxdata, f, indent=2)
    log.info("  auxdata saved → %s", path)
    return auxdata, path


# ===========================================================================
# Main – build everything and return DataLoaders + auxdata
# ===========================================================================
def main(
    obs_len: int = 12,
    pre_len: int = 3,
    batch_size: int = 16,
    stride: int = 1,
    device: str = "cpu",
    params: dict = None,
) -> tuple[dict, dict]:
    """
    Build DataLoaders and auxdata for the KCC pipeline.

    Returns
    -------
    data_repo : dict  – { 'training': DataLoader, 'validation': ..., 'test': ... }
    auxdata   : dict  – prov_pop, imax, imin, date_list
    """
    if params is None:
        params = {
            "data_type": "kcc",
            "normalize": "norm",
            "obs_len": obs_len,
            "pre_len": pre_len,
        }

    log.info("=" * 60)
    log.info("PHASE 3 – Dataset Builder")
    log.info("=" * 60)

    window_size = obs_len + pre_len

    # 3.1 Load data
    tensor, full_dates, states, harvest_area_mean = load_complete_panel(
        SIR_PATH)

    # 3.2 Sliding windows
    windows, start_indices = build_windows(
        tensor, window_size=window_size, stride=stride)

    # 3.3 Split
    splits = split_windows(windows, start_indices,
                           full_dates, obs_len, pre_len)

    # 3.4 DataLoaders
    data_repo = build_dataloaders(splits, batch_size=batch_size, device=device)

    # 3.5 auxdata
    auxdata, auxdata_path = build_auxdata(
        splits=splits,
        full_dates=full_dates,
        start_indices=start_indices,
        harvest_area_mean=harvest_area_mean,
        obs_len=obs_len,
        output_dir=OUTPUT_DIR,
        params=params,
    )

    log.info("Phase 3 complete.")
    return data_repo, auxdata


if __name__ == "__main__":
    repo, aux = main()
    print("\nDataLoader sizes:")
    for k, v in repo.items():
        print(f"  {k}: {len(v.dataset)} windows")
    print("\nauxdata keys:", list(aux.keys()))
    print("prov_pop (first 5):", [round(x, 1) for x in aux["prov_pop"][:5]])
