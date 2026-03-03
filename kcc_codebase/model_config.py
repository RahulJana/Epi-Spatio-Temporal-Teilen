# -*- coding: utf-8 -*-
"""
model_config.py
===============
KCC-specific hyperparameter configuration for the SSIR-STGCN model.

Returns a `params` dict ready for use with `train_eval.KCCTrainer`.

Phase 4 responsibilities
------------------------
4.1  Define KCC-specific hyperparameters.
4.2  Set up output directories.
4.3  Prepare the correct static adjacency-matrix path so `Static` graph
     mode works without modifying code/Constant.py.

Usage (standalone)
------------------
    from model_config import get_params
    params = get_params()
"""

import logging
import os
import shutil

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CODE_DIR = os.path.join(ROOT_DIR, "code")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
# 4.1  Hyperparameter defaults
# ===========================================================================
DEFAULT_PARAMS = {
    # ---- Identity --------------------------------------------------------
    "data_type": "kcc",
    "graph_type": "Dynamic",          # Static | Dynamic | Adaptive

    # ---- Architecture ----------------------------------------------------
    "obs_len": 12,                    # look-back window (months)
    "pre_len": 3,                     # forecast horizon (months)
    "kernel_size": 3,
    "num_layers": 3,
    "t_out_dim": 16,
    "s_out_dim": 16,
    "dropout": 0.1,

    # ---- Training --------------------------------------------------------
    "batch_size": 8,
    "max_epoch": 200,
    "early_stop": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "momentum": 0.95,
    "clip": 5.0,
    "random_seed": 42,

    # ---- Loss & physics --------------------------------------------------
    "loss_type": "cMAE",              # MSE | MAE | cMSE | cMAE
    "optimizer": "Adam",              # Adam | SGD | RMSprop
    "scheduler": "ReduceLROnPlateau",  # StepLR | ExponentialLR | ReduceLROnPlateau
    "mode": "min",
    "factor": 0.5,
    "patience": 10,
    "step_size": 50,
    "gamma": 0.9999,
    "milestones": [50, 100, 150],
    "t_max": 50,
    "eta_min": 1e-5,
    "w4pre": 1,
    "w4phy": 1,
    "phyloss4all": False,

    # ---- SIR physics -----------------------------------------------------
    "ssir": "ssir",                   # 'ssir' uses beta, gamma + state param
    "beta_incorporated": True,

    # ---- Data options ----------------------------------------------------
    "normalize": "norm",
    "daily": False,                   # monthly aggregation

    # ---- Misc ------------------------------------------------------------
    "dev": "cpu",
    "grad_print": False,
    "test": True,

    # ---- Derived (set by get_params) ------------------------------------
    "window_size": None,              # obs_len + pre_len
    "max_horizon": None,              # = pre_len
    "output_dir": None,
    "input_dir": None,
}


# ===========================================================================
# 4.2  Set up derived fields and directories
# ===========================================================================
def get_params(overrides: dict = None) -> dict:
    """
    Return a fully-populated params dict, merging DEFAULT_PARAMS with any
    overrides provided.

    Parameters
    ----------
    overrides : dict or None
        Key-value pairs that override DEFAULT_PARAMS entries.

    Returns
    -------
    params : dict
    """
    params = DEFAULT_PARAMS.copy()
    if overrides:
        params.update(overrides)

    # Derived fields
    params["window_size"] = params["obs_len"] + params["pre_len"]
    params["max_horizon"] = params["pre_len"]

    # Directories
    params["output_dir"] = RESULTS_DIR
    params["input_dir"] = PROCESSED_DIR

    # For Static graph: ensure adjacency matrix is accessible at the path
    # the Trainer would look for via Constant.Paths.NEIGHBOR_ADJACENCY_MATRIX
    if params["graph_type"] == "Static":
        _ensure_static_adjacency(params)

    log.info("Model config loaded. data_type=%s  graph_type=%s  obs_len=%d  pre_len=%d",
             params["data_type"], params["graph_type"],
             params["obs_len"], params["pre_len"])
    return params


# ===========================================================================
# 4.3  Static adjacency – ensure file is in the path Trainer expects
# ===========================================================================
def _ensure_static_adjacency(params: dict) -> None:
    """
    Constant.Paths.NEIGHBOR_ADJACENCY_MATRIX resolves to:
       Spatial-SIR-C_GNNs/raw_data/report_data/{data_type}_epidata/mobility/
       neighbor_adjacency_matrix.csv

    Copy our kcc adjacency matrix there so the existing Trainer can load it
    without modification.
    """
    from Constant import Paths   # noqa: available when code/ is on sys.path

    target_path = Paths.NEIGHBOR_ADJACENCY_MATRIX.format(
        data_type=params["data_type"]
    )
    src_path = os.path.join(PROCESSED_DIR, "neighbor_adjacency_matrix.csv")

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Static adjacency matrix not found at {src_path}. "
            "Run Phase 2 first."
        )

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(src_path, target_path)
    log.info("Copied static adjacency matrix → %s", target_path)


# ===========================================================================
# Standalone check
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, CODE_DIR)
    p = get_params()
    print("\n=== KCC Model Config ===")
    for k, v in sorted(p.items()):
        print(f"  {k:25s} = {v}")
