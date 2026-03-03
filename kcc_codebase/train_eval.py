# -*- coding: utf-8 -*-
"""
train_eval.py
=============
Phase 5 – Training & Evaluation for the KCC pest-attack prediction pipeline.

Wraps the existing `code/Train.Trainer` class, overriding only `get_model`
to add 'kcc' as a recognised `data_type` (31 nodes).

Usage (standalone)
------------------
    python kcc_codebase/train_eval.py

Usage (from run_pipeline.py)
-----------------------------
    from train_eval import main as phase5_main
    result = phase5_main(data_repo, auxdata, params, model_types=['SSIR_STGCN'])
"""

import Train
import EpiODEfit
import EpiGCN
import torch
import json
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Ensure code/ and kcc_codebase/ are importable.
# kcc_codebase must come BEFORE code/ so the EpiGAT stub is found first
# (Train.py imports EpiGAT at the top level; the real EpiGAT.py is absent).
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CODE_DIR = os.path.join(ROOT_DIR, "code")
for _p in [CODE_DIR, BASE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# kcc_codebase must remain ahead of code/ for the EpiGAT stub to win
if sys.path.index(BASE_DIR) > sys.path.index(CODE_DIR):
    sys.path.remove(BASE_DIR)
    sys.path.insert(0, BASE_DIR)

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
# Imports from code/
# ---------------------------------------------------------------------------

# EpiGAT is optional (file may not exist in all repo versions)
try:
    import EpiGAT
    _HAS_EPIGAT = True
except ModuleNotFoundError:
    _HAS_EPIGAT = False


# ===========================================================================
# KCCTrainer – subclass Trainer with 'kcc' data_type support
# ===========================================================================
class KCCTrainer(Train.Trainer):
    """
    Thin subclass of Trainer that adds 'kcc' as a recognised data_type.
    KCC has 31 states → nsize=31 (same as 'china').
    All other behaviour is inherited unchanged.
    """

    NSIZE_MAP = {
        "china": 31,
        "kcc": 31,
        "germany": 16,
    }

    def get_model(self):
        nsize = self.NSIZE_MAP.get(self.params["data_type"], 31)
        model_type = self.model_type

        if model_type == "SSIR_STGCN":
            return EpiGCN.SSIR_STGCN(
                obs_len=self.params["obs_len"],
                pre_len=self.params["pre_len"],
                kernel_size=self.params["kernel_size"],
                num_layers=self.params["num_layers"],
                adj_type=self.adj_type,
                neighbor_matrix=self.neighbor_matrix,
                in_dim=3,
                t_out_dim=self.params.get("t_out_dim", 16),
                s_out_dim=self.params.get("s_out_dim", 16),
                dropout=self.params.get("dropout", 0.1),
                nsize=nsize,
                beta_incorporated=self.params["beta_incorporated"],
            )

        elif model_type == "SSIR_STGAT":
            if not _HAS_EPIGAT:
                raise ImportError(
                    "EpiGAT.py is not present in code/. "
                    "SSIR_STGAT model is unavailable."
                )
            return EpiGAT.SSIR_STGAT(
                obs_len=self.params["obs_len"],
                pre_len=self.params["pre_len"],
                kernel_size=self.params["kernel_size"],
                num_layers=self.params["num_layers"],
                in_dim=3,
                nsize=nsize,
                t_out_dim=self.params.get("t_out_dim", 16),
                s_out_dim=self.params.get("s_out_dim", 16),
                dropout=self.params.get("dropout", 0.1),
                beta_incorporated=self.params["beta_incorporated"],
            )

        elif model_type == "SSIR_ODEFIT":
            return EpiODEfit.SSIR_ODEFIT(
                mtype=self.params["ssir"],
                obs_len=self.params["obs_len"],
                pre_len=self.params["pre_len"],
            )

        else:
            raise NotImplementedError(f"Unsupported model: {model_type}")


# ===========================================================================
# Main entry point
# ===========================================================================
def main(
    data_repo: dict,
    auxdata: dict,
    params: dict,
    model_types: list = None,
) -> dict:
    """
    Run training and evaluation for one or more model types.

    Parameters
    ----------
    data_repo   : dict  { 'training': DataLoader, 'validation': ..., 'test': ... }
    auxdata     : dict  { 'prov_pop', 'imax', 'imin', 'date_list' }
    params      : dict  – from model_config.get_params()
    model_types : list  – e.g. ['SSIR_STGCN']. Defaults to ['SSIR_STGCN'].

    Returns
    -------
    results : dict  { model_type: { 'trainer': KCCTrainer } }
    """
    import torch
    torch.manual_seed(params.get("random_seed", 42))

    if model_types is None:
        model_types = ["SSIR_STGCN"]

    log.info("=" * 60)
    log.info("PHASE 5 – Train & Eval")
    log.info("=" * 60)

    # Validate data loaders
    for split in ("training", "validation", "test"):
        n = len(data_repo.get(split, {}).dataset) if split in data_repo else 0
        log.info("  DataLoader '%s': %d samples", split, n)

    if len(data_repo.get("training", {}).dataset) == 0:
        raise ValueError("Training DataLoader is empty. Check Phase 3 output.")

    results = {}
    for model_type in model_types:
        log.info("-" * 40)
        log.info("Model: %s", model_type)

        trainer = KCCTrainer(params=params, model_type=model_type)
        log.info("  Model params: %d",
                 sum(p.numel() for p in trainer.model.parameters()))

        # Training
        if model_type != "SSIR_ODEFIT":
            trainer.train(
                data_loader=data_repo,
                modes=["training", "validation"],
            )
            # Evaluation on all splits
            if params.get("test", False) and len(data_repo.get("test", {}).dataset) > 0:
                trainer.test(
                    data_loader=data_repo,
                    modes=["training", "validation", "test"],
                    auxdata=auxdata,
                )
            else:
                log.warning(
                    "Skipping test phase: test DataLoader empty or test=False."
                )
        else:
            if len(data_repo.get("test", {}).dataset) == 0:
                log.warning("ODE estimator skipped: test DataLoader empty.")
            else:
                trainer.ode_estimator(
                    data_loader=data_repo["test"],
                    auxdata=auxdata,
                )

        results[model_type] = {"trainer": trainer}
        log.info("Model %s complete.", model_type)

    log.info("Phase 5 complete.")
    return results


# ===========================================================================
# Standalone execution
# ===========================================================================
if __name__ == "__main__":
    from model_config import get_params
    from dataset_builder import main as build_data

    params = get_params()

    # Build data (Phase 3)
    data_repo, auxdata = build_data(
        obs_len=params["obs_len"],
        pre_len=params["pre_len"],
        batch_size=params["batch_size"],
        device=params["dev"],
        params=params,
    )

    # Train (Phase 5)
    results = main(
        data_repo=data_repo,
        auxdata=auxdata,
        params=params,
        model_types=["SSIR_STGCN"],
    )
    print("\nTraining complete.")
