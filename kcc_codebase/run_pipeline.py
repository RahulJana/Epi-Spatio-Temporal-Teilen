# -*- coding: utf-8 -*-
"""
run_pipeline.py
===============
Main runner for the KCC Pest Attack Prediction pipeline.

Phases
------
1. data_preprocessing   – Clean, impute, aggregate, build SIR variables
2. graph_construction   – Build static / soft / correlation adjacency matrices
3. dataset_builder      – Sliding-window DataLoaders, train/val/test split
4. model_config         – Hyperparameter config and Constant.py patches
5. train_eval           – Training loop + evaluation metrics
6. visualization        – Forecast curves, spatial heatmaps, epi-param plots

Usage
-----
    python kcc_codebase/run_pipeline.py [--phases 1 2 3]

    Omitting --phases runs all phases in sequence.
"""

import argparse
import logging
import sys
import time

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
# Phase runners
# ---------------------------------------------------------------------------

def run_phase_1():
    """Data preprocessing – clean, impute, aggregate, SIR construction."""
    from data_preprocessing import main as phase1_main
    log.info("=" * 60)
    log.info("PHASE 1 – Data Preprocessing")
    log.info("=" * 60)
    result = phase1_main()
    log.info("Phase 1 complete. Output shape: %s", result.shape)
    return result


def run_phase_2():
    """Spatial graph construction – adjacency matrices."""
    from graph_construction import main as phase2_main
    log.info("=" * 60)
    log.info("PHASE 2 – Graph Construction")
    log.info("=" * 60)
    geo, soft, corr = phase2_main()
    log.info("Phase 2 complete. Adjacency matrices: geo=%s, soft=%s, corr=%s",
             geo.shape, soft.shape, corr.shape)
    return geo, soft, corr


def run_phase_3(context: dict):
    """Dataset builder – sliding window DataLoaders."""
    from dataset_builder import main as phase3_main
    log.info("=" * 60)
    log.info("PHASE 3 – Dataset Builder")
    log.info("=" * 60)
    params = context.get("params", {})
    data_repo, auxdata = phase3_main(
        obs_len=params.get("obs_len", 12),
        pre_len=params.get("pre_len", 3),
        batch_size=params.get("batch_size", 8),
        device=params.get("dev", "cpu"),
        params=params,
    )
    context["data_repo"] = data_repo
    context["auxdata"] = auxdata
    sizes = {k: len(v.dataset) for k, v in data_repo.items()}
    log.info("Phase 3 complete. DataLoader sizes: %s", sizes)
    return data_repo, auxdata


def run_phase_4(context: dict):
    """Model configuration – hyperparameters."""
    from model_config import get_params
    log.info("=" * 60)
    log.info("PHASE 4 – Model Config")
    log.info("=" * 60)
    params = get_params()
    context["params"] = params
    log.info("Phase 4 complete. graph_type=%s  obs_len=%d  pre_len=%d",
             params["graph_type"], params["obs_len"], params["pre_len"])
    return params


def run_phase_5(context: dict):
    """Training and evaluation."""
    from train_eval import main as phase5_main
    log.info("=" * 60)
    log.info("PHASE 5 – Train & Eval")
    log.info("=" * 60)
    data_repo = context.get("data_repo")
    auxdata = context.get("auxdata")
    params = context.get("params")
    if data_repo is None or params is None:
        raise RuntimeError(
            "Phase 5 requires data_repo (Phase 3) and params (Phase 4) "
            "to have been run first."
        )
    results = phase5_main(
        data_repo=data_repo,
        auxdata=auxdata,
        params=params,
        model_types=["SSIR_STGCN"],
    )
    context["results"] = results
    log.info("Phase 5 complete. Models trained: %s", list(results.keys()))
    return results


def run_phase_6():
    """Visualization – forecast curves, heatmaps, epi-param analysis."""
    log.info("=" * 60)
    log.info("PHASE 6 – Visualization  [NOT YET IMPLEMENTED]")
    log.info("=" * 60)
    raise NotImplementedError(
        "visualization.py has not been created yet."
    )


# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------
# Note: phases 3–5 accept a shared context dict for inter-phase data passing.
PHASE_REGISTRY = {
    1: run_phase_1,
    2: run_phase_2,
    3: run_phase_3,
    4: run_phase_4,
    5: run_phase_5,
    6: run_phase_6,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="KCC Pest Attack Prediction Pipeline"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        choices=list(PHASE_REGISTRY.keys()),
        default=list(PHASE_REGISTRY.keys()),
        help="Phases to run (default: all). Example: --phases 1 2",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import os
    # Ensure imports resolve relative to kcc_codebase/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    args = parse_args()
    phases_to_run = sorted(args.phases)

    log.info("Pipeline starting – phases to run: %s", phases_to_run)
    overall_start = time.time()

    # Shared context carries state between phases
    context: dict = {}

    # Pre-load model config if any of phases 3,4,5 are in scope, so phase 3
    # can use the correct obs_len / pre_len / batch_size.
    if any(p in phases_to_run for p in [3, 4, 5]):
        log.info("Pre-loading model config for inter-phase context ...")
        try:
            from model_config import get_params
            context["params"] = get_params()
        except Exception as exc:
            log.warning("Could not pre-load model config: %s", exc)

    for phase_num in phases_to_run:
        t0 = time.time()
        try:
            fn = PHASE_REGISTRY[phase_num]
            # Phases 3, 4, 5 accept a context dict
            import inspect
            sig = inspect.signature(fn)
            if "context" in sig.parameters:
                fn(context)
            else:
                fn()
        except NotImplementedError as exc:
            log.warning("Skipping Phase %d: %s", phase_num, exc)
            log.info("Stopping pipeline – implement remaining phases first.")
            break
        except Exception:
            log.exception(
                "Phase %d failed with an unexpected error.", phase_num)
            sys.exit(1)
        elapsed = time.time() - t0
        log.info("Phase %d finished in %.1f s\n", phase_num, elapsed)

    total = time.time() - overall_start
    log.info("Pipeline done in %.1f s", total)


if __name__ == "__main__":
    main()
