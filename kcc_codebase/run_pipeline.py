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
    log.info("=" * 60)
    log.info("PHASE 2 – Graph Construction  [NOT YET IMPLEMENTED]")
    log.info("=" * 60)
    raise NotImplementedError(
        "graph_construction.py has not been created yet."
    )


def run_phase_3():
    """Dataset builder – sliding window DataLoaders."""
    log.info("=" * 60)
    log.info("PHASE 3 – Dataset Builder  [NOT YET IMPLEMENTED]")
    log.info("=" * 60)
    raise NotImplementedError(
        "dataset_builder.py has not been created yet."
    )


def run_phase_4():
    """Model configuration – hyperparameters and Constant.py patches."""
    log.info("=" * 60)
    log.info("PHASE 4 – Model Config  [NOT YET IMPLEMENTED]")
    log.info("=" * 60)
    raise NotImplementedError(
        "model_config.py has not been created yet."
    )


def run_phase_5():
    """Training and evaluation."""
    log.info("=" * 60)
    log.info("PHASE 5 – Train & Eval  [NOT YET IMPLEMENTED]")
    log.info("=" * 60)
    raise NotImplementedError(
        "train_eval.py has not been created yet."
    )


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

    for phase_num in phases_to_run:
        t0 = time.time()
        try:
            PHASE_REGISTRY[phase_num]()
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
