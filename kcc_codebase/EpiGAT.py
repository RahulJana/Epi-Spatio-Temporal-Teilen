# -*- coding: utf-8 -*-
"""
EpiGAT.py  (kcc_codebase stub)
================================
EpiGAT.py is not present in this repository version.
This stub satisfies the `import EpiGAT` in code/Train.py so the pipeline can
import and run SSIR_STGCN without EpiGAT.

SSIR_STGAT will raise NotImplementedError if instantiated.
"""
from torch import nn


class SSIR_STGAT(nn.Module):
    """Stub – raises NotImplementedError when instantiated."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SSIR_STGAT is not available: EpiGAT.py is missing from code/. "
            "Use SSIR_STGCN instead."
        )

    def forward(self, x):
        raise NotImplementedError
