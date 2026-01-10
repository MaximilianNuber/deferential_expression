"""
Input validation utilities for sva functions.

Provides centralized checks for SummarizedExperiment variants.
Supports SE, RSE, SCE, and RESummarizedExperiment.
"""

from __future__ import annotations
from typing import Any, Optional
import pandas as pd


def check_se(se: Any, name: str = "se") -> None:
    """Check that input is a SummarizedExperiment-like object.
    
    Accepts any object with assays, row_names, column_names attributes
    (duck typing for SE, RSE, SCE, RESummarizedExperiment).
    """
    required_attrs = ["assays", "assay_names"]
    for attr in required_attrs:
        if not hasattr(se, attr):
            raise TypeError(
                f"Expected `{name}` to be a SummarizedExperiment-like object "
                f"(SE, RSE, SCE, or RESummarizedExperiment), "
                f"got {type(se).__name__} which lacks '{attr}'"
            )


def check_rse(rse: Any, name: str = "rse") -> None:
    """Check that input is a SummarizedExperiment-like object.
    
    DEPRECATED: Use check_se instead. Kept for backward compatibility.
    """
    check_se(rse, name)


def check_r_assay(se: Any, assay: str) -> None:
    """Check that the specified assay is R-initialized (RMatrixAdapter)."""
    from ..r_init import check_r_initialized
    check_r_initialized(se, assay)


def check_assay_exists(se: Any, assay: str) -> None:
    """Check that the specified assay exists in the SummarizedExperiment."""
    if assay not in se.assay_names:
        available = list(se.assay_names)
        raise KeyError(
            f"Assay '{assay}' not found. Available assays: {available}"
        )


def check_mod(mod: Any, n_samples: Optional[int] = None) -> None:
    """Check that mod is a valid pandas DataFrame."""
    if not isinstance(mod, pd.DataFrame):
        raise TypeError(
            f"Expected `mod` to be a pandas DataFrame, "
            f"got {type(mod).__name__}"
        )
    if n_samples is not None and len(mod) != n_samples:
        raise ValueError(
            f"Model matrix has {len(mod)} rows but expected {n_samples} samples"
        )
