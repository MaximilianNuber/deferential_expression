"""
Input validation utilities for edgeR functions.

Provides centralized checks for SummarizedExperiment variants and EdgeRModel inputs.
Supports SE, RSE, SCE, and RESummarizedExperiment.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Sequence
import pandas as pd

if TYPE_CHECKING:
    pass


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
    Now accepts any SE variant, not just RESummarizedExperiment.
    """
    check_se(rse, name)


def check_r_assay(se: Any, assay: str) -> None:
    """Check that the specified assay is R-initialized (RMatrixAdapter).
    
    Args:
        se: SummarizedExperiment-like object.
        assay: Assay name to check.
    
    Raises:
        KeyError: If assay doesn't exist.
        TypeError: If assay is not R-initialized.
    """
    from ..r_init import check_r_initialized
    check_r_initialized(se, assay)


def check_assay_exists(se: Any, assay: str) -> None:
    """Check that the specified assay exists in the SummarizedExperiment."""
    if assay not in se.assay_names:
        available = list(se.assay_names)
        raise KeyError(
            f"Assay '{assay}' not found. Available assays: {available}"
        )


def check_design(design: Any, n_samples: Optional[int] = None) -> None:
    """Check that design is a valid pandas DataFrame."""
    if not isinstance(design, pd.DataFrame):
        raise TypeError(
            f"Expected `design` to be a pandas DataFrame, "
            f"got {type(design).__name__}"
        )
    if n_samples is not None and len(design) != n_samples:
        raise ValueError(
            f"Design matrix has {len(design)} rows but expected {n_samples} samples"
        )


def check_edger_model(model: Any) -> None:
    """Check that input is a valid EdgeRModel with a fit object."""
    from .glm_ql_fit import EdgeRModel
    if not isinstance(model, EdgeRModel):
        raise TypeError(
            f"Expected an EdgeRModel, got {type(model).__name__}"
        )
    if model.fit is None:
        raise ValueError("EdgeRModel.fit is None - model has not been fitted")


def check_coef_or_contrast(
    coef: Optional[Any],
    contrast: Optional[Sequence],
    design: Optional[pd.DataFrame] = None
) -> None:
    """Check that at least one of coef or contrast is provided, but not both."""
    if coef is None and contrast is None:
        raise ValueError("Either `coef` or `contrast` must be specified")
    if coef is not None and contrast is not None:
        raise ValueError("Specify either `coef` or `contrast`, not both")
