"""
R Initialization Utilities for SummarizedExperiment objects.

This module provides utilities for initializing R-backed assays in any
BiocPy SummarizedExperiment variant (SE, RSE, SCE).

Usage:
    >>> from deferential_expression import initialize_r, check_r_initialized
    >>> se = initialize_r(se, assay="counts")  # Convert counts to RMatrixAdapter
    >>> # Now edgeR/limma/sva functions can use the R-backed assay
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, TypeVar, Union
import numpy as np

# Re-export RMatrixAdapter
from .rmatrixadapter import RMatrixAdapter

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")  # SummarizedExperiment, RangedSE, SingleCellExperiment


def initialize_r(
    se: SE,
    assay: str = "counts",
    in_place: bool = False,
) -> SE:
    """
    Initialize R backing for a SummarizedExperiment assay.
    
    Converts the specified assay to an RMatrixAdapter, enabling R-based
    analysis functions. This must be called before using edgeR/limma/sva functions
    on the SummarizedExperiment.
    
    Works with any BiocPy SummarizedExperiment variant:
    - SummarizedExperiment
    - RangedSummarizedExperiment
    - SingleCellExperiment
    
    Args:
        se: Input SummarizedExperiment (any variant).
        assay: Name of the assay to convert. Default: "counts".
        in_place: If True, modify se in place. If False, return a new SE.
    
    Returns:
        The same SE type with the assay converted to RMatrixAdapter.
    
    Example:
        >>> from summarizedexperiment import SummarizedExperiment
        >>> se = SummarizedExperiment(assays={'counts': counts_array})
        >>> se = initialize_r(se, assay="counts")
        >>> # Now se.assays["counts"] is an RMatrixAdapter
        >>> # Proceed with edger.calc_norm_factors(se, ...)
    
    Raises:
        KeyError: If the specified assay does not exist.
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from bioc2ri import numpy_plugin
    from bioc2ri.rnames import set_rownames, set_colnames
    
    if assay not in se.assay_names:
        raise KeyError(f"Assay '{assay}' not found. Available: {list(se.assay_names)}")
    
    # Get the assay data
    arr = se.assays[assay]
    
    # If already an RMatrixAdapter, return as-is
    if isinstance(arr, RMatrixAdapter):
        return se
    
    # Convert to R matrix
    r = get_r_environment()
    np_eng = numpy_plugin()
    
    arr_np = np.asarray(arr)
    rmat = np_eng.py2r(arr_np)
    
    # Set dimnames if available
    if se.row_names is not None:
        rn = np.asarray(se.row_names, dtype=str)
        rmat = set_rownames(rmat, np_eng.py2r(rn))
    if se.column_names is not None:
        cn = np.asarray(se.column_names, dtype=str)
        rmat = set_colnames(rmat, np_eng.py2r(cn))
    
    # Create adapter
    adapter = RMatrixAdapter(rmat, r)
    
    # Update the assay using BiocPy's pattern
    new_assays = dict(se.assays)
    new_assays[assay] = adapter
    
    # Use _define_output pattern for in_place support
    output = se._define_output(in_place=in_place)
    output._assays = new_assays
    
    return output


def check_r_initialized(se: Any, assay: str) -> None:
    """
    Check that an assay is R-initialized (is an RMatrixAdapter).
    
    Args:
        se: SummarizedExperiment (any variant).
        assay: Assay name to check.
    
    Raises:
        KeyError: If the assay does not exist.
        TypeError: If the assay is not an RMatrixAdapter.
    
    Example:
        >>> check_r_initialized(se, "counts")  # Raises TypeError if not initialized
    """
    if assay not in se.assay_names:
        raise KeyError(f"Assay '{assay}' not found. Available: {list(se.assay_names)}")
    
    arr = se.assays[assay]
    if not isinstance(arr, RMatrixAdapter):
        raise TypeError(
            f"Assay '{assay}' is not R-initialized. "
            f"Call initialize_r(se, assay='{assay}') first."
        )


def is_r_initialized(se: Any, assay: str) -> bool:
    """
    Check if an assay is R-initialized (is an RMatrixAdapter).
    
    Args:
        se: SummarizedExperiment (any variant).
        assay: Assay name to check.
    
    Returns:
        True if the assay is an RMatrixAdapter, False otherwise.
    """
    if assay not in se.assay_names:
        return False
    return isinstance(se.assays[assay], RMatrixAdapter)


def get_rmat(se: Any, assay: str) -> Any:
    """
    Get the underlying R matrix from an R-initialized assay.
    
    Args:
        se: SummarizedExperiment (any variant).
        assay: Assay name.
    
    Returns:
        The underlying R matrix (rpy2 SEXP object).
    
    Raises:
        TypeError: If the assay is not R-initialized.
    """
    check_r_initialized(se, assay)
    adapter: RMatrixAdapter = se.assays[assay]
    return adapter.rmat
