"""
Filter genes by expression using edgeR::filterByExpr.

This module provides a functional interface to compute an expression filter
mask for genes in a SummarizedExperiment.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, TypeVar
import numpy as np
import pandas as pd

from .utils import _prep_edger, pandas_to_r_matrix
from .checks import check_se, check_assay_exists, check_r_assay

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


def filter_by_expr(
    se: SE,
    assay: str = "counts",
    group: Optional[Sequence[str]] = None,
    design: Optional[pd.DataFrame] = None,
    lib_size: Optional[Sequence] = None,
    min_count: float = 10,
    min_total_count: float = 15,
    large_n: int = 10,
    min_prop: float = 0.7,
    **kwargs
) -> np.ndarray:
    """
    Compute expression filter mask using edgeR's filterByExpr.
    
    Returns a boolean mask indicating which genes pass the expression
    filter. Does NOT modify the SE - use the mask to subset manually.
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized count assay.
        assay: Name of the counts assay. Default: "counts".
        group: Optional group factor for filtering.
        design: Optional design matrix as pandas DataFrame.
        lib_size: Optional library sizes.
        min_count: Minimum count required in at least some samples. Default: 10.
        min_total_count: Minimum total count across all samples. Default: 15.
        large_n: Number of samples where min_count must be exceeded. Default: 10.
        min_prop: Minimum proportion of samples with counts above min_count. Default: 0.7.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        Boolean numpy array mask (True = keep gene).
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.edger as edger
        >>> se = initialize_r(se, assay="counts")
        >>> mask = edger.filter_by_expr(se, min_count=10)
        >>> se_filtered = se[mask, :]
    """
    from ..r_init import get_rmat
    
    # Validate inputs
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    
    r, pkg = _prep_edger()
    rmat = get_rmat(se, assay)
    
    # Convert Python args to R
    group_r = r.StrVector(group) if group is not None else r.ro.NULL
    design_r = pandas_to_r_matrix(design) if design is not None else r.ro.NULL
    lib_size_r = r.ro.NULL if lib_size is None else r.FloatVector(np.asarray(lib_size, dtype=float))
    
    mask_r = pkg.filterByExpr(
        rmat,
        group=group_r,
        design=design_r,
        lib_size=lib_size_r,
        min_count=min_count,
        min_total_count=min_total_count,
        large_n=large_n,
        min_prop=min_prop,
        **kwargs
    )
    
    return np.asarray(mask_r, dtype=bool)
