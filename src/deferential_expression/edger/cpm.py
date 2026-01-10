"""
Compute counts per million (CPM) using edgeR::cpm.

This module provides a functional interface to compute CPM and store
the result as a new assay in a SummarizedExperiment.
"""

from __future__ import annotations
from typing import Any, TypeVar
import numpy as np

from .utils import _prep_edger
from .checks import check_se, check_assay_exists, check_r_assay

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


def cpm(
    se: SE,
    assay: str = "counts",
    log: bool = False,
    prior_count: float = 2.0,
    normalized_lib_sizes: bool = True,
    in_place: bool = False,
    **kwargs
) -> SE:
    """
    Compute counts per million and add as new assay.
    
    Wraps ``edgeR::cpm``. The result is stored as an R-backed assay
    via RMatrixAdapter.
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized count assay.
        assay: Name of the input counts assay. Default: "counts".
        log: Whether to compute log2-CPM. Default: False.
        prior_count: Prior count to add before log transformation. Default: 2.0.
        normalized_lib_sizes: Whether to use normalized library sizes. Default: True.
        in_place: If True, modify se in place. Default: False.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        SummarizedExperiment with 'cpm' (or 'logcpm' if log=True) assay.
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.edger as edger
        >>> se = initialize_r(se, assay="counts")
        >>> se = edger.cpm(se, log=True)
        >>> cpm_values = np.asarray(se.assays["logcpm"])
    """
    from ..r_init import get_rmat
    from ..rmatrixadapter import RMatrixAdapter
    
    # Validate inputs
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    
    r, pkg = _prep_edger()
    rmat = get_rmat(se, assay)
    
    cpm_r = pkg.cpm(
        rmat,
        log=log,
        prior_count=prior_count,
        normalized_lib_sizes=normalized_lib_sizes,
        **kwargs
    )
    
    # Store as R-backed assay
    assay_name = "logcpm" if log else "cpm"
    
    # Use SE's _define_output pattern for class preservation
    output = se._define_output(in_place=in_place)
    new_assays = dict(output.assays)
    new_assays[assay_name] = RMatrixAdapter(cpm_r, r)
    output._assays = new_assays
    
    return output
