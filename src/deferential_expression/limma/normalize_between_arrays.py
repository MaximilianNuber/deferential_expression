"""
Normalize between arrays using limma::normalizeBetweenArrays.

This module provides a functional interface for array normalization.
"""

from __future__ import annotations
from typing import TypeVar

from .utils import _limma
from .checks import check_se, check_assay_exists, check_r_assay

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


def normalize_between_arrays(
    se: SE,
    assay: str = "log_expr",
    normalized_assay: str = "log_expr_norm",
    method: str = "quantile",
    in_place: bool = False,
    **kwargs
) -> SE:
    """
    Normalize expression values between arrays/samples.
    
    Wraps ``limma::normalizeBetweenArrays``.
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized expression assay.
        assay: Input expression assay name. Default: "log_expr".
        normalized_assay: Output normalized assay name. Default: "log_expr_norm".
        method: Normalization method ("quantile", "scale", "cyclicloess", etc.).
        in_place: If True, modify se in place. Default: False.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        SummarizedExperiment with normalized assay.
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.limma as limma
        >>> se = initialize_r(se, assay="log_expr")
        >>> se = limma.normalize_between_arrays(se, method="quantile")
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from ..r_init import get_rmat
    from ..rmatrixadapter import RMatrixAdapter
    
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    
    r = get_r_environment()
    limma_pkg = _limma()
    
    exprs_r = get_rmat(se, assay)
    out_r = limma_pkg.normalizeBetweenArrays(exprs_r, method=method, **kwargs)
    
    # Use SE's _define_output pattern for class preservation
    output = se._define_output(in_place=in_place)
    new_assays = dict(output.assays)
    new_assays[normalized_assay] = RMatrixAdapter(out_r, r)
    output._assays = new_assays
    
    return output
