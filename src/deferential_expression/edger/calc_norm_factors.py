"""
Compute normalization factors using edgeR::calcNormFactors.

This module provides a functional interface to calculate normalization factors
and store them in the column_data of a SummarizedExperiment.
"""

from __future__ import annotations
from typing import Any, Optional, TypeVar
import numpy as np

from .utils import _prep_edger
from .checks import check_se, check_assay_exists, check_r_assay

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


def calc_norm_factors(
    se: SE,
    assay: str = "counts",
    method: str = "TMM",
    refColumn: Optional[int] = None,
    logratioTrim: float = 0.3,
    sumTrim: float = 0.05,
    doWeighting: bool = True,
    Acutoff: float = -1e10,
    p: float = 0.75,
    in_place: bool = False,
    **kwargs
) -> SE:
    """
    Compute normalization factors and store in column_data.
    
    Wraps ``edgeR::calcNormFactors``. The normalization factors are
    converted to a Python array and stored in column_data['norm.factors'].
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized count assay.
        assay: Name of the counts assay. Default: "counts".
        method: Normalization method: "TMM", "RLE", "upperquartile", or "none".
        refColumn: Reference column for normalization (0-indexed, or None for auto).
        logratioTrim: Amount of trimming of the log-ratios (TMM only).
        sumTrim: Amount of trimming of intensity values (TMM only).
        doWeighting: Whether to use weighted trimmed mean (TMM only).
        Acutoff: Cutoff on average log-expression (TMM only).
        p: Quantile for upperquartile normalization.
        in_place: If True, modify se in place. Default: False.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        SummarizedExperiment with 'norm.factors' in column_data.
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.edger as edger
        >>> se = initialize_r(se, assay="counts")
        >>> se = edger.calc_norm_factors(se, method="TMM")
        >>> se.column_data["norm.factors"]
    """
    from ..r_init import get_rmat
    
    # Validate inputs
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    
    r, pkg = _prep_edger()
    rmat = get_rmat(se, assay)
    
    # Handle None -> R NULL
    refColumn_r = r.ro.NULL if refColumn is None else refColumn
    
    r_factors = pkg.calcNormFactors(
        rmat,
        method=method,
        refColumn=refColumn_r,
        logratioTrim=logratioTrim,
        sumTrim=sumTrim,
        doWeighting=doWeighting,
        Acutoff=Acutoff,
        p=p,
        **kwargs
    )
    
    # Convert to Python and store in column_data
    norm_factors = np.asarray(r_factors)
    
    # Use SE's set_column_data method for class preservation
    output = se._define_output(in_place=in_place)
    coldata = output.get_column_data()
    if coldata is not None:
        new_coldata = coldata.set_column("norm.factors", norm_factors)
    else:
        from biocframe import BiocFrame
        new_coldata = BiocFrame({"norm.factors": norm_factors})
    
    # Use the proper setter - SummarizedExperiment uses set_column_data
    return output.set_column_data(new_coldata, in_place=True)
