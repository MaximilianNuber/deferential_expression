"""
Run voom transformation using limma::voom.

This module provides a functional interface to perform voom transformation
and store results as assays in a SummarizedExperiment.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, TypeVar, Union
import numpy as np
import pandas as pd

from .utils import _limma
from .checks import check_se, check_assay_exists, check_r_assay, check_design

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


def voom(
    se: SE,
    design: pd.DataFrame,
    assay: str = "counts",
    lib_size: Optional[Union[pd.Series, Sequence, np.ndarray]] = None,
    block: Optional[Union[pd.Series, Sequence, np.ndarray, pd.Categorical]] = None,
    log_expr_assay: str = "log_expr",
    weights_assay: str = "weights",
    plot: bool = False,
    in_place: bool = False,
    **kwargs
) -> SE:
    """
    Run voom transformation on counts to compute log-CPM and weights.
    
    Wraps ``limma::voom``. Results stored as assays: 'log_expr' and 'weights'.
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized count assay.
        design: Design matrix (samples × covariates) as pandas DataFrame.
        assay: Input counts assay name. Default: "counts".
        lib_size: Optional library sizes per sample.
        block: Optional blocking factor (e.g., batch).
        log_expr_assay: Name for log-expression assay. Default: "log_expr".
        weights_assay: Name for weights assay. Default: "weights".
        plot: Whether to show voom diagnostic plot. Default: False.
        in_place: If True, modify se in place. Default: False.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        SummarizedExperiment with log_expr and weights assays.
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.limma as limma
        >>> se = initialize_r(se, assay="counts")
        >>> se = limma.voom(se, design)
        >>> log_expr = np.asarray(se.assays["log_expr"])
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from ..edger.utils import pandas_to_r_matrix
    from ..r_init import get_rmat
    from ..rmatrixadapter import RMatrixAdapter
    
    # Validate inputs
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    n_samples = len(se.column_names) if se.column_names else se.shape[1]
    check_design(design, n_samples)
    
    r = get_r_environment()
    limma_pkg = _limma()
    
    counts_r = get_rmat(se, assay)
    design_r = pandas_to_r_matrix(design)
    
    # Handle lib_size
    coldata = se.get_column_data()
    if lib_size is not None:
        lib_size = np.asarray(lib_size, dtype=float)
        lib_size_r = r.FloatVector(lib_size)
    elif coldata is not None and "norm.factors" in coldata.column_names:
        nf = np.asarray(coldata["norm.factors"], dtype=float)
        lib_size_r = r.FloatVector(nf)
    else:
        lib_size_r = r.ro.NULL
    
    # Handle block
    if block is not None:
        if isinstance(block, pd.Categorical):
            with r.localconverter(r.default_converter + r.pandas2ri.converter):
                block_r = r.get_conversion().py2rpy(block)
        else:
            block_r = r.ro.StrVector(np.asarray(block, dtype=str))
    else:
        block_r = r.ro.NULL
    
    # Call voom
    voom_out = limma_pkg.voom(
        counts_r, design_r,
        plot=plot,
        lib_size=lib_size_r,
        block=block_r,
        **kwargs
    )
    
    # Extract E and weights
    E_r = r.ro.baseenv["[["](voom_out, "E")
    weights_r = r.ro.baseenv["[["](voom_out, "weights")
    
    # Use SE's _define_output pattern for class preservation
    output = se._define_output(in_place=in_place)
    new_assays = dict(output.assays)
    new_assays[log_expr_assay] = RMatrixAdapter(E_r, r)
    new_assays[weights_assay] = RMatrixAdapter(weights_r, r)
    output._assays = new_assays
    
    return output
