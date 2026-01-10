"""
Apply ComBat batch correction using sva::ComBat.

This module provides a functional interface for batch correction
of continuous/normalized expression data.
"""

from __future__ import annotations
from typing import Optional, Sequence, TypeVar, Union
import numpy as np
import pandas as pd

from .utils import _prep_sva, resolve_batch
from .checks import check_se, check_assay_exists, check_r_assay
from ..edger.utils import pandas_to_r_matrix

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


def combat(
    se: SE,
    batch: Union[str, Sequence, np.ndarray, pd.Series],
    assay: str = "cpm",
    output_assay: Optional[str] = None,
    mod: Optional[pd.DataFrame] = None,
    par_prior: bool = True,
    prior_plots: bool = False,
    mean_only: bool = False,
    ref_batch: Optional[str] = None,
    in_place: bool = False,
    **kwargs
) -> SE:
    """
    Apply ComBat batch correction to continuous/normalized expression data.
    
    Wraps `sva::ComBat`. Takes a matrix of continuous values (e.g., log-CPM)
    and returns a batch-corrected matrix of the same dimensions.
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized expression assay.
        batch: Batch factor. Either column name in column_data (str) or
            array-like of batch labels.
        assay: Input assay name (should be continuous, e.g., "cpm", "log_expr").
            Default: "cpm".
        output_assay: Output assay name. Defaults to "{assay}_combat".
        mod: Optional model matrix for biological covariates to preserve.
        par_prior: Use parametric adjustments. Default: True.
        prior_plots: Show prior distribution plots. Default: False.
        mean_only: Only adjust mean (not variance). Default: False.
        ref_batch: Reference batch level. Default: None.
        in_place: If True, modify se in place. Default: False.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        SummarizedExperiment with batch-corrected assay.
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.sva as sva
        >>> se = initialize_r(se, assay="log_expr")
        >>> se = sva.combat(se, batch="batch_id", assay="log_expr")
    """
    from ..r_init import get_rmat
    from ..rmatrixadapter import RMatrixAdapter
    
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    
    r, sva_pkg = _prep_sva()
    
    # Get input matrix
    dat_r = get_rmat(se, assay)
    
    # Resolve batch to R vector
    batch_r = resolve_batch(se, batch)
    
    # Handle mod (model matrix)
    if mod is not None:
        mod_r = pandas_to_r_matrix(mod)
    else:
        mod_r = r.ro.NULL
    
    # Handle ref.batch
    ref_batch_r = r.ro.NULL if ref_batch is None else ref_batch
    
    # Call ComBat
    result_r = sva_pkg.ComBat(
        dat_r,
        batch=batch_r,
        mod=mod_r,
        par_prior=par_prior,
        prior_plots=prior_plots,
        mean_only=mean_only,
        ref_batch=ref_batch_r,
        **kwargs
    )
    
    # Store result using _define_output for class preservation
    out_name = output_assay if output_assay else f"{assay}_combat"
    output = se._define_output(in_place=in_place)
    new_assays = dict(output.assays)
    new_assays[out_name] = RMatrixAdapter(result_r, r)
    output._assays = new_assays
    
    return output
