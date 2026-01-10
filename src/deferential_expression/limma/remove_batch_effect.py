"""
Remove batch effects using limma::removeBatchEffect.

This module provides a functional interface for batch effect removal.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence, Union
import numpy as np
import pandas as pd

from .utils import _limma
from .checks import check_rse, check_assay_exists
from ..resummarizedexperiment import RMatrixAdapter

if TYPE_CHECKING:
    from ..resummarizedexperiment import RESummarizedExperiment


def remove_batch_effect(
    rse: "RESummarizedExperiment",
    batch: Union[str, Sequence, np.ndarray, pd.Series],
    batch2: Union[str, Sequence, np.ndarray, pd.Series, None] = None,
    exprs_assay: str = "log_expr",
    corrected_assay: str = "log_expr_bc",
    design: Optional[pd.DataFrame] = None,
    covariates: Optional[pd.DataFrame] = None,
    **kwargs,
) -> "RESummarizedExperiment":
    """
    Remove batch effects from expression data.
    
    Wraps ``limma::removeBatchEffect``. Preserves biological variation
    when design matrix is provided.
    
    Args:
        rse: Input RESummarizedExperiment with expression data.
        batch: Primary batch factor (column name in column_data or array).
        batch2: Optional secondary batch factor.
        exprs_assay: Input expression assay name. Default: "log_expr".
        corrected_assay: Output batch-corrected assay name. Default: "log_expr_bc".
        design: Optional design matrix to preserve biological variation.
        covariates: Optional continuous covariates to adjust for.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        New RESummarizedExperiment with batch-corrected assay.
    
    Example:
        >>> import deferential_expression.limma as limma
        >>> rse_bc = limma.remove_batch_effect(rse, batch="batch_column")
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from rpy2.robjects.vectors import StrVector
    from ..edger.utils import pandas_to_r_matrix
    
    check_rse(rse)
    check_assay_exists(rse, exprs_assay)
    
    r = get_r_environment()
    limma_pkg = _limma()
    
    E_r = rse.assay_r(exprs_assay)
    
    # Resolve batch
    def _resolve_batch(b):
        if isinstance(b, str):
            cd = rse.column_data_df
            if cd is None or b not in cd.columns:
                raise KeyError(f"Batch column '{b}' not found in column_data.")
            arr = cd[b].to_numpy()
        else:
            arr = np.asarray(b)
        vals = ["" if v is None else str(v) for v in arr.tolist()]
        return StrVector(vals)
    
    call_kwargs = {"batch": _resolve_batch(batch)}
    
    if batch2 is not None:
        call_kwargs["batch2"] = _resolve_batch(batch2)
    
    if covariates is not None:
        call_kwargs["covariates"] = pandas_to_r_matrix(covariates)
    
    if design is not None:
        call_kwargs["design"] = pandas_to_r_matrix(design)
    
    call_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
    
    out_r = limma_pkg.removeBatchEffect(E_r, **call_kwargs)
    
    assays = dict(rse.assays)
    assays[corrected_assay] = RMatrixAdapter(out_r, r)
    
    from ..resummarizedexperiment import RESummarizedExperiment
    return RESummarizedExperiment(
        assays=assays,
        row_data=rse.row_data_df,
        column_data=rse.column_data_df,
        row_names=rse.row_names,
        column_names=rse.column_names,
        metadata=dict(rse.metadata),
    )
