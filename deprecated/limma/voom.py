import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from .utils import _limma
from bioc2ri.lazy_r_env import get_r_environment
from ..resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter
from ..edger.utils import numpy_to_r_matrix, pandas_to_r_matrix

# ——— 1) voom.default → log-expression + weights ———
def voom(
    se: "RESummarizedExperiment",
    design: pd.DataFrame,
    lib_size: Optional[Union[pd.Series, Sequence, np.ndarray]] = None,
    block: Optional[Union[pd.Series, Sequence, np.ndarray, pd.Categorical]] = None,
    log_expr_assay: str = "log_expr",
    weights_assay: str = "weights",
    plot: bool = False,
    **kwargs
) -> "RESummarizedExperiment":
    """Run `limma::voom` (default) on counts to compute log-CPM and weights.

    This wraps the R `voom` default method using an R-backed counts assay and a
    pandas design matrix. It writes two new assays into the returned object:
    `log_expr_assay` (log-CPM matrix) and `weights_assay` (precision weights).

    Args:
        se: Input `RESummarizedExperiment` containing a `"counts"` R-backed assay.
        design: Design matrix (samples × covariates) as a pandas DataFrame.
        lib_size: Optional library sizes per sample (length = n_samples). If not
            provided, attempts to fall back to `column_data['norm.factors']` if present.
        block: Optional blocking factor (e.g., batch) as array-like or pandas
            Categorical for voom’s correlation/duplicateCorrelation workflows.
        log_expr_assay: Name for the output log-expression assay.
        weights_assay: Name for the output weights assay.
        plot: Whether to enable voom’s diagnostic plot in R.
        **kwargs: Additional keyword arguments forwarded to `limma::voom`.

    Returns:
        RESummarizedExperiment: A new object with added assays for log-CPM and weights.

    Notes:
        This function assumes the `"counts"` assay is an R matrix accessible
        via `se.assay_r("counts")`. New assays are stored as `RMatrixAdapter`s.
    """
    limma_pkg = _limma()
    r = get_r_environment()
    # extract raw counts R matrix
    counts_r = se.assay_r("counts")
    # R design
    design_r = pandas_to_r_matrix(design)

    # convert lib_size to R if provided
    if lib_size is not None:
        lib_size = np.asarray(lib_size, dtype=float)
        assert lib_size.ndim == 1, "lib_size must be a 1D array-like"
        lib_size = r.FloatVector(lib_size)
    else:
        if "norm.factors" in se.column_data.column_names:
            # use norm factors as lib_size if available
            lib_size = se.column_data["norm.factors"]
            lib_size = np.asarray(lib_size, dtype=float)
            
            lib_size = r.FloatVector(lib_size)
        else:
            lib_size = r.ro.NULL

    

    if block is not None:
        if isinstance(block, pd.Categorical):
            with r.localconverter(r.default_converter + r.pandas2ri.converter):
                block = r.get_conversion().py2rpy(block)
        else:
            block = np.asarray(block, dtype=str)
            block = r.ro.StrVector(block)
    else:
        block = r.ro.NULL

    

    # call voom.default
    voom_fn = limma_pkg.voom  # direct .default
    voom_out = voom_fn(counts_r, design_r, plot=plot, lib_size = lib_size, block = block, **kwargs)
    
    # voom_out is an EList-like list with $E and $weights slots
    E_r       = r.ro.baseenv["[["](voom_out, "E")
    weights_r = r.ro.baseenv["[["](voom_out, "weights")
    
    # wrap and insert into assays
    assays = dict(se.assays)
    assays[log_expr_assay]    = RMatrixAdapter(E_r, r)
    assays[weights_assay]     = RMatrixAdapter(weights_r, r)
    
    return RESummarizedExperiment(
        assays=assays,
        row_data=se.row_data_df,
        column_data=se.column_data_df,
        row_names=se.row_names,
        column_names=se.column_names,
        metadata=dict(se.metadata),
    )