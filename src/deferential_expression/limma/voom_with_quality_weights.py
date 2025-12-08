import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from .utils import _limma
from bioc2ri.lazy_r_env import get_r_environment
from ..resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter
from ..edger.utils import numpy_to_r_matrix, pandas_to_r_matrix

# ——— 2) voomWithQualityWeights.default — same pattern ———
def voom_with_quality_weights(
    se: RESummarizedExperiment,
    design: pd.DataFrame,
    lib_size: Optional[Union[pd.Series, Sequence, np.ndarray]] = None,

    log_expr_assay: str = "log_expr",
    weights_assay: str = "weights",
    plot: bool = False,
    **kwargs
) -> RESummarizedExperiment:
    """Run `limma::voomWithQualityWeights` (default) to compute log-CPM and QW weights.

    Args:
        se: Input `RESummarizedExperiment` containing a `"counts"` R-backed assay.
        design: Design matrix (samples × covariates) as a pandas DataFrame.
        log_expr_assay: Name for the output log-expression assay.
        weights_assay: Name for the output weights assay.
        plot: Whether to enable voom’s diagnostic plot in R.
        **kwargs: Additional keyword arguments forwarded to `voomWithQualityWeights`.

    Returns:
        RESummarizedExperiment: A new object with `log_expr_assay` and `weights_assay`
        assays added as `RMatrixAdapter`s.
    """
    limma = _limma()
    _r = get_r_environment()

    # extract raw counts R matrix
    counts_r = se.assay_r("counts")

    assert isinstance(design, pd.DataFrame), "design must be a pandas DataFrame"
    design_r = pandas_to_r_matrix(design)

    if lib_size is not None:
        lib_size = np.asarray(lib_size, dtype=float)
        assert lib_size.ndim == 1, "lib_size must be a 1D array-like"
        lib_size = _r.FloatVector(lib_size)
    else:
        if "norm.factors" in se.column_data.column_names:
            # use norm factors as lib_size if available
            lib_size = se.column_data["norm.factors"]
            lib_size = np.asarray(lib_size, dtype=float)
            
            lib_size = _r.FloatVector(lib_size)
        else:
            lib_size = _r.ro.NULL

    voom_qw_fn = limma.voomWithQualityWeights
    out = voom_qw_fn(counts_r, design_r, plot=plot, **kwargs)
    E_r       = _r.ro.baseenv["[["](out, "E")
    weights_r = _r.ro.baseenv["[["](out, "weights")
    assays = dict(se.assays)
    assays[log_expr_assay]    = RMatrixAdapter(E_r, _r)
    assays[weights_assay]     = RMatrixAdapter(weights_r, _r)
    return RESummarizedExperiment(
        assays=assays,
        row_data=se.row_data_df,
        column_data=se.column_data_df,
        row_names=se.row_names,
        column_names=se.column_names,
        metadata=dict(se.metadata),
    )