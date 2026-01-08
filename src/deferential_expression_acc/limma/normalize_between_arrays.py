import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from .utils import _limma
from bioc2ri.lazy_r_env import get_r_environment
from ..resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter
from ..edger.utils import numpy_to_r_matrix, pandas_to_r_matrix

# ——— 3) normalizeBetweenArrays.default ———
def normalize_between_arrays(
    se: "RESummarizedExperiment",
    exprs_assay: str = "log_expr",
    normalized_assay: str = "log_expr_norm",
    method: str = "quantile",
    **kwargs
) -> "RESummarizedExperiment":
    """Normalize expression values between arrays/samples.

    Wraps the R ``limma::normalizeBetweenArrays`` function to perform between-sample
    normalization on an expression assay. Common methods include quantile normalization,
    scaling, and cyclical loess.

    Args:
        se: Input ``RESummarizedExperiment`` with an R-backed expression assay.
        exprs_assay: Name of the input expression assay to normalize. Default: ``"log_expr"``.
        normalized_assay: Name for the output normalized assay. Default: ``"log_expr_norm"``.
        method: Normalization method. Options include ``"quantile"``, ``"scale"``,
            ``"cyclicloess"``, ``"Aquantile"``, ``"Gquantile"``, ``"Rquantile"``,
            ``"Tquantile"``, or ``"none"``. Default: ``"quantile"``.
        **kwargs: Additional keyword arguments forwarded to ``limma::normalizeBetweenArrays``.

    Returns:
        RESummarizedExperiment: New instance with the normalized assay stored under
            ``normalized_assay`` as an R-backed matrix (``RMatrixAdapter``).

    Notes:
        - Original object remains unchanged (functional/immutable style).
        - The input assay must be R-backed and accessible via ``se.assay_r(exprs_assay)``.

    Examples:
        >>> se_norm = normalize_between_arrays(se, method="quantile")
        >>> se_norm.assay_names
        ['log_expr', 'log_expr_norm']
    """
    limma = _limma()
    _r = get_r_environment()
    # get the E matrix (either R mat or python)
    exprs_r = se.assay_r(exprs_assay)
    norm_fn = limma.normalizeBetweenArrays
    out_r = norm_fn(exprs_r, method=method, **kwargs)
    assays = dict(se.assays)
    assays[normalized_assay] = RMatrixAdapter(out_r, _r)
    return RESummarizedExperiment(
        assays=assays,
        row_data=se.row_data_df,
        column_data=se.column_data_df,
        row_names=se.row_names,
        column_names=se.column_names,
        metadata=dict(se.metadata),
    )