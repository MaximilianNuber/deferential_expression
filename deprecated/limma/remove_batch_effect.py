import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union

from .utils import _limma
from bioc2ri.lazy_r_env import get_r_environment
from ..resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter
from ..edger.utils import pandas_to_r_matrix
from rpy2.robjects.vectors import StrVector


def _resolve_batch(
    se: RESummarizedExperiment,
    batch: Union[str, Sequence, np.ndarray, pd.Series],
) -> StrVector:
    """Convert batch specification to an R character vector.

    Args:
        se: ``RESummarizedExperiment`` containing column metadata.
        batch: Batch specification. If string, interpreted as column name in
            ``column_data``. Otherwise, treated as sequence/array of batch labels.

    Returns:
        StrVector: rpy2 R character vector suitable for limma functions.

    Raises:
        KeyError: If ``batch`` is a string but not found in ``column_data``.
    """
    if isinstance(batch, str):
        cd = se.column_data_df
        if cd is None or batch not in cd.columns:
            raise KeyError(f"Batch column '{batch}' not found in column_data.")
        arr = cd[batch].to_numpy()
    else:
        arr = np.asarray(batch)

    # Force to plain Python strings
    vals = ["" if v is None else str(v) for v in arr.tolist()]
    return StrVector(vals)


def remove_batch_effect(
    se: RESummarizedExperiment,
    batch: Union[str, Sequence, np.ndarray, pd.Series],
    batch2: Union[str, Sequence, np.ndarray, pd.Series, None] = None,
    exprs_assay: str = "log_expr",
    corrected_assay: str = "log_expr_bc",
    design: Optional[pd.DataFrame] = None,
    covariates: Optional[pd.DataFrame] = None,
    **kwargs,
) -> RESummarizedExperiment:
    """Remove batch effects from expression data.

    Wraps the R ``limma::removeBatchEffect`` function to adjust for batch effects
    while preserving biological variation. Can handle one or two batch factors and
    optional covariates to protect during batch correction.

    Args:
        se: Input ``RESummarizedExperiment`` with an R-backed expression assay.
        batch: Primary batch factor. Either a column name in ``column_data`` (string)
            or an array-like of batch labels (length = n_samples).
        batch2: Optional secondary batch factor. Same format as ``batch``.
        exprs_assay: Name of the input expression assay to correct. Default: ``"log_expr"``.
        corrected_assay: Name for the output batch-corrected assay. Default: ``"log_expr_bc"``.
        design: Optional design matrix (samples × covariates) as pandas DataFrame.
            Biological factors in the design are preserved during batch correction.
        covariates: Optional continuous covariates matrix as pandas DataFrame to adjust for.
        **kwargs: Additional keyword arguments forwarded to ``limma::removeBatchEffect``.

    Returns:
        RESummarizedExperiment: New instance with the batch-corrected assay stored under
            ``corrected_assay`` as an R-backed matrix (``RMatrixAdapter``).

    Raises:
        KeyError: If ``batch`` or ``batch2`` is a string but not found in ``column_data``.

    Notes:
        - Original object remains unchanged (functional/immutable style).
        - The input assay must be R-backed and accessible via ``se.assay_r(exprs_assay)``.
        - Batch correction is typically applied to log-transformed data.

    Examples:
        >>> se_corrected = remove_batch_effect(se, batch="batch_id", design=design_df)
        >>> se_corrected.assay_names
        ['log_expr', 'log_expr_bc']
    """
    limma = _limma()
    _r = get_r_environment()

    # 1) main expression matrix from RESummarizedExperiment
    E_r = se.assay_r(exprs_assay)

    # 2) required argument: batch
    batch_r = _resolve_batch(se, batch)

    call_kwargs = {"batch": batch_r}

    # 3) optional batch2
    if batch2 is not None:
        batch2_r = _resolve_batch(se, batch2)
        call_kwargs["batch2"] = batch2_r

    # 4) optional covariates (pandas DataFrame -> R matrix)
    if covariates is not None:
        call_kwargs["covariates"] = pandas_to_r_matrix(covariates)

    # 5) optional design (pandas DataFrame -> R matrix)
    if design is not None:
        call_kwargs["design"] = pandas_to_r_matrix(design)

    # 6) extra kwargs (drop any that are None so they don't reach R)
    extra = {k: v for k, v in kwargs.items() if v is not None}
    call_kwargs.update(extra)

    # print("call_kwargs keys:", call_kwargs.keys())
    # for k, v in call_kwargs.items():
    #     print(k, type(v))

    # 7) call limma::removeBatchEffect
    out_r = limma.removeBatchEffect(E_r, **call_kwargs)

    # 8) wrap back into RESummarizedExperiment
    assays = dict(se.assays)
    assays[corrected_assay] = RMatrixAdapter(out_r, _r)

    return RESummarizedExperiment(
        assays=assays,
        row_data=se.row_data_df,
        column_data=se.column_data_df,
        row_names=se.row_names,
        column_names=se.column_names,
        metadata=dict(se.metadata),
    )