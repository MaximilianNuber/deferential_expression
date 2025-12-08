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
    """Turn a batch spec into an R character vector.

    - If `batch` is a string → interpret as column name in column_data_df.
    - Otherwise, treat as sequence/array of labels.
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
    """Run `limma::removeBatchEffect` to correct an expression assay."""
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


# ——— 4) removeBatchEffect.default ———
# def remove_batch_effect(
#     se: RESummarizedExperiment,
#     batch: Union[Sequence, str],
#     batch2: Sequence | str | None = None,
#     exprs_assay: str = "log_expr",
#     corrected_assay: str = "log_expr_bc",
#     design: Optional[pd.DataFrame] = None,
#     **kwargs
# ) -> RESummarizedExperiment:
#     """Run `limma::removeBatchEffect` to correct an expression assay.

#     Args:
#         se: Input `RESummarizedExperiment` with an R-backed expression assay.
#         batch: Batch labels per sample (length = n_samples).
#         exprs_assay: Name of the input expression assay to correct.
#         corrected_assay: Name for the output batch-corrected assay.
#         design: Optional design matrix (samples × covariates) used as covariates
#             to protect biological signal during batch correction.
#         **kwargs: Additional keyword arguments forwarded to `removeBatchEffect`.

#     Returns:
#         RESummarizedExperiment: A new object with the batch-corrected assay
#         stored under `corrected_assay` as an `RMatrixAdapter`.
#     """
#     limma = _limma()
#     _r = get_r_environment()
#     E_r = se.assay_r(exprs_assay)

#     # batch → R
#     if isinstance(batch, str):
#         batch = se.column_data[batch]
#     batch = np.asarray(batch, dtype=str)
#     batch_r = _r.StrVector(batch)

#     if batch2 is not None:
#         if isinstance(batch2, str):
#             batch2 = se.column_data[batch2]
#         batch2 = np.asarray(batch2, dtype=str)
#         batch2_r = _r.StrVector(batch2)
#     else:
#         batch2_r = _r.ro.NULL

#     # design optional
#     design_r = pandas_to_r_matrix(design) if design is not None else _r.ro.NULL
#     print(type(design_r))

#     rbe = limma.removeBatchEffect
#     out_r = rbe(
#         E_r, 
#         batch=batch_r, 
#         batch2 = batch2_r,
#         # design=design_r, 
#         **kwargs
#     )
#     assays = dict(se.assays)
#     assays[corrected_assay] = RMatrixAdapter(out_r, _r)
#     return RESummarizedExperiment(
#         assays=assays,
#         row_data=se.row_data_df,
#         column_data=se.column_data_df,
#         row_names=se.row_names,
#         column_names=se.column_names,
#         metadata=dict(se.metadata),
#     )