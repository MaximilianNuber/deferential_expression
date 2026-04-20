"""
Run voomWithQualityWeights transformation using limma::voomWithQualityWeights.

This module provides a functional interface to perform voom transformation
with sample-specific quality weights and store results in a
SummarizedExperiment.
"""

from __future__ import annotations
from typing import Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd

from .utils import _limma
from .checks import check_se, check_assay_exists, check_r_assay, check_design

SE = TypeVar("SE")


def voom_with_quality_weights(
    se: SE,
    design: pd.DataFrame,
    assay: str = "counts",
    lib_size: Optional[Union[pd.Series, Sequence, np.ndarray]] = None,
    normalize_method: str = "none",
    span: float = 0.5,
    var_design: Optional[pd.DataFrame] = None,
    var_group: Optional[Union[pd.Series, Sequence, np.ndarray, pd.Categorical]] = None,
    method: str = "genebygene",
    maxiter: int = 50,
    tol: float = 1e-5,
    trace: bool = False,
    plot: bool = False,
    col: Optional[Union[str, Sequence[str]]] = None,
    log_expr_assay: str = "log_expr",
    weights_assay: str = "weights",
    sample_weights_col: str = "sample_weights",
    in_place: bool = False,
    **kwargs,
) -> SE:
    """
    Run voomWithQualityWeights transformation on counts.

    Wraps ``limma::voomWithQualityWeights``. Results are stored as:
    - assay ``log_expr_assay``: log2-CPM values
    - assay ``weights_assay``: observation-level precision weights
    - column_data ``sample_weights_col``: sample-specific quality weights

    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The input assay must be R-initialized using initialize_r() first.

    Args:
        se:
            Input SummarizedExperiment with R-initialized count assay.
        design:
            Design matrix (samples × covariates) as pandas DataFrame.
        assay:
            Input counts assay name. Default: ``"counts"``.
        lib_size:
            Optional library sizes per sample.
        normalize_method:
            Microarray-style normalization method passed to limma.
            Default: ``"none"``.
        span:
            Lowess span for voom mean-variance trend. Default: 0.5.
        var_design:
            Optional design matrix for variance model used by arrayWeights.
        var_group:
            Optional grouping factor for groupwise sample weights.
        method:
            Algorithm for arrayWeights. One of ``"genebygene"``, ``"reml"``,
            or ``"auto"`` where supported. Default: ``"genebygene"``.
        maxiter:
            Maximum iterations for REML algorithm. Default: 50.
        tol:
            Convergence tolerance. Default: 1e-5.
        trace:
            Whether to print progress from arrayWeights. Default: False.
        plot:
            Whether to show diagnostic plots. Default: False.
        col:
            Optional bar colors for sample-weight plot.
        log_expr_assay:
            Name for output log-expression assay. Default: ``"log_expr"``.
        weights_assay:
            Name for output weights assay. Default: ``"weights"``.
        sample_weights_col:
            Name for per-sample quality weights in column_data.
            Default: ``"sample_weights"``.
        in_place:
            If True, modify ``se`` in place. Default: False.
        **kwargs:
            Additional arguments forwarded to ``limma::voomWithQualityWeights``.
            These may include arguments accepted by limma's internal voom calls,
            e.g. ``block`` or ``correlation``.

    Returns:
        SummarizedExperiment with log-expression assay, observation weights,
        and per-sample quality weights.

    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.limma as limma
        >>> se = initialize_r(se, assay="counts")
        >>> se = limma.voom_with_quality_weights(se, design)
        >>> log_expr = np.asarray(se.assays["log_expr"])
        >>> obs_weights = np.asarray(se.assays["weights"])
        >>> sample_weights = np.asarray(se.get_column_data()["sample_weights"])
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from ..edger.utils import pandas_to_r_matrix
    from ..r_init import get_rmat
    from ..rmatrixadapter import RMatrixAdapter

    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)

    n_samples = len(se.column_names) if se.column_names else se.shape[1]
    check_design(design, n_samples)

    if var_design is not None:
        check_design(var_design, n_samples)

    r = get_r_environment()
    limma_pkg = _limma()

    counts_r = get_rmat(se, assay)
    design_r = pandas_to_r_matrix(design)

    # lib.size
    if lib_size is not None:
        lib_size = np.asarray(lib_size, dtype=float)
        if lib_size.shape[0] != n_samples:
            raise ValueError(
                f"lib_size must have length {n_samples}, got {lib_size.shape[0]}."
            )
        lib_size_r = r.FloatVector(lib_size)
    else:
        lib_size_r = r.ro.NULL

    # var.design
    if var_design is not None:
        var_design_r = pandas_to_r_matrix(var_design)
    else:
        var_design_r = r.ro.NULL

    # var.group
    if var_group is not None:
        var_group = np.asarray(var_group)
        if var_group.shape[0] != n_samples:
            raise ValueError(
                f"var_group must have length {n_samples}, got {var_group.shape[0]}."
            )

        # Use factor conversion on purpose, because limma documents var.group
        # as a vector/factor indicating shared variance groups.
        with r.localconverter(r.default_converter + r.pandas2ri.converter):
            var_group_r = r.get_conversion().py2rpy(
                pd.Categorical(var_group)
            )
    else:
        var_group_r = r.ro.NULL

    # col
    if col is None:
        col_r = r.ro.NULL
    elif isinstance(col, str):
        col_r = r.ro.StrVector([col])
    else:
        col_r = r.ro.StrVector(list(col))

    # Run limma::voomWithQualityWeights
    vq = limma_pkg.voomWithQualityWeights(
        counts_r,
        design=design_r,
        lib_size=lib_size_r,
        normalize_method=normalize_method,
        plot=plot,
        span=span,
        var_design=var_design_r,
        var_group=var_group_r,
        method=method,
        maxiter=maxiter,
        tol=tol,
        trace=trace,
        col=col_r,
        **kwargs,
    )

    # Extract components from returned EList
    E_r = r.ro.baseenv["[["](vq, "E")
    weights_r = r.ro.baseenv["[["](vq, "weights")

    # sample.weights lives in vq$targets$sample.weights in limma
    targets_r = r.ro.baseenv["[["](vq, "targets")
    sample_weights_r = r.ro.baseenv["$"](targets_r, "sample.weights")
    sample_weights = np.asarray(sample_weights_r, dtype=float)

    output = se._define_output(in_place=in_place)

    # assays
    new_assays = dict(output.assays)
    new_assays[log_expr_assay] = RMatrixAdapter(E_r, r)
    new_assays[weights_assay] = RMatrixAdapter(weights_r, r)
    output._assays = new_assays

    # column_data
    coldata = output.get_column_data()
    if coldata is None:
        coldata = pd.DataFrame(index=output.column_names)
    else:
        coldata = coldata.copy()

    coldata[sample_weights_col] = sample_weights
    output._column_data = coldata

    return output