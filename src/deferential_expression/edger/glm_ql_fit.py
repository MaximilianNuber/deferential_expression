import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Union
from dataclasses import dataclass
from deferential_expression.edger.utils import _prep_edger, numpy_to_r_matrix, pandas_to_r_matrix

@dataclass
class GlmQlFitConfig:
    dispersion: Optional[Union[pd.DataFrame, np.ndarray]] = None
    offset: Optional[Union[pd.DataFrame, np.ndarray]] = None
    lib_size: Optional[Sequence] = None
    weights: Optional[Union[int, float, Sequence[float], pd.DataFrame, np.ndarray]] = None
    legacy: bool = False
    top_proportion: float = 0.1
    assay: str = "counts"
    user_kwargs: Dict[str, Any] = None

@dataclass
class EdgeRModel:
    """Container for limma results including fit, coefficients, and metadata."""
    sample_names: Optional[Sequence[str]] = None  # Sample names (column names)
    feature_names: Optional[Sequence[str]] = None  # Feature names (row names)
    fit: Optional[Any] = None  # R object from lmFit
    fit_config: Optional[Any] = None  # Configuration used for fitting

    design: Optional[pd.DataFrame] = None  # Design matrix used for fitting

    coefficients: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None

def _glm_ql_fit_impl(
    rmat,
    design_r,
    dispersion_r,
    offset,
    weights,
    legacy,
    top_proportion,
    **user_kwargs
):
    r, pkg = _prep_edger()
    return pkg.glmQLFit_default(
        rmat, 
        design = design_r,
        dispersion = dispersion_r,
        offset = offset,
        weights = weights, 
        legacy = legacy,
        top_proportion = top_proportion,
        **user_kwargs
    )

def glm_ql_fit(
    obj: "RESummarizedExperiment", 
    design: pd.DataFrame, 
    *,
    dispersion: pd.DataFrame | np.ndarray | None = None, 
    offset: pd.DataFrame | np.ndarray | None = None,
    lib_size: Sequence | None = None,
    weights: int | float | Sequence[float] | pd.DataFrame | np.ndarray | None = None,
    legacy: bool = False,
    top_proportion: 'float' = 0.1,
    assay: 'str' = 'counts',
    **user_kwargs
):
    """Functional ``edgeR::glmQLFit`` with optional dispersion/offset/weights.

    Args:
        obj: ``EdgeR`` instance with a counts assay.
        design: Design matrix (samples × covariates) as pandas DataFrame.
        dispersion: Optional per-observation dispersion values/matrix.
        offset: Optional offset matrix (e.g., log library sizes).
        lib_size: Optional library sizes per sample.
        weights: Optional observation/sample weights.
        legacy: Pass-through boolean to the underlying R function (if supported).
        top_proportion: Pass-through numeric parameter for robust fitting.
        assay: Counts assay name.
        **user_kwargs: Additional args forwarded to the R implementation.

    Returns:
        EdgeR: New object with ``glm`` fit stored.

    Notes:
        Inputs that are pandas/NumPy are converted to R matrices/vectors as needed.
    """
    
    config = GlmQlFitConfig(
        dispersion = dispersion,
        offset = offset,
        lib_size = lib_size,
        weights = weights,
        legacy = legacy,
        top_proportion = top_proportion,
        assay = assay,
        user_kwargs = user_kwargs
    )
    

    r, pkg = _prep_edger()
    rmat = obj.assay_r(assay)

    design_r = pandas_to_r_matrix(design)

    if dispersion is not None:
        if isinstance(dispersion, pd.DataFrame):
            dispersion = pandas_to_r_matrix(dispersion)
        elif isinstance(dispersion, np.ndarray):
            dispersion = numpy_to_r_matrix(dispersion)
    else:
        dispersion = r.ro.NULL

    if offset is not None:
        if isinstance(offset, pd.DataFrame):
            offset = pandas_to_r_matrix(offset)
        elif isinstance(offset, np.ndarray):
            offset = numpy_to_r_matrix(offset)
    else:
        if "norm.factors" in obj.column_data.column_names:
            # use norm factors as offset if available
            offset = obj.column_data["norm.factors"]
            offset = np.asarray(offset, dtype=float)
            offset = r.FloatVector(offset)
        else:
            offset = r.ro.NULL

    if lib_size is not None:
        lib_size = r.FloatVector(np.asarray(lib_size, dtype = float))
    else: 
        lib_size = r.ro.NULL

    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            weights = pandas_to_r_matrix(weights)
        elif isinstance(weights, np.ndarray):
            weights = numpy_to_r_matrix(weights)
        elif isinstance(weights, int):
            weights = r.ro.IntVector([weights])
        elif isinstance(weights, float):
            weights = r.ro.FloatVector([weights])
        elif isinstance(weights, Sequence):
            weights = r.ro.FloatVector(
                np.asarray(weights, dtype = float)
            )
    else:
        weights = r.ro.NULL

    fit_obj = _glm_ql_fit_impl(
        rmat,
        design_r,
        dispersion,
        offset,
        weights,
        legacy,
        top_proportion,
        **user_kwargs
    )
    
    # return obj._clone(glm = fit_obj)
    return EdgeRModel(
        sample_names = obj.column_names,
        feature_names = obj.row_names,
        fit = fit_obj,
        fit_config = config,
        design = design,
        
    )