"""
Fit quasi-likelihood GLM using edgeR::glmQLFit.

This module provides the EdgeRModel dataclass for storing fit results
and the glm_ql_fit function for fitting the model.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, TypeVar, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .utils import _prep_edger, numpy_to_r_matrix, pandas_to_r_matrix
from .checks import check_se, check_assay_exists, check_r_assay, check_design

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


@dataclass
class GlmQlFitConfig:
    """Configuration used for GLM QL fitting."""
    dispersion: Optional[Union[pd.DataFrame, np.ndarray]] = None
    offset: Optional[Union[pd.DataFrame, np.ndarray]] = None
    lib_size: Optional[Sequence] = None
    weights: Optional[Union[int, float, Sequence[float], pd.DataFrame, np.ndarray]] = None
    legacy: bool = False
    top_proportion: float = 0.1
    assay: str = "counts"
    user_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class EdgeRModel:
    """Container for edgeR GLM fit results.
    
    This dataclass stores the R fit object and associated metadata from
    glm_ql_fit. Use with glm_ql_ftest() or top_tags() for downstream analysis.
    
    Attributes:
        sample_names: Sample names (column names) from the input SE.
        feature_names: Feature names (row names) from the input SE.
        fit: R object from glmQLFit.
        fit_config: Configuration used for fitting.
        design: Design matrix used for fitting.
        coefficients: Optional extracted coefficients.
        metadata: Optional additional metadata.
    """
    sample_names: Optional[Sequence[str]] = None
    feature_names: Optional[Sequence[str]] = None
    fit: Optional[Any] = None
    fit_config: Optional[GlmQlFitConfig] = None
    design: Optional[pd.DataFrame] = None
    coefficients: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None

    def glm_ql_ftest(
        self,
        coef: Optional[Union[str, int]] = None,
        contrast: Optional[Sequence] = None,
        poisson_bound: bool = True,
        adjust_method: str = "BH",
    ) -> pd.DataFrame:
        """
        Run quasi-likelihood F-test on this fitted model.
        
        Convenience method that delegates to the glm_ql_ftest function.
        
        Args:
            coef: Coefficient name (str) or index (int, 1-based) to test.
            contrast: Contrast vector (alternative to coef).
            poisson_bound: Whether to apply Poisson bound. Default: True.
            adjust_method: Multiple testing adjustment method. Default: "BH".
        
        Returns:
            pd.DataFrame: Results table with logFC, PValue, FDR, etc.
        
        Example:
            >>> model = edger.glm_ql_fit(se, design)
            >>> results = model.glm_ql_ftest(coef=2)
        """
        from .glm_ql_ftest import glm_ql_ftest as _glm_ql_ftest
        return _glm_ql_ftest(
            self,
            coef=coef,
            contrast=contrast,
            poisson_bound=poisson_bound,
            adjust_method=adjust_method,
        )


def glm_ql_fit(
    se: SE,
    design: pd.DataFrame,
    assay: str = "counts",
    dispersion: Optional[Union[pd.DataFrame, np.ndarray, float]] = None,
    offset: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    lib_size: Optional[Sequence] = None,
    weights: Optional[Union[float, Sequence[float], pd.DataFrame, np.ndarray]] = None,
    legacy: bool = False,
    top_proportion: float = 0.1,
    **kwargs
) -> EdgeRModel:
    """
    Fit quasi-likelihood GLM using edgeR::glmQLFit.
    
    Returns an EdgeRModel containing the fit object for downstream testing
    with glm_ql_ftest() or top_tags().
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized count assay.
        design: Design matrix (samples × covariates) as pandas DataFrame.
        assay: Counts assay name. Default: "counts".
        dispersion: Optional dispersion values/matrix.
        offset: Optional offset matrix (e.g., log library sizes).
        lib_size: Optional library sizes per sample.
        weights: Optional observation/sample weights.
        legacy: Legacy mode flag for R function. Default: False.
        top_proportion: Proportion parameter for robust fitting. Default: 0.1.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        EdgeRModel: Container with the fitted model.
    
    Raises:
        TypeError: If se lacks required attributes or design is not a DataFrame.
        KeyError: If the specified assay does not exist.
        ValueError: If design matrix rows don't match sample count.
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.edger as edger
        >>> import pandas as pd
        >>> se = initialize_r(se, assay="counts")
        >>> design = pd.DataFrame({'Intercept': [1]*6, 'Condition': [0,0,0,1,1,1]})
        >>> model = edger.glm_ql_fit(se, design)
        >>> results = edger.glm_ql_ftest(model, coef=2)
    """
    from ..r_init import get_rmat
    
    # Validate inputs
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    n_samples = len(se.column_names) if se.column_names else se.shape[1]
    check_design(design, n_samples)
    
    r, pkg = _prep_edger()
    rmat = get_rmat(se, assay)
    design_r = pandas_to_r_matrix(design)
    
    # Handle dispersion
    if dispersion is not None:
        if isinstance(dispersion, pd.DataFrame):
            dispersion_r = pandas_to_r_matrix(dispersion)
        elif isinstance(dispersion, np.ndarray):
            dispersion_r = numpy_to_r_matrix(dispersion)
        elif isinstance(dispersion, (int, float)):
            dispersion_r = r.FloatVector([float(dispersion)])
        else:
            dispersion_r = dispersion
    else:
        dispersion_r = r.ro.NULL
    
    # Handle offset - check for norm.factors in column_data
    if offset is not None:
        if isinstance(offset, pd.DataFrame):
            offset_r = pandas_to_r_matrix(offset)
        elif isinstance(offset, np.ndarray):
            offset_r = numpy_to_r_matrix(offset)
        else:
            offset_r = offset
    else:
        # Use norm.factors if available
        coldata = se.get_column_data()
        if coldata is not None and "norm.factors" in coldata.column_names:
            nf = np.asarray(coldata["norm.factors"], dtype=float)
            offset_r = r.FloatVector(nf)
        else:
            offset_r = r.ro.NULL
    
    # Handle lib_size
    lib_size_r = r.ro.NULL if lib_size is None else r.FloatVector(np.asarray(lib_size, dtype=float))
    
    # Handle weights
    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            weights_r = pandas_to_r_matrix(weights)
        elif isinstance(weights, np.ndarray):
            weights_r = numpy_to_r_matrix(weights)
        elif isinstance(weights, (int, float)):
            weights_r = r.FloatVector([float(weights)])
        elif isinstance(weights, Sequence):
            weights_r = r.FloatVector(np.asarray(weights, dtype=float))
        else:
            weights_r = weights
    else:
        weights_r = r.ro.NULL
    
    # Call glmQLFit_default
    fit_obj = pkg.glmQLFit_default(
        rmat,
        design=design_r,
        dispersion=dispersion_r,
        offset=offset_r,
        weights=weights_r,
        legacy=legacy,
        top_proportion=top_proportion,
        **kwargs
    )
    
    config = GlmQlFitConfig(
        dispersion=dispersion,
        offset=offset,
        lib_size=lib_size,
        weights=weights,
        legacy=legacy,
        top_proportion=top_proportion,
        assay=assay,
        user_kwargs=kwargs if kwargs else None
    )
    
    return EdgeRModel(
        sample_names=se.column_names,
        feature_names=se.row_names,
        fit=fit_obj,
        fit_config=config,
        design=design,
    )
