"""
Run quasi-likelihood F-test using edgeR::glmQLFTest.

This module provides a functional interface to perform F-tests on
an EdgeRModel and return results as a pandas DataFrame.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd

from .utils import _prep_edger
from .checks import check_edger_model
from .glm_ql_fit import EdgeRModel


def glm_ql_ftest(
    model: EdgeRModel,
    coef: Optional[Union[str, int]] = None,
    contrast: Optional[Sequence] = None,
    poisson_bound: bool = True,
    adjust_method: str = "BH",
) -> pd.DataFrame:
    """
    Run quasi-likelihood F-test on a fitted EdgeRModel.
    
    Wraps ``edgeR::glmQLFTest``. Returns a DataFrame with test results
    including log fold-changes, p-values, and FDR-adjusted p-values.
    
    Args:
        model: EdgeRModel from glm_ql_fit().
        coef: Coefficient name (str) or index (int, 1-based) to test.
            Either coef or contrast must be specified.
        contrast: Contrast vector (alternative to coef).
            Either coef or contrast must be specified.
        poisson_bound: Whether to apply Poisson bound. Default: True.
        adjust_method: Multiple testing adjustment method. Default: "BH".
    
    Returns:
        pd.DataFrame: Results table with columns including:
            - logFC: log fold-change
            - logCPM: log counts per million
            - F: F-statistic
            - PValue: raw p-value
            - FDR: adjusted p-value
    
    Raises:
        TypeError: If model is not an EdgeRModel.
        ValueError: If model.fit is None or neither coef nor contrast is specified.
    
    Example:
        >>> import deferential_expression.edger as edger
        >>> model = edger.glm_ql_fit(rse, design)
        >>> results = edger.glm_ql_ftest(model, coef=2)
        >>> sig_genes = results[results["FDR"] < 0.05]
    """
    # Validate inputs
    check_edger_model(model)
    
    if coef is None and contrast is None:
        # Default to last coefficient if neither specified
        if model.design is not None:
            coef = model.design.shape[1]  # Last column (1-based in R)
        else:
            raise ValueError("Either `coef` or `contrast` must be specified")
    
    r, pkg = _prep_edger()
    
    # Handle coef
    if coef is not None:
        if isinstance(coef, int):
            coef_r = r.IntVector([coef])
        else:
            coef_r = r.StrVector([str(coef)])
    else:
        coef_r = r.ro.NULL
    
    # Handle contrast
    if contrast is not None:
        contrast_r = r.IntVector(np.asarray(contrast, dtype=int))
    else:
        contrast_r = r.ro.NULL
    
    poisson_bound_r = r.BoolVector([poisson_bound])
    
    # Run test
    res = pkg.glmQLFTest(
        model.fit,
        coef=coef_r,
        contrast=contrast_r,
        poisson_bound=poisson_bound_r
    )
    
    # Extract results via topTags
    res = pkg.topTags(
        res,
        n=r.ro.r("Inf"),
        adjust_method=adjust_method,
        sort_by=r.ro.NULL,
        p_value=r.IntVector([1])
    )
    
    # Convert to pandas
    res = r.ro.baseenv["as.data.frame"](res)
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        df = r.get_conversion().rpy2py(res)
    
    return df
