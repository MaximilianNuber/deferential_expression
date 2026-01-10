"""
Apply contrasts using limma::contrasts.fit.

This module provides a functional interface to apply contrasts to a fitted model.
"""

from __future__ import annotations
from typing import Sequence, Union
from dataclasses import replace
import numpy as np

from .utils import _limma
from .checks import check_limma_model_fitted
from .lm_fit import LimmaModel


def contrasts_fit(
    model: LimmaModel,
    contrast: Sequence[Union[int, float]],
) -> LimmaModel:
    """
    Apply contrast to fitted linear model.
    
    Wraps ``limma::contrasts.fit``. Returns LimmaModel with contrast_fit slot set.
    
    Args:
        model: LimmaModel from lm_fit().
        contrast: 1D contrast vector (length = number of coefficients).
    
    Returns:
        LimmaModel: With contrast_fit slot set.
    
    Raises:
        TypeError: If model is not a LimmaModel.
        ValueError: If model is not fitted.
    
    Example:
        >>> import deferential_expression.limma as limma
        >>> model = limma.lm_fit(rse_voom, design)
        >>> model_c = limma.contrasts_fit(model, [0, 1, -1])
        >>> results = model_c.e_bayes().top_table()
    """
    from bioc2ri.lazy_r_env import get_r_environment
    
    check_limma_model_fitted(model)
    
    r = get_r_environment()
    limma_pkg = _limma()
    
    contrast_arr = np.asarray(contrast, dtype=float)
    contrast_r = r.ro.FloatVector(contrast_arr)
    
    fit_r = limma_pkg.contrasts_fit(model.lm_fit, contrast=contrast_r)
    
    return replace(model, contrast_fit=fit_r)
