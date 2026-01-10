"""
Apply empirical Bayes moderation using limma::eBayes.

This module provides a functional interface to compute moderated statistics.
"""

from __future__ import annotations
from typing import Any
from dataclasses import replace

from .utils import _limma
from .checks import check_limma_model_fitted
from .lm_fit import LimmaModel


def e_bayes(
    model: LimmaModel,
    proportion: float = 0.01,
    trend: bool = False,
    robust: bool = False,
    **kwargs: Any
) -> LimmaModel:
    """
    Compute empirical Bayes moderated statistics.
    
    Wraps ``limma::eBayes``. Returns a new LimmaModel with the ebayes slot set.
    
    Args:
        model: LimmaModel from lm_fit() or contrasts_fit().
        proportion: Assumed proportion of DE genes. Default: 0.01.
        trend: Fit mean-variance trend. Default: False.
        robust: Use robust empirical Bayes. Default: False.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        LimmaModel: With ebayes slot set.
    
    Raises:
        TypeError: If model is not a LimmaModel.
        ValueError: If model is not fitted.
    
    Example:
        >>> import deferential_expression.limma as limma
        >>> model = limma.lm_fit(rse_voom, design)
        >>> model_eb = limma.e_bayes(model, robust=True)
        >>> results = limma.top_table(model_eb)
    """
    check_limma_model_fitted(model)
    
    limma_pkg = _limma()
    
    r_fit = model.contrast_fit if model.contrast_fit is not None else model.lm_fit
    
    call_kwargs = {"proportion": proportion, "trend": trend, "robust": robust}
    call_kwargs.update(kwargs)
    
    eb = limma_pkg.eBayes(r_fit, **call_kwargs)
    
    return replace(model, ebayes=eb)
