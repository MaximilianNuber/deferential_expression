"""
Apply TREAT using limma::treat.

This module provides a functional interface for fold-change threshold testing.
"""

from __future__ import annotations
from typing import Any
from dataclasses import replace

from .utils import _limma
from .checks import check_limma_model_fitted
from .lm_fit import LimmaModel


def treat(
    model: LimmaModel,
    lfc: float = 1.0,
    robust: bool = False,
    trend: bool = False,
    **kwargs: Any
) -> LimmaModel:
    """
    Test for differential expression relative to a fold-change threshold.
    
    Wraps ``limma::treat``. Tests for |logFC| > lfc threshold.
    
    Args:
        model: LimmaModel from lm_fit() or contrasts_fit().
        lfc: Log-fold-change threshold. Default: 1.0.
        robust: Use robust empirical Bayes. Default: False.
        trend: Fit mean-variance trend. Default: False.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        LimmaModel: With TREAT fit (replaces lm_fit slot).
    
    Example:
        >>> import deferential_expression.limma as limma
        >>> model = limma.lm_fit(rse_voom, design)
        >>> model_treat = limma.treat(model, lfc=1.0)
        >>> results = model_treat.top_table()
    """
    check_limma_model_fitted(model)
    
    limma_pkg = _limma()
    
    r_fit = model.contrast_fit if model.contrast_fit is not None else model.lm_fit
    
    call_kwargs = {"lfc": lfc, "robust": robust, "trend": trend}
    call_kwargs.update(kwargs)
    
    treat_fit = limma_pkg.treat(r_fit, **call_kwargs)
    
    return replace(model, lm_fit=treat_fit, method="treat")
