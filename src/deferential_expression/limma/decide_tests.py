"""
Classify genes using limma::decideTests.

This module provides a functional interface to classify genes as up/down/not significant.
"""

from __future__ import annotations
from typing import Any
import pandas as pd

from .utils import _limma
from .checks import check_limma_model
from .lm_fit import LimmaModel


def decide_tests(
    model: LimmaModel,
    method: str = "separate",
    adjust_method: str = "BH",
    p_value: float = 0.05,
    lfc: float = 0,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Classify genes as significantly up, down, or not significant.
    
    Wraps ``limma::decideTests``. Will run eBayes if not already done.
    
    Args:
        model: LimmaModel (will run eBayes if ebayes slot not set).
        method: Testing method ("separate", "global", etc.). Default: "separate".
        adjust_method: Multiple testing method. Default: "BH".
        p_value: Significance threshold. Default: 0.05.
        lfc: Log-fold-change threshold. Default: 0.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        pd.DataFrame: Values -1 (down), 0 (not significant), 1 (up).
            Index: gene names, Columns: coefficient names.
    
    Example:
        >>> import deferential_expression.limma as limma
        >>> model = limma.lm_fit(rse_voom, design).e_bayes()
        >>> dt = limma.decide_tests(model, p_value=0.01)
        >>> n_up = (dt == 1).sum()
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from .e_bayes import e_bayes
    
    check_limma_model(model)
    
    r = get_r_environment()
    limma_pkg = _limma()
    
    # Use ebayes if available, otherwise compute it
    if model.ebayes is not None:
        eb = model.ebayes
    else:
        model = e_bayes(model)
        eb = model.ebayes
    
    call_kwargs = {
        "method": method,
        "adjust.method": adjust_method,
        "p.value": p_value,
        "lfc": lfc
    }
    call_kwargs.update(kwargs)
    
    decide_r = limma_pkg.decideTests(eb, **call_kwargs)
    
    r_rownames = r.ro.baseenv["rownames"](decide_r)
    r_colnames = r.ro.baseenv["colnames"](decide_r)
    
    rownames = list(r.r2py(r_rownames)) if r_rownames is not r.ro.NULL else None
    colnames = list(r.r2py(r_colnames)) if r_colnames is not r.ro.NULL else None
    
    arr = r.ro.baseenv["as.matrix"](decide_r)
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        matrix_arr = r.r2py(arr)
    
    return pd.DataFrame(matrix_arr, index=rownames, columns=colnames)
