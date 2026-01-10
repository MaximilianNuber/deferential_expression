"""
Extract top-ranked genes using limma::topTable.

This module provides a functional interface to extract DE results.
"""

from __future__ import annotations
from typing import Any, Optional, Union
import pandas as pd

from .utils import _limma
from .checks import check_limma_model
from .lm_fit import LimmaModel


def top_table(
    model: LimmaModel,
    coef: Optional[Union[int, str]] = None,
    n: Optional[int] = None,
    adjust_method: str = "BH",
    sort_by: str = "PValue",
    **kwargs: Any
) -> pd.DataFrame:
    """
    Extract top-ranked genes from differential expression analysis.
    
    Wraps ``limma::topTable``. Will run eBayes if not already done.
    
    Args:
        model: LimmaModel (will run eBayes if ebayes slot not set).
        coef: Coefficient to extract (1-based index or name).
        n: Number of top genes (None = all).
        adjust_method: Multiple testing method. Default: "BH".
        sort_by: Sort column. Default: "PValue".
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        pd.DataFrame: Results table with columns:
            - gene: gene identifier
            - log_fc: log fold-change
            - ave_expr: average expression
            - t_statistic: t-statistic
            - p_value: raw p-value
            - adj_p_value: adjusted p-value
            - b_statistic: B-statistic
    
    Example:
        >>> import deferential_expression.limma as limma
        >>> model = limma.lm_fit(rse_voom, design).e_bayes()
        >>> results = limma.top_table(model, coef=2, n=100)
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
    
    if n is None:
        n = int(r.r2py(r.ro.baseenv["nrow"](eb)))
    
    # Handle coef for contrast fits
    if model.contrast_fit is not None and coef is None:
        coef = 1
    
    # Map sort_by
    if coef is None:
        sort_by_map = {"PValue": "F", "logFC": "F", "AveExpr": "F", "B": "F", "F": "F", "none": "none"}
        sort_by_r = sort_by_map.get(sort_by, "F")
        coef_r = r.ro.NULL
    else:
        sort_by_map = {"PValue": "p", "logFC": "logFC", "AveExpr": "AveExpr", "B": "B", "F": "B", "none": "none"}
        sort_by_r = sort_by_map.get(sort_by, "p")
        coef_r = coef
    
    call_kwargs = {"number": n, "adjust.method": adjust_method, "coef": coef_r, "sort.by": sort_by_r}
    call_kwargs.update(kwargs)
    
    top_r = limma_pkg.topTable(eb, **call_kwargs)
    
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        df = r.get_conversion().rpy2py(top_r)
    
    return df.reset_index(names="gene").rename(columns={
        'P.Value': 'p_value',
        'logFC': 'log_fc',
        'adj.P.Val': 'adj_p_value',
        'AveExpr': 'ave_expr',
        't': 't_statistic',
        'B': 'b_statistic'
    })
