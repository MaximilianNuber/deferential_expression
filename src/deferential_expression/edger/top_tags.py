"""
Extract top-ranked genes using edgeR::topTags.

This module provides a functional interface to extract results from
an edgeR test object (e.g., from glmQLFTest) as a pandas DataFrame.
"""

from __future__ import annotations
from typing import Any, Optional
import pandas as pd

from .utils import _prep_edger


def top_tags(
    lrt_obj: Any,
    n: Optional[int] = None,
    adjust_method: str = "BH",
    sort_by: str = "PValue",
    **kwargs
) -> pd.DataFrame:
    """
    Extract top-ranked genes from a test result.
    
    Wraps ``edgeR::topTags``. Returns a DataFrame with standardized
    column names for convenient downstream analysis.
    
    Args:
        lrt_obj: R object from glmQLFTest or similar edgeR test.
        n: Number of top genes to return. None = all genes.
        adjust_method: Multiple testing correction method. Default: "BH".
        sort_by: Column to sort by. Default: "PValue".
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        pd.DataFrame: Results table with standardized columns:
            - gene: gene identifier (from row names)
            - log_fc: log fold-change
            - log_cpm: log counts per million
            - p_value: raw p-value
            - adj_p_value: adjusted p-value (FDR)
    
    Example:
        >>> import deferential_expression.edger as edger
        >>> model = edger.glm_ql_fit(rse, design)
        >>> # Get raw test result
        >>> r, pkg = edger.utils._prep_edger()
        >>> lrt = pkg.glmQLFTest(model.fit, coef=2)
        >>> top_genes = edger.top_tags(lrt, n=100)
    """
    r, pkg = _prep_edger()
    
    if n is None:
        n_genes = int(r.r2py(r.ro.baseenv["nrow"](lrt_obj)))
        n = n_genes
    
    top_r = pkg.topTags(
        lrt_obj,
        n=n,
        **{"adjust.method": adjust_method, "sort.by": sort_by},
        **kwargs
    )
    
    table_r = r.ro.baseenv["$"](top_r, "table")
    
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        df = r.r2py(table_r)
    
    df = df.reset_index(names="gene")
    df = df.rename(columns={
        "PValue": "p_value",
        "FDR": "adj_p_value",
        "logFC": "log_fc",
        "logCPM": "log_cpm",
        "LR": "lr_statistic",
        "F": "f_statistic",
    })
    
    return df
