from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from bioc2ri.lazy_r_env import get_r_environment
from .lm_fit import LimmaModel
from .utils import _limma


def top_table(
    lm_obj: LimmaModel,
    coef: Optional[Union[int, str]] = None,
    n: Optional[int] = None,
    adjust_method: str = "BH",
    sort_by: str = "PValue",
    **kwargs: Any
) -> pd.DataFrame:
    """Extract top-ranked genes from differential expression analysis.

    Wraps the R ``limma::topTable`` function to extract and rank genes by evidence
    of differential expression. Automatically runs ``eBayes`` if not already computed.

    Args:
        lm_obj: ``LimmaModel`` instance with ``ebayes``, ``contrast_fit``, or ``lm_fit`` set.
        coef: Coefficient to extract. Either a coefficient name (string), 1-based index
            (integer), or ``None`` to use all coefficients. Default: ``None``.
        n: Number of top genes to return. If ``None``, returns all genes. Default: ``None``.
        adjust_method: Multiple testing correction method. Options: ``"BH"`` (Benjamini-Hochberg),
            ``"fdr"``, ``"bonferroni"``, ``"holm"``, ``"none"``. Default: ``"BH"``.
        sort_by: Column to sort by. Options: ``"PValue"``, ``"logFC"``, ``"AveExpr"``,
            ``"none"``. Default: ``"PValue"``.
        **kwargs: Additional keyword arguments forwarded to ``limma::topTable``.

    Returns:
        pd.DataFrame: DataFrame of top-ranked features with standardized column names:
            ``gene`` (index), ``log_fc``, ``p_value``, ``adj_p_value``, and other limma statistics.

    Raises:
        AssertionError: If no fit object (``ebayes``, ``contrast_fit``, or ``lm_fit``) is set.

    Notes:
        - If ``ebayes`` is already computed, uses it directly; otherwise computes it.
        - Column names are standardized: ``P.Value`` → ``p_value``, ``logFC`` → ``log_fc``,
          ``adj.P.Val`` → ``adj_p_value``.

    Examples:
        >>> lm_eb = e_bayes(lm_fit(se, design=design_df))
        >>> top_genes = top_table(lm_eb, n=10)
        >>> print(top_genes[['log_fc', 'p_value', 'adj_p_value']])
    """
    assert isinstance(lm_obj, LimmaModel), "lm_obj must be a LimmaModel instance"

    # Use ebayes if available, otherwise compute it
    if lm_obj.ebayes is not None:
        eb = lm_obj.ebayes
    else:
        r_fit = lm_obj.contrast_fit if lm_obj.contrast_fit is not None else lm_obj.lm_fit
        assert r_fit is not None, "lm_fit or contrast_fit must be set in the LimmaModel instance"
        
        # Import e_bayes here to avoid circular imports
        from .e_bayes import e_bayes
        lm_obj_eb = e_bayes(lm_obj)
        eb = lm_obj_eb.ebayes

    r = get_r_environment()
    limma_pkg = _limma()

    if n is None:
        n = int(r.r2py(r.ro.baseenv["nrow"](eb)))
    
    # call topTable with correct parameter names (use dot notation for R params)
    call_kwargs = {"number": n, "adjust.method": adjust_method}
    
    # Handle coef and sort.by mapping
    # If we have a contrast fit, coef should be 1 (the contrast coefficient)
    if lm_obj.contrast_fit is not None and coef is None:
        coef = 1
    
    if coef is None:
        call_kwargs["coef"] = r.ro.NULL
        # For F-statistic version (when coef=NULL), map sort_by to "F" or "none"
        f_stat_sort_map = {
            "PValue": "F",
            "logFC": "F",
            "AveExpr": "F",
            "B": "F",
            "F": "F",
            "none": "none"
        }
        call_kwargs["sort.by"] = f_stat_sort_map.get(sort_by, "F")
    else:
        call_kwargs["coef"] = coef
        # For single coefficient version, use standard sort options
        sort_by_map = {
            "PValue": "p",
            "logFC": "logFC",
            "AveExpr": "AveExpr",
            "B": "B",
            "F": "B",  # Map F-statistic to B for single coef
            "none": "none"
        }
        call_kwargs["sort.by"] = sort_by_map.get(sort_by, "p")
    
    call_kwargs.update(kwargs)
    
    top_r = limma_pkg.topTable(eb, **call_kwargs)
    
    # convert to pandas DataFrame
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        df = r.get_conversion().rpy2py(top_r)
    
    return df.reset_index(names="gene").rename(
        columns={
            'P.Value': 'p_value',
            'logFC': 'log_fc',
            'adj.P.Val': 'adj_p_value',
            'AveExpr': 'ave_expr',
            't': 't_statistic',
            'B': 'b_statistic'
        }
    )
