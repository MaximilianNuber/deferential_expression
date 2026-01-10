"""Extract top differential expression results from edgeR GLM test."""

from typing import Any, Optional, Union
import pandas as pd

from bioc2ri.lazy_r_env import get_r_environment
from .utils import _prep_edger


def top_tags(
    lrt_obj: Any,
    n: Optional[int] = None,
    adjust_method: str = "BH",
    sort_by: str = "PValue",
    **kwargs: Any
) -> pd.DataFrame:
    """Extract top-ranked genes from edgeR GLM test results.

    Wraps the R ``edgeR::topTags`` function to extract and rank genes by evidence
    of differential expression from a GLM likelihood ratio test.

    Args:
        lrt_obj: R object from ``edgeR::glmQLFTest`` or similar test result.
        n: Number of top genes to return. If ``None``, returns all genes.
            Default: ``None``.
        adjust_method: Multiple testing correction method. Options: ``"BH"``
            (Benjamini-Hochberg), ``"fdr"``, ``"bonferroni"``, ``"holm"``,
            ``"none"``. Default: ``"BH"``.
        sort_by: Column to sort by. Options: ``"PValue"``, ``"logFC"``,
            ``"logCPM"``, ``"LR"``, ``"none"``. Default: ``"PValue"``.
        **kwargs: Additional keyword arguments forwarded to ``edgeR::topTags``.

    Returns:
        pd.DataFrame: DataFrame of top-ranked features with columns for log-fold-change,
            log-CPM, test statistic, p-value, and adjusted p-value.

    Raises:
        AssertionError: If n is not a positive integer when provided.

    Notes:
        - LRT (Likelihood Ratio Test) is recommended for multi-degree-of-freedom tests.
        - QLF (Quasi-Likelihood F) test is recommended for single-coefficient tests
          and is more conservative.
        - Column names are standardized for consistency with other packages.

    Examples:
        >>> results = top_tags(lrt_result, n=100)
        >>> print(results[['logFC', 'PValue', 'FDR']])
    """
    r = get_r_environment()
    r_pkg = _prep_edger()[1]

    if n is None:
        # Get number of genes from the result object
        n_genes = int(r.r2py(r.ro.baseenv["nrow"](lrt_obj)))
        n = n_genes

    # Map sort_by parameter - edgeR topTags accepts specific values
    sort_by_map = {
        "PValue": "PValue",
        "logFC": "logFC",
        "logCPM": "logCPM",
        "LR": "LR",
        "none": "none"
    }
    sort_by_val = sort_by_map.get(sort_by, "PValue")

    # Prepare keyword arguments with correct R parameter names
    call_kwargs = {
        "n": n,
        "adjust.method": adjust_method,
        "sort.by": sort_by_val,
    }
    call_kwargs.update(kwargs)

    # Call topTags
    top_r = r_pkg.topTags(lrt_obj, **call_kwargs)

    # Extract the table slot
    table_r = r.ro.baseenv["$"](top_r, "table")

    # Convert to pandas DataFrame
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        df = r.r2py(table_r)

    # Reset index to make gene names a column
    df = df.reset_index(names="gene")

    # Standardize column names
    df = df.rename(
        columns={
            "PValue": "p_value",
            "FDR": "adj_p_value",
            "logFC": "log_fc",
            "logCPM": "log_cpm",
            "LR": "lr_statistic",
        }
    )

    return df
