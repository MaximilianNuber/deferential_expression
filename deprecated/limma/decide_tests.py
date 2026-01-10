from typing import Any, Optional

import pandas as pd

from bioc2ri.lazy_r_env import get_r_environment
from .lm_fit import LimmaModel
from .utils import _limma


def decide_tests(
    lm_obj: LimmaModel,
    method: str = "separate",
    adjust_method: str = "BH",
    p_value: float = 0.05,
    lfc: float = 0,
    **kwargs: Any
) -> pd.DataFrame:
    """Classify genes as significantly up, down, or not significant.

    Wraps the R ``limma::decideTests`` function to perform multiple testing across
    genes and contrasts, returning classification codes for each gene.

    Args:
        lm_obj: ``LimmaModel`` instance with ``ebayes``, ``contrast_fit``, or ``lm_fit`` set.
        method: Testing method. Options:
            - ``"separate"``: Test each contrast separately
            - ``"global"``: Global F-test across all contrasts
            - ``"hierarchical"``: Hierarchical testing
            - ``"nestedF"``: Nested F-tests
            Default: ``"separate"``.
        adjust_method: Multiple testing correction method. Options: ``"BH"``, ``"fdr"``,
            ``"bonferroni"``, ``"holm"``, ``"none"``. Default: ``"BH"``.
        p_value: Significance threshold for adjusted p-values. Default: 0.05.
        lfc: Log-fold-change threshold. Genes must have |logFC| > lfc to be considered
            significant. Default: 0 (no threshold).
        **kwargs: Additional keyword arguments forwarded to ``limma::decideTests``.

    Returns:
        pd.DataFrame: DataFrame with genes as rows and contrasts as columns.
            Values: -1 (down-regulated), 0 (not significant), 1 (up-regulated).

    Raises:
        AssertionError: If no fit object is set in ``lm_obj``.

    Notes:
        - If ``ebayes`` is not computed, it will be computed automatically.
        - The returned matrix is useful for Venn diagrams and summary statistics.

    Examples:
        >>> lm_eb = e_bayes(lm_fit(se, design=design_df))
        >>> results = decide_tests(lm_eb, p_value=0.01, lfc=1)
        >>> print((results != 0).sum())  # Count significant genes per contrast
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

    # Call decideTests with correct R parameter names
    call_kwargs = {
        "method": method,
        "adjust.method": adjust_method,
        "p.value": p_value,
        "lfc": lfc
    }
    call_kwargs.update(kwargs)
    
    decide_r = limma_pkg.decideTests(eb, **call_kwargs)

    # Convert to pandas DataFrame
    # Extract rownames and colnames from R object
    r_rownames = r.ro.baseenv["rownames"](decide_r)
    r_colnames = r.ro.baseenv["colnames"](decide_r)
    
    rownames = list(r.r2py(r_rownames)) if r_rownames is not r.ro.NULL else None
    colnames = list(r.r2py(r_colnames)) if r_colnames is not r.ro.NULL else None
    
    # Convert matrix to numpy array
    arr = r.ro.baseenv["as.matrix"](decide_r)
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        matrix_arr = r.r2py(arr)
    
    # Create DataFrame with proper index and columns
    df = pd.DataFrame(
        matrix_arr,
        index=rownames,
        columns=colnames
    )
    
    return df
