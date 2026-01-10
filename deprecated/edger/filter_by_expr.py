from deferential_expression.edger.utils import _prep_edger
from deferential_expression.edger.utils import numpy_to_r_matrix
import numpy as np
from typing import Sequence
import pandas as pd


def filter_by_expr(obj: "RESummarizedExperiment", group: Sequence[str] | None = None, design: pd.DataFrame | None = None, 
                   assay = "counts", **kwargs):
    """Functional wrapper for ``edgeR::filterByExpr`` on an ``EdgeR`` container.

    Args:
        obj: ``EdgeR`` instance holding at least the counts assay.
        group: Optional group labels (factor) for filtering.
        design: Optional design matrix as pandas DataFrame.
        assay: Assay name containing counts.
        **kwargs: Additional args forwarded to ``filterByExpr``.

    Returns:
        np.ndarray: Boolean mask indicating rows to retain.
    """
    r, pkg = _prep_edger()

    if group is not None:
        group = r.StrVector(group)
    else:
        group = r.ro.NULL
        
    if design is not None:
        _mat = design.values
        design = numpy_to_r_matrix(_mat, rownames = design.index.to_list(), colnames = design.columns.to_list())
    else:
        design = r.ro.NULL

    # mask_r = pkg.filterByExpr(obj.assay_r(assay), group = group, design = design, **kwargs)
    mask_r = _filter_by_expr_impl(
        obj.assay_r(assay), 
        group = group, 
        design = design, 
        **kwargs
    )
    mask = np.asarray(mask_r)
    return mask.astype(bool)

def _filter_by_expr_impl(
    rmat, 
    group = None, 
    design = None, 
    lib_size = None,
    min_count = 10,
    min_total_count = 15,
    large_n = 10, 
    min_prop = 0.7,
    **kwargs
):
    r, pkg = _prep_edger()
    
    group = r.ro.NULL if group is None else group
    design = r.ro.NULL if design is None else design
    lib_size = r.ro.NULL if lib_size is None else lib_size

    return pkg.filterByExpr(
        rmat, 
        group = group, 
        design = design, 
        lib_size = lib_size,
        min_count = min_count,
        min_total_count = min_total_count,
        large_n = large_n,
        min_prop = min_prop,
        **kwargs
        )