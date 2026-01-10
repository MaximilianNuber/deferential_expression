"""
Apply ComBat-seq batch correction using sva::ComBat_seq.

This module provides a functional interface for batch correction
of count data.
"""

from __future__ import annotations
from typing import Optional, Sequence, TypeVar, Union
import numpy as np
import pandas as pd

from .utils import _prep_sva, resolve_batch
from .checks import check_se, check_assay_exists, check_r_assay
from ..edger.utils import pandas_to_r_matrix

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


def combat_seq(
    se: SE,
    batch: Union[str, Sequence, np.ndarray, pd.Series],
    assay: str = "counts",
    output_assay: Optional[str] = None,
    group: Optional[Union[str, Sequence, np.ndarray, pd.Series]] = None,
    covar_mod: Optional[pd.DataFrame] = None,
    full_mod: bool = True,
    shrink: bool = False,
    shrink_disp: bool = False,
    gene_subset_n: Optional[int] = None,
    in_place: bool = False,
    **kwargs
) -> SE:
    """
    Apply ComBat-seq batch correction to count data.
    
    Wraps `sva::ComBat_seq`. Takes a count matrix and returns a
    batch-corrected count matrix of the same dimensions.
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized count assay.
        batch: Batch factor. Either column name in column_data (str) or
            array-like of batch labels.
        assay: Input count assay name. Default: "counts".
        output_assay: Output assay name. Defaults to "{assay}_combat_seq".
        group: Optional biological group factor to preserve.
        covar_mod: Optional covariate model matrix.
        full_mod: Include group in model matrix. Default: True.
        shrink: Apply shrinkage. Default: False.
        shrink_disp: Shrink dispersion estimates. Default: False.
        gene_subset_n: Number of genes for subset. Default: None.
        in_place: If True, modify se in place. Default: False.
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        SummarizedExperiment with batch-corrected count assay.
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.sva as sva
        >>> se = initialize_r(se, assay="counts")
        >>> se = sva.combat_seq(se, batch="batch_id", group="condition")
    """
    from ..r_init import get_rmat
    from ..rmatrixadapter import RMatrixAdapter
    
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    
    r, sva_pkg = _prep_sva()
    
    # Get input matrix
    counts_r = get_rmat(se, assay)
    
    # Resolve batch
    batch_r = resolve_batch(se, batch)
    
    # Resolve group if provided
    if group is not None:
        group_r = resolve_batch(se, group)
    else:
        group_r = r.ro.NULL
    
    # Handle covar_mod
    if covar_mod is not None:
        covar_mod_r = pandas_to_r_matrix(covar_mod)
    else:
        covar_mod_r = r.ro.NULL
    
    # Handle gene_subset_n
    gene_subset_n_r = r.ro.NULL if gene_subset_n is None else gene_subset_n
    
    # Call ComBat_seq
    result_r = sva_pkg.ComBat_seq(
        counts_r,
        batch=batch_r,
        group=group_r,
        covar_mod=covar_mod_r,
        full_mod=full_mod,
        shrink=shrink,
        shrink_disp=shrink_disp,
        gene_subset_n=gene_subset_n_r,
        **kwargs
    )
    
    # Store result using _define_output for class preservation
    out_name = output_assay if output_assay else f"{assay}_combat_seq"
    output = se._define_output(in_place=in_place)
    new_assays = dict(output.assays)
    new_assays[out_name] = RMatrixAdapter(result_r, r)
    output._assays = new_assays
    
    return output
