"""
Perform Surrogate Variable Analysis using sva::sva.

This module provides a functional interface for SVA.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd

from .utils import _prep_sva
from .checks import check_rse, check_assay_exists, check_mod
from ..edger.utils import pandas_to_r_matrix

if TYPE_CHECKING:
    from ..resummarizedexperiment import RESummarizedExperiment


def sva(
    rse: "RESummarizedExperiment",
    mod: pd.DataFrame,
    assay: str = "cpm",
    mod0: Optional[pd.DataFrame] = None,
    n_sv: Optional[int] = None,
    method: str = "irw",
    vfilter: Optional[int] = None,
    B: int = 5,
    numSVmethod: str = "be",
    **kwargs
) -> "RESummarizedExperiment":
    """
    Perform Surrogate Variable Analysis.
    
    Wraps `sva::sva`. Identifies and estimates surrogate variables representing
    unknown batch effects or other unwanted variation.
    
    Results are stored in the returned RESummarizedExperiment:
    - Surrogate variable matrix → metadata["sva$sv"] (numpy array)
    - pprob.gam (posterior prob of association) → row_data["sva$pprob.gam"]
    - pprob.b (posterior prob for each sv) → row_data["sva$pprob.b"]
    - n.sv (number of surrogate variables) → metadata["sva$n.sv"]
    
    Use `get_sv()` to extract the surrogate variables as a DataFrame.
    
    Args:
        rse: Input RESummarizedExperiment with expression data.
        mod: Full model matrix (samples × covariates) including biological variables.
        assay: Expression assay name (should be continuous). Default: "cpm".
        mod0: Null model matrix (intercept only by default).
        n_sv: Number of surrogate variables. If None, estimated automatically.
        method: SVA method: "irw" (default) or "two-step".
        vfilter: Number of most variable genes to use. Default: None (all genes).
        B: Number of bootstrap iterations for "irw". Default: 5.
        numSVmethod: Method for estimating n.sv: "be" or "leek". Default: "be".
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        New RESummarizedExperiment with SVA results in metadata and row_data.
    
    Example:
        >>> import deferential_expression.sva as sva
        >>> import pandas as pd
        >>> design = pd.DataFrame({'Intercept': [1]*6, 'Cond': [0,0,0,1,1,1]})
        >>> rse_sva = sva.sva(rse, mod=design, assay="log_expr")
        >>> sv_df = sva.get_sv(rse_sva)  # Get surrogate variables as DataFrame
    """
    check_rse(rse)
    check_assay_exists(rse, assay)
    n_samples = len(rse.column_names) if rse.column_names else rse.shape[1]
    check_mod(mod, n_samples)
    
    r, sva_pkg = _prep_sva()
    
    # Get input matrix
    dat_r = rse.assay_r(assay)
    
    # Convert mod to R matrix
    mod_r = pandas_to_r_matrix(mod)
    
    # Handle mod0 - default to intercept only
    if mod0 is not None:
        mod0_r = pandas_to_r_matrix(mod0)
    else:
        # Create intercept-only matrix from first column of mod
        mod0_df = mod.iloc[:, [0]]
        mod0_r = pandas_to_r_matrix(mod0_df)
    
    # Handle optional parameters
    n_sv_r = r.ro.NULL if n_sv is None else n_sv
    vfilter_r = r.ro.NULL if vfilter is None else vfilter
    
    # Call sva
    sva_result = sva_pkg.sva(
        dat_r,
        mod=mod_r,
        mod0=mod0_r,
        n_sv=n_sv_r,
        method=method,
        vfilter=vfilter_r,
        B=B,
        numSVmethod=numSVmethod,
        **kwargs
    )
    
    # Extract results from R list
    # n.sv: number of surrogate variables (extract first to handle edge case)
    n_sv_result = int(r.ro.baseenv["$"](sva_result, "n.sv")[0])
    
    # sv: matrix of surrogate variables (n_samples x n_sv)
    sv_r = r.ro.baseenv["$"](sva_result, "sv")
    sv_np = np.asarray(sv_r)
    # Ensure 2D even if empty
    if sv_np.ndim == 0 or (sv_np.ndim == 1 and n_sv_result == 0):
        sv_np = np.empty((len(rse.column_names or []), 0))
    elif sv_np.ndim == 1:
        sv_np = sv_np.reshape(-1, 1)
    
    # pprob.gam: posterior probabilities for each gene
    pprob_gam_r = r.ro.baseenv["$"](sva_result, "pprob.gam")
    pprob_gam = np.asarray(pprob_gam_r) if pprob_gam_r is not r.ro.NULL else np.zeros(len(rse.row_names or []))
    
    # pprob.b: posterior probabilities for each gene for each SV
    pprob_b_r = r.ro.baseenv["$"](sva_result, "pprob.b")
    pprob_b = np.asarray(pprob_b_r) if pprob_b_r is not r.ro.NULL else np.array([])
    
    # Update metadata
    new_metadata = dict(rse.metadata)
    new_metadata["sva$sv"] = sv_np
    new_metadata["sva$n.sv"] = n_sv_result
    
    # Update row_data with pprob vectors
    row_data = rse.row_data_df
    if row_data is None:
        row_data = pd.DataFrame(index=rse.row_names)
    else:
        row_data = row_data.copy()
    
    # Add pprob.gam if available
    if pprob_gam.size > 0:
        row_data["sva$pprob.gam"] = pprob_gam
    
    # pprob.b might be a matrix if multiple SVs, or empty if n.sv=0
    if pprob_b.size > 0:
        if pprob_b.ndim == 1:
            row_data["sva$pprob.b"] = pprob_b
        elif pprob_b.ndim == 2 and pprob_b.shape[1] > 0:
            # Store each column separately for multiple SVs
            for i in range(pprob_b.shape[1]):
                row_data[f"sva$pprob.b.{i+1}"] = pprob_b[:, i]
    
    # Return new SE with updated metadata and row_data
    from ..resummarizedexperiment import RESummarizedExperiment
    return RESummarizedExperiment(
        assays=dict(rse.assays),
        row_data=row_data,
        column_data=rse.column_data_df,
        row_names=rse.row_names,
        column_names=rse.column_names,
        metadata=new_metadata,
    )
