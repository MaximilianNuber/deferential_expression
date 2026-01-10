"""
Extract surrogate variables from RESummarizedExperiment.

This module provides a functional interface to retrieve SVA results.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas as pd

from .checks import check_rse

if TYPE_CHECKING:
    from ..resummarizedexperiment import RESummarizedExperiment


def get_sv(
    rse: "RESummarizedExperiment",
    key: str = "sva$sv",
    as_pandas: bool = True
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Extract surrogate variables from metadata.
    
    Args:
        rse: RESummarizedExperiment with SVA results.
        key: Metadata key where SV matrix is stored. Default: "sva$sv".
        as_pandas: If True, return as DataFrame with column_names as index
            and SV1, SV2, ... as columns. Default: True.
    
    Returns:
        Surrogate variable matrix as numpy array or pandas DataFrame.
        If no SVs were found (n.sv=0), returns empty array/DataFrame.
    
    Raises:
        TypeError: If rse is not an RESummarizedExperiment.
        KeyError: If the key is not found in metadata.
    
    Example:
        >>> import deferential_expression.sva as sva
        >>> rse_sva = sva.sva(rse, mod=design)
        >>> sv_df = sva.get_sv(rse_sva)  # DataFrame with SVs
        >>> sv_np = sva.get_sv(rse_sva, as_pandas=False)  # numpy array
    """
    check_rse(rse)
    
    if key not in rse.metadata:
        raise KeyError(f"Key '{key}' not found in metadata. Run sva() first.")
    
    sv_np = rse.metadata[key]
    
    if not as_pandas:
        return sv_np
    
    # Handle empty case (n.sv=0)
    if sv_np.size == 0 or (sv_np.ndim == 2 and sv_np.shape[1] == 0):
        index = rse.column_names if rse.column_names else list(range(sv_np.shape[0] if sv_np.ndim >= 1 else 0))
        return pd.DataFrame(index=index)
    
    # Create DataFrame with proper index and column names
    n_sv = sv_np.shape[1] if sv_np.ndim > 1 else 1
    columns = [f"SV{i+1}" for i in range(n_sv)]
    
    index = rse.column_names if rse.column_names else list(range(sv_np.shape[0]))
    
    return pd.DataFrame(sv_np, index=index, columns=columns)
