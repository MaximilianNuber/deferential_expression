"""
Utility functions for sva module.

Provides helper functions for R environment and batch resolution.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Any, Sequence, Union
import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _prep_sva():
    """Lazily prepare the sva runtime.
    
    Returns:
        Tuple[Any, Any]: (r_env, sva_pkg)
    """
    from bioc2ri.lazy_r_env import get_r_environment
    r = get_r_environment()
    sva_pkg = r.lazy_import_r_packages("sva")
    return r, sva_pkg


def _sva():
    """Get the sva R package."""
    _, pkg = _prep_sva()
    return pkg


def resolve_batch(
    se: Any,
    batch: Union[str, Sequence, np.ndarray, pd.Series]
):
    """Resolve batch specification to R character vector.
    
    Works with any SummarizedExperiment variant (SE, RSE, SCE).
    
    Args:
        se: SummarizedExperiment for column_data lookup.
        batch: Batch factor - either column name in column_data (str) or array-like.
    
    Returns:
        R StrVector with batch labels.
    
    Raises:
        KeyError: If batch is a string but column not found in column_data.
    """
    from rpy2.robjects.vectors import StrVector
    
    if isinstance(batch, str):
        # Treat as column name in column_data
        cd = se.get_column_data()
        if cd is None or batch not in cd.column_names:
            raise KeyError(f"Batch column '{batch}' not found in column_data.")
        arr = np.asarray(cd[batch])
    else:
        arr = np.asarray(batch)
    
    # Convert to strings for R factor
    vals = [str(v) for v in arr.tolist()]
    return StrVector(vals)
