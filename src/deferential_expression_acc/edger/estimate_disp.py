"""Estimate dispersion parameters for edgeR analysis."""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from bioc2ri.lazy_r_env import get_r_environment
from .utils import _prep_edger, pandas_to_r_matrix
from .EdgeR import EdgeR


def estimate_disp(
    obj: EdgeR,
    design: pd.DataFrame,
    trend: str = "loess",
    tagwise: bool = True,
    prior_df: float = 20.0,
    robust: bool = False,
    **kwargs: Any
) -> EdgeR:
    """Estimate dispersion parameters for edgeR GLM analysis.

    Wraps the R ``edgeR::estimateDisp`` function to compute common, trended, and
    tagwise dispersions. Dispersion estimates are crucial for accurate differential
    expression testing in count data.

    Args:
        obj: ``EdgeR`` instance with a counts assay (typically after normalization).
        design: Design matrix (samples × covariates) as a pandas DataFrame.
        trend: Type of trend fitting. Options: ``"none"``, ``"loess"``,
            ``"locfit"``, ``"movingave"``, ``"locfit.mixed"``. Default: ``"loess"``.
        tagwise: If ``True``, computes tagwise dispersions in addition to common
            and trended. Default: ``True``.
        prior_df: Prior degrees of freedom for empirical Bayes shrinkage.
            Default: 20.0.
        robust: If ``True``, uses robust estimation. Default: ``False``.
        **kwargs: Additional keyword arguments forwarded to ``edgeR::estimateDisp``.

    Returns:
        EdgeR: New instance with the ``disp`` slot containing the dispersion object.

    Notes:
        - Common dispersion represents the average dispersion across all genes.
        - Trended dispersion models intensity-dependent variation.
        - Tagwise dispersion uses empirical Bayes to shrink individual gene
          dispersions towards the common/trended value.

    Examples:
        >>> obj_disp = estimate_disp(obj, design=design_df)
        >>> # Can then use obj_disp for glmQLFit or other downstream analyses
    """
    r = get_r_environment()
    r_pkg = _prep_edger()[1]

    # Get DGEList - create from counts if needed
    if hasattr(obj, 'dge') and obj.dge is not None:
        dge = obj.dge
    else:
        # Create DGEList from counts
        counts_r = obj.assay_r("counts")
        dge = r_pkg.DGEList(counts_r)
    
    # Convert design to R matrix
    design_r = pandas_to_r_matrix(design)

    # Prepare keyword arguments
    call_kwargs: Dict[str, Any] = {
        "trend": trend,
        "tagwise": tagwise,
        "prior.df": prior_df,
        "robust": robust,
    }
    call_kwargs.update(kwargs)

    # Call estimateDisp
    disp_result = r_pkg.estimateDisp(dge, design_r, **call_kwargs)

    # Return new EdgeR with disp set
    return obj._clone(dge=disp_result, disp=disp_result)
