from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bioc2ri.lazy_r_env import get_r_environment
from ..resummarizedexperiment import RESummarizedExperiment
from .lm_fit import LimmaModel
from .utils import _limma
from ..edger.utils import pandas_to_r_matrix
from dataclasses import replace


def treat(
    lm_obj: Union[LimmaModel, RESummarizedExperiment],
    design: Optional[pd.DataFrame] = None,
    lfc: float = 1.0,
    robust: bool = False,
    trend: bool = False,
    winsor_tail_p: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> LimmaModel:
    """Test for differential expression relative to a fold-change threshold.

    Wraps the R ``limma::treat`` function, which tests whether log-fold-changes are
    significantly greater than a threshold (in absolute value), rather than simply
    testing whether they differ from zero.

    Can be called either on a fitted LimmaModel or on a RESummarizedExperiment.
    If given a RESummarizedExperiment, will first apply voom transformation and lmFit.

    Args:
        lm_obj: Either a ``LimmaModel`` instance or a ``RESummarizedExperiment`` with
            ``"log_expr"`` assay.
        design: Design matrix (only required if lm_obj is RESummarizedExperiment).
        lfc: Log-fold-change threshold for testing. Tests |logFC| > lfc.
            Default: 1.0 (2-fold change).
        robust: If ``True``, uses robust empirical Bayes. Default: ``False``.
        trend: If ``True``, fits a mean-variance trend. Default: ``False``.
        winsor_tail_p: Optional tuple (lower, upper) tail probabilities for Winsorizing
            when ``robust=True``.
        **kwargs: Additional keyword arguments forwarded to ``limma::treat``.

    Returns:
        LimmaModel: Instance with the ``lm_fit`` slot containing the TREAT fit object.

    Notes:
        - TREAT provides better ranking and p-values when you care about effect size.
        - The lfc threshold is applied symmetrically (tests |logFC| > lfc).
        - Use with ``top_table()`` to extract ranked results.

    Examples:
        >>> lm_treat = treat(lm, lfc=1.0)  # Apply TREAT to fitted model
        >>> results = top_table(lm_treat, n=100)
    """
    r = get_r_environment()
    limma_pkg = _limma()
    
    # Handle both RESummarizedExperiment and LimmaModel inputs
    if isinstance(lm_obj, RESummarizedExperiment):
        # Need design for SummarizedExperiment input
        assert design is not None, "design must be provided when lm_obj is RESummarizedExperiment"
        
        # Apply voom and lmFit first
        from .voom import voom
        from .lm_fit import lm_fit
        
        se_voom = voom(lm_obj, design=design)
        lm_obj = lm_fit(se_voom, design=design)
    
    assert isinstance(lm_obj, LimmaModel), "lm_obj must be a LimmaModel instance"
    
    # Get the fit object (prefer contrast_fit if available)
    r_fit = lm_obj.contrast_fit if lm_obj.contrast_fit is not None else lm_obj.lm_fit
    assert r_fit is not None, "lm_fit or contrast_fit must be set in the LimmaModel instance"
    
    # Prepare optional arguments
    call_kwargs: Dict[str, Any] = {"lfc": lfc, "robust": robust, "trend": trend}
    
    if winsor_tail_p is not None:
        call_kwargs["winsor.tail.p"] = r.FloatVector(winsor_tail_p)
    
    call_kwargs.update(kwargs)
    
    # Apply treat to the fitted model
    treat_fit = limma_pkg.treat(r_fit, **call_kwargs)
    
    # Return new LimmaModel with treat result
    return replace(lm_obj, lm_fit=treat_fit, method="treat")
