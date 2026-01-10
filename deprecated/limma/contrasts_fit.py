from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Literal

import numpy as np
import pandas as pd

from bioc2ri.lazy_r_env import get_r_environment                  

from ..resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter
from ..edger.utils import pandas_to_r_matrix             
from .utils import _limma                                
from.lm_fit import LimmaModel               

def contrasts_fit(
    lm_obj: LimmaModel,
    contrast: Sequence[int | float],
) -> LimmaModel:
    """Apply contrast to fitted linear model.

    Wraps the R ``limma::contrasts.fit`` function to compute coefficients and standard
    errors for a specified contrast. Returns a new ``LimmaModel`` with the contrast
    fit stored.

    Args:
        lm_obj: ``LimmaModel`` instance with a valid ``lm_fit`` object from ``lm_fit()``.
        contrast: 1D numeric contrast vector (length = number of design columns).
            Can be constructed manually, from a contrast matrix, or using formulaic
            contrast syntax. Each element specifies the coefficient for the corresponding
            design column.

    Returns:
        LimmaModel: New instance with the ``contrast_fit`` slot containing the R object
            returned by ``limma::contrasts.fit``.

    Raises:
        AssertionError: If ``lm_obj`` is not a ``LimmaModel``, if ``lm_obj.lm_fit``
            is missing, or if ``contrast`` is not 1D array-like.

    Notes:
        - Original ``LimmaModel`` remains unchanged (functional/immutable style).
        - After fitting contrasts, use ``.e_bayes()`` to compute moderated statistics.

    Examples:
        >>> lm = lm_fit(se, design=design_df)
        >>> contrast_vec = [0, 1, -1]  # Compare condition 2 vs condition 3
        >>> lm_contrast = contrasts_fit(lm, contrast=contrast_vec)
        >>> lm_ebayes = lm_contrast.e_bayes()
    """
    # --- Validation ---------------------------------------------------------
    assert isinstance(lm_obj, LimmaModel), "lm_obj must be a LimmaModel instance"
    assert lm_obj.lm_fit is not None, "lm_fit must be set before running contrasts.fit"
    assert isinstance(contrast, (list, np.ndarray, pd.Series)), (
        "contrast must be a list, numpy array, or pandas Series"
    )

    # --- Prepare environment and package -----------------------------------
    _r = get_r_environment()
    limma_pkg = _limma()

    # --- Convert contrast vector to R --------------------------------------
    contrast = np.asarray(contrast, dtype=float)
    assert contrast.ndim == 1, "contrast must be a 1D array-like"

    contrast_r = _r.ro.FloatVector(contrast)

    # --- Apply contrasts.fit -----------------------------------------------
    fit_r = limma_pkg.contrasts_fit(lm_obj.lm_fit, contrast=contrast_r)

    # --- Return new LimmaModel with updated slot ---------------------------
    return replace(
        lm_obj,
        contrast_fit=fit_r,
    )
