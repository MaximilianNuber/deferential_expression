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
    """
    Apply `limma::contrasts.fit` to an existing `lm_fit` inside a `LimmaModel`.

    Args:
        lm_obj:
            A `LimmaModel` instance with a valid `lm_fit` object.
        contrast:
            1D array-like numeric contrast vector
            (length must equal the number of columns in the design).
            This can come from `formulaic_contrasts`, a design matrix,
            or be constructed manually.

    Returns:
        LimmaModel:
            A *new* `LimmaModel` in which the `contrast_fit` slot contains the
            R object returned by `limma::contrasts.fit`.

    Raises:
        AssertionError:
            - If `lm_obj` is not a `LimmaModel`.
            - If `lm_obj.lm_fit` is missing.
            - If `contrast` is not 1D or not array-like.
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
