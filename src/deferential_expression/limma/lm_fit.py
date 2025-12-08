from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Literal

import numpy as np
import pandas as pd

from bioc2ri.lazy_r_env import get_r_environment
from bioc2ri import numpy_plugin
from ..resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter
from ..edger.utils import numpy_to_r_matrix, pandas_to_r_matrix
from .utils import _limma


# --- Limma model container -------------------------------------------------


@dataclass
class LimmaModel:
    """
    Container for limma results including fit, coefficients, and metadata.

    This class encapsulates the results of fitting a linear model using limma,
    storing the R objects produced by `lmFit`, `contrasts.fit`, and `eBayes`.
    It also holds sample and feature names, the design matrix, fitting method,
    and extracted coefficients as a pandas DataFrame.

    Args:
        sample_names:
            Optional sequence of sample names (column names).
        feature_names:
            Optional sequence of feature names (row names).
        lm_fit:
            Optional R object from `limma::lmFit`.
        contrast_fit:
            Optional R object from `limma::contrasts.fit`.
        ebayes:
            Optional R object from `limma::eBayes`.
        design:
            Optional design matrix (samples × covariates) as a pandas DataFrame.
        ndups:
            Optional number of technical replicates (if applicable).
        method:
            Fitting method used, e.g., `"ls"` (least squares) or `"robust"`.
        coefficients:
            Optional pandas DataFrame of extracted coefficients.
        metadata:
            Optional free-form dictionary for additional metadata.
    """

    # Names
    sample_names: Optional[Sequence[str]] = None
    feature_names: Optional[Sequence[str]] = None

    # Core R objects
    lm_fit: Optional[Any] = None
    contrast_fit: Optional[Any] = None
    ebayes: Optional[Any] = None

    # Design / fitting info
    design: Optional[pd.DataFrame] = None
    ndups: Optional[int] = None
    method: Literal["ls", "robust"] = "ls"

    # Extracted results / metadata
    coefficients: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_sample_names(self) -> Sequence[str]:
        """Get sample names from `sample_names` or the `lm_fit` object."""
        if self.sample_names is not None:
            return self.sample_names

        if self.lm_fit is not None:
            r = get_r_environment()
            return list(r.ro.baseenv["colnames"](self.lm_fit))

        return ()

    def get_feature_names(self) -> Sequence[str]:
        """Get feature names from `feature_names` or the `lm_fit` object."""
        if self.feature_names is not None:
            return self.feature_names

        if self.lm_fit is not None:
            r = get_r_environment()
            return list(r.ro.baseenv["rownames"](self.lm_fit))

        return ()

    def get_lmfit_names(self) -> Sequence[str]:
        """Get the slot names of the `lm_fit` object for access and conversion."""
        r = get_r_environment()
        return list(r.ro.baseenv["names"](self.lm_fit))

    def get_coefficients(self) -> pd.DataFrame:
        """Extract coefficients as a pandas DataFrame from the limma `lm_fit` object."""
        assert self.lm_fit is not None, "lm_fit must be set to extract coefficients"
        # _r = get_r_environment()
        coefs_r = self.lm_fit.rx2("coefficients")
        np_eng = numpy_plugin()
        coefs = np_eng.r2py(coefs_r)
        return coefs

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #

    def e_bayes(self) -> "LimmaModel":
        """Run `eBayes` on the `lm_fit` object and store the result."""
        assert self.lm_fit is not None, "lm_fit must be set to run eBayes"
        _r = get_r_environment()
        limma_pkg = _limma()
        eb = limma_pkg.eBayes(self.lm_fit)
        return replace(self, ebayes=eb)


# --- lmFit wrapper ---------------------------------------------------------


def lm_fit(
    se: RESummarizedExperiment,
    design: pd.DataFrame,
    ndups: Optional[int] = None,
    method: Literal["ls", "robust"] = "ls",
    return_result_object: bool = False,
    **kwargs: Any,
) -> LimmaModel:
    """
    Run `limma::lmFit` on an expression assay and store the fit in a `LimmaModel`.

    Args:
        se:
            Input `RESummarizedExperiment` containing a `"log_expr"` assay
            (by convention) and optionally a `"weights"` assay.
        design:
            Design matrix (samples × covariates) as a pandas DataFrame.
        ndups:
            Number of technical replicates per unique sample (or `None`).
        method:
            Fitting method, e.g., `"ls"` (least squares) or `"robust"`.
        return_result_object:
            Currently unused flag; reserved for future API variants.
        **kwargs:
            Additional keyword arguments forwarded to `limma::lmFit`.

    Returns:
        LimmaModel:
            An instance of `LimmaModel` with the `lm_fit` R object set.
    """
    _r = get_r_environment()
    limma_pkg = _limma()

    lmres = LimmaModel(
        method=method,
        sample_names=se.column_names,
        feature_names=se.row_names,
    )

    exprs_r = se.assay_r("log_expr")
    design_r = pandas_to_r_matrix(design)
    lmres.design = design

    if "weights" in se.assay_names:
        weights = se.assay_r("weights")
    else:
        weights = _r.ro.NULL

    if ndups is None:
        ndups_r = _r.ro.NULL
    else:
        assert isinstance(ndups, int), "ndups must be an integer or None"
        ndups_r = ndups

    fit = limma_pkg.lmFit(
        exprs_r,
        design_r,
        weights=weights,
        method=method,
        **kwargs,
    )
    lmres.lm_fit = fit

    return lmres
