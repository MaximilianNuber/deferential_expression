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
    # Post-processing methods
    # ------------------------------------------------------------------ #

    def contrasts_fit(
        self,
        contrast: Sequence[Union[int, float]],
    ) -> "LimmaModel":
        """
        Apply contrast to fitted linear model.
        
        Args:
            contrast: 1D contrast vector (length = number of design columns).
        
        Returns:
            LimmaModel: New instance with contrast_fit slot set.
        
        Example:
            >>> model_contrast = model.contrasts_fit([0, 1, -1])
        """
        assert self.lm_fit is not None, "lm_fit must be set before running contrasts.fit"
        
        r = get_r_environment()
        limma_pkg = _limma()
        
        contrast_arr = np.asarray(contrast, dtype=float)
        contrast_r = r.ro.FloatVector(contrast_arr)
        
        fit_r = limma_pkg.contrasts_fit(self.lm_fit, contrast=contrast_r)
        
        return replace(self, contrast_fit=fit_r)

    def e_bayes(
        self,
        proportion: float = 0.01,
        trend: bool = False,
        robust: bool = False,
        **kwargs: Any
    ) -> "LimmaModel":
        """
        Compute empirical Bayes moderated statistics.
        
        Args:
            proportion: Assumed proportion of DE genes. Default: 0.01.
            trend: Fit mean-variance trend. Default: False.
            robust: Use robust empirical Bayes. Default: False.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            LimmaModel: New instance with ebayes slot set.
        
        Example:
            >>> model_eb = model.e_bayes(robust=True)
        """
        r_fit = self.contrast_fit if self.contrast_fit is not None else self.lm_fit
        assert r_fit is not None, "lm_fit or contrast_fit must be set"
        
        r = get_r_environment()
        limma_pkg = _limma()
        
        call_kwargs: Dict[str, Any] = {"proportion": proportion, "trend": trend, "robust": robust}
        call_kwargs.update(kwargs)
        
        eb = limma_pkg.eBayes(r_fit, **call_kwargs)
        
        return replace(self, ebayes=eb)

    def top_table(
        self,
        coef: Optional[Union[int, str]] = None,
        n: Optional[int] = None,
        adjust_method: str = "BH",
        sort_by: str = "PValue",
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Extract top-ranked genes from differential expression analysis.
        
        Runs eBayes if not already computed.
        
        Args:
            coef: Coefficient to extract (name or 1-based index).
            n: Number of top genes (None = all).
            adjust_method: Multiple testing method ("BH", "bonferroni", etc.).
            sort_by: Sort column ("PValue", "logFC", etc.).
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            pd.DataFrame: Results with standardized column names.
        
        Example:
            >>> results = model.e_bayes().top_table(n=100)
        """
        # Ensure eBayes is computed
        if self.ebayes is not None:
            eb = self.ebayes
        else:
            model_eb = self.e_bayes()
            eb = model_eb.ebayes
        
        r = get_r_environment()
        limma_pkg = _limma()
        
        if n is None:
            n = int(r.r2py(r.ro.baseenv["nrow"](eb)))
        
        # Handle coef for contrast fits
        if self.contrast_fit is not None and coef is None:
            coef = 1
        
        # Map sort_by
        if coef is None:
            sort_by_map = {"PValue": "F", "logFC": "F", "AveExpr": "F", "B": "F", "F": "F", "none": "none"}
            sort_by_r = sort_by_map.get(sort_by, "F")
            coef_r = r.ro.NULL
        else:
            sort_by_map = {"PValue": "p", "logFC": "logFC", "AveExpr": "AveExpr", "B": "B", "F": "B", "none": "none"}
            sort_by_r = sort_by_map.get(sort_by, "p")
            coef_r = coef
        
        call_kwargs = {"number": n, "adjust.method": adjust_method, "coef": coef_r, "sort.by": sort_by_r}
        call_kwargs.update(kwargs)
        
        top_r = limma_pkg.topTable(eb, **call_kwargs)
        
        with r.localconverter(r.default_converter + r.pandas2ri.converter):
            df = r.get_conversion().rpy2py(top_r)
        
        return df.reset_index(names="gene").rename(columns={
            'P.Value': 'p_value',
            'logFC': 'log_fc',
            'adj.P.Val': 'adj_p_value',
            'AveExpr': 'ave_expr',
            't': 't_statistic',
            'B': 'b_statistic'
        })

    def decide_tests(
        self,
        method: str = "separate",
        adjust_method: str = "BH",
        p_value: float = 0.05,
        lfc: float = 0,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Classify genes as significantly up, down, or not significant.
        
        Runs eBayes if not already computed.
        
        Args:
            method: Testing method ("separate", "global", etc.).
            adjust_method: Multiple testing method.
            p_value: Significance threshold.
            lfc: Log-fold-change threshold.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            pd.DataFrame: Values -1 (down), 0 (not sig), 1 (up).
        
        Example:
            >>> sig_genes = model.e_bayes().decide_tests(p_value=0.01)
        """
        # Ensure eBayes is computed
        if self.ebayes is not None:
            eb = self.ebayes
        else:
            model_eb = self.e_bayes()
            eb = model_eb.ebayes
        
        r = get_r_environment()
        limma_pkg = _limma()
        
        call_kwargs = {
            "method": method,
            "adjust.method": adjust_method,
            "p.value": p_value,
            "lfc": lfc
        }
        call_kwargs.update(kwargs)
        
        decide_r = limma_pkg.decideTests(eb, **call_kwargs)
        
        r_rownames = r.ro.baseenv["rownames"](decide_r)
        r_colnames = r.ro.baseenv["colnames"](decide_r)
        
        rownames = list(r.r2py(r_rownames)) if r_rownames is not r.ro.NULL else None
        colnames = list(r.r2py(r_colnames)) if r_colnames is not r.ro.NULL else None
        
        arr = r.ro.baseenv["as.matrix"](decide_r)
        with r.localconverter(r.default_converter + r.pandas2ri.converter):
            matrix_arr = r.r2py(arr)
        
        return pd.DataFrame(matrix_arr, index=rownames, columns=colnames)

    def treat(
        self,
        lfc: float = 1.0,
        robust: bool = False,
        trend: bool = False,
        **kwargs: Any
    ) -> "LimmaModel":
        """
        Test for differential expression relative to a fold-change threshold.
        
        Args:
            lfc: Log-fold-change threshold. Default: 1.0 (2-fold).
            robust: Use robust empirical Bayes.
            trend: Fit mean-variance trend.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            LimmaModel: New instance with TREAT fit.
        
        Example:
            >>> model_treat = model.treat(lfc=1.0)
            >>> results = model_treat.top_table()
        """
        r_fit = self.contrast_fit if self.contrast_fit is not None else self.lm_fit
        assert r_fit is not None, "lm_fit or contrast_fit must be set"
        
        r = get_r_environment()
        limma_pkg = _limma()
        
        call_kwargs: Dict[str, Any] = {"lfc": lfc, "robust": robust, "trend": trend}
        call_kwargs.update(kwargs)
        
        treat_fit = limma_pkg.treat(r_fit, **call_kwargs)
        
        return replace(self, lm_fit=treat_fit, ebayes=treat_fit, method="treat")


# --- lmFit wrapper ---------------------------------------------------------


def lm_fit(
    se: RESummarizedExperiment,
    design: pd.DataFrame,
    assay: str = "log_expr",
    ndups: Optional[int] = None,
    method: Literal["ls", "robust"] = "ls",
    return_result_object: bool = False,
    **kwargs: Any,
) -> LimmaModel:
    """Fit linear model to expression data using limma.

    Wraps the R ``limma::lmFit`` function to fit a linear model for each gene/feature.
    Returns a ``LimmaModel`` container with the fitted model object and metadata.

    Args:
        se: Input ``RESummarizedExperiment`` containing a ``"log_expr"`` assay
            (by convention) and optionally a ``"weights"`` assay for precision weights.
        design: Design matrix (samples × covariates) as a pandas DataFrame.
        assay: The assay in the RESummarizedExperiment to use.
        ndups: Number of technical replicates per unique sample. If ``None``, assumes
            no technical replication.
        method: Fitting method. Options: ``"ls"`` (least squares) or ``"robust"``
            (robust regression with M-estimation). Default: ``"ls"``.
        return_result_object: Reserved for future API extensions. Currently unused.
        **kwargs: Additional keyword arguments forwarded to ``limma::lmFit``.

    Returns:
        LimmaModel: Container object with the fitted R ``lmFit`` object, design matrix,
            and sample/feature names.

    Notes:
        - If ``"weights"`` assay exists, it is automatically used in the fitting.
        - The returned ``LimmaModel`` can be used with ``contrasts_fit`` and ``e_bayes``.

    Examples:
        >>> lm = lm_fit(se, design=design_df, method="robust")
        >>> lm_ebayes = lm.e_bayes()
    """
    _r = get_r_environment()
    limma_pkg = _limma()

    lmres = LimmaModel(
        method=method,
        sample_names=se.column_names,
        feature_names=se.row_names,
    )

    exprs_r = se.assay_r(assay)
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
