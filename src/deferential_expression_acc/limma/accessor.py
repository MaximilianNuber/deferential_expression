"""
Limma accessor for RESummarizedExperiment.

Provides limma differential expression methods via the accessor pattern.

Usage:
    import deferential_expression_acc.limma  # Triggers accessor registration
    
    se = RESummarizedExperiment(...)
    se_voom = se.limma.voom(design)
    model = se_voom.limma.lm_fit(design)
    results = se_voom.limma.top_table(model)

All methods return new objects (functional/immutable style).
Matrix results go to assays via RMatrixAdapter.
Vector results go to column_data after Python conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union, Tuple, Literal
import numpy as np
import pandas as pd

from ..extensions import register_rese_accessor
from ..resummarizedexperiment import RMatrixAdapter

if TYPE_CHECKING:
    from ..resummarizedexperiment import RESummarizedExperiment
    from .lm_fit import LimmaModel


@register_rese_accessor("limma")
class LimmaAccessor:
    """
    Accessor providing limma differential expression methods.
    
    This accessor encapsulates all R FFI calls for limma in OOP methods.
    All results are written directly to the RESummarizedExperiment:
    - Matrix results → assays via RMatrixAdapter
    - Vector results → column_data after Python conversion
    
    Attributes:
        _se: Reference to the parent RESummarizedExperiment.
        _r: Cached rpy2 environment.
        _pkg: Cached limma R package.
    """
    
    def __init__(self, se: RESummarizedExperiment) -> None:
        """
        Initialize the accessor.
        
        Args:
            se: The parent RESummarizedExperiment instance.
        """
        self._se = se
        self._r: Any = None
        self._pkg: Any = None
    
    @property
    def _limma(self):
        """Lazy import of limma R package."""
        if self._pkg is None:
            from .utils import _limma
            self._pkg = _limma()
        return self._pkg
    
    @property
    def _renv(self):
        """Lazy access to rpy2 environment."""
        if self._r is None:
            from bioc2ri.lazy_r_env import get_r_environment
            self._r = get_r_environment()
        return self._r
    
    # =========================================================================
    # Preprocessing methods (return new SE with new assays)
    # =========================================================================
    
    def voom(
        self,
        design: pd.DataFrame,
        lib_size: Optional[Union[pd.Series, Sequence, np.ndarray]] = None,
        block: Optional[Union[pd.Series, Sequence, np.ndarray, pd.Categorical]] = None,
        log_expr_assay: str = "log_expr",
        weights_assay: str = "weights",
        plot: bool = False,
        **kwargs
    ) -> RESummarizedExperiment:
        """
        Run voom transformation on counts to compute log-CPM and weights.
        
        Results stored as assays: 'log_expr' and 'weights'.
        
        Args:
            design: Design matrix (samples × covariates) as pandas DataFrame.
            lib_size: Optional library sizes per sample.
            block: Optional blocking factor (e.g., batch).
            log_expr_assay: Name for log-expression assay.
            weights_assay: Name for weights assay.
            plot: Whether to show voom diagnostic plot.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            New RESummarizedExperiment with log_expr and weights assays.
        """
        from ..edger.utils import pandas_to_r_matrix
        
        counts_r = self._se.assay_r("counts")
        design_r = pandas_to_r_matrix(design)
        
        # Handle lib_size
        if lib_size is not None:
            lib_size = np.asarray(lib_size, dtype=float)
            lib_size_r = self._renv.FloatVector(lib_size)
        elif "norm.factors" in self._se.column_data.column_names:
            nf = np.asarray(self._se.column_data["norm.factors"], dtype=float)
            lib_size_r = self._renv.FloatVector(nf)
        else:
            lib_size_r = self._renv.ro.NULL
        
        # Handle block
        if block is not None:
            if isinstance(block, pd.Categorical):
                with self._renv.localconverter(self._renv.default_converter + self._renv.pandas2ri.converter):
                    block_r = self._renv.get_conversion().py2rpy(block)
            else:
                block_r = self._renv.ro.StrVector(np.asarray(block, dtype=str))
        else:
            block_r = self._renv.ro.NULL
        
        # Call voom
        voom_out = self._limma.voom(
            counts_r, design_r,
            plot=plot,
            lib_size=lib_size_r,
            block=block_r,
            **kwargs
        )
        
        # Extract E and weights
        E_r = self._renv.ro.baseenv["[["](voom_out, "E")
        weights_r = self._renv.ro.baseenv["[["](voom_out, "weights")
        
        # Create new SE with new assays
        assays = dict(self._se.assays)
        assays[log_expr_assay] = RMatrixAdapter(E_r, self._renv)
        assays[weights_assay] = RMatrixAdapter(weights_r, self._renv)
        
        from ..resummarizedexperiment import RESummarizedExperiment
        return RESummarizedExperiment(
            assays=assays,
            row_data=self._se.row_data_df,
            column_data=self._se.column_data_df,
            row_names=self._se.row_names,
            column_names=self._se.column_names,
            metadata=dict(self._se.metadata),
        )
    
    def normalize_between_arrays(
        self,
        exprs_assay: str = "log_expr",
        normalized_assay: str = "log_expr_norm",
        method: str = "quantile",
        **kwargs
    ) -> RESummarizedExperiment:
        """
        Normalize expression values between arrays/samples.
        
        Args:
            exprs_assay: Input expression assay name.
            normalized_assay: Output normalized assay name.
            method: Normalization method ("quantile", "scale", "cyclicloess", etc.).
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            New RESummarizedExperiment with normalized assay.
        """
        exprs_r = self._se.assay_r(exprs_assay)
        out_r = self._limma.normalizeBetweenArrays(exprs_r, method=method, **kwargs)
        
        assays = dict(self._se.assays)
        assays[normalized_assay] = RMatrixAdapter(out_r, self._renv)
        
        from ..resummarizedexperiment import RESummarizedExperiment
        return RESummarizedExperiment(
            assays=assays,
            row_data=self._se.row_data_df,
            column_data=self._se.column_data_df,
            row_names=self._se.row_names,
            column_names=self._se.column_names,
            metadata=dict(self._se.metadata),
        )
    
    def remove_batch_effect(
        self,
        batch: Union[str, Sequence, np.ndarray, pd.Series],
        batch2: Union[str, Sequence, np.ndarray, pd.Series, None] = None,
        exprs_assay: str = "log_expr",
        corrected_assay: str = "log_expr_bc",
        design: Optional[pd.DataFrame] = None,
        covariates: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> RESummarizedExperiment:
        """
        Remove batch effects from expression data.
        
        Args:
            batch: Primary batch factor (column name or array).
            batch2: Optional secondary batch factor.
            exprs_assay: Input expression assay name.
            corrected_assay: Output batch-corrected assay name.
            design: Optional design matrix to preserve biological variation.
            covariates: Optional continuous covariates to adjust for.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            New RESummarizedExperiment with batch-corrected assay.
        """
        from ..edger.utils import pandas_to_r_matrix
        from rpy2.robjects.vectors import StrVector
        
        E_r = self._se.assay_r(exprs_assay)
        
        # Resolve batch
        def _resolve_batch(b):
            if isinstance(b, str):
                cd = self._se.column_data_df
                if cd is None or b not in cd.columns:
                    raise KeyError(f"Batch column '{b}' not found in column_data.")
                arr = cd[b].to_numpy()
            else:
                arr = np.asarray(b)
            vals = ["" if v is None else str(v) for v in arr.tolist()]
            return StrVector(vals)
        
        call_kwargs = {"batch": _resolve_batch(batch)}
        
        if batch2 is not None:
            call_kwargs["batch2"] = _resolve_batch(batch2)
        
        if covariates is not None:
            call_kwargs["covariates"] = pandas_to_r_matrix(covariates)
        
        if design is not None:
            call_kwargs["design"] = pandas_to_r_matrix(design)
        
        call_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        out_r = self._limma.removeBatchEffect(E_r, **call_kwargs)
        
        assays = dict(self._se.assays)
        assays[corrected_assay] = RMatrixAdapter(out_r, self._renv)
        
        from ..resummarizedexperiment import RESummarizedExperiment
        return RESummarizedExperiment(
            assays=assays,
            row_data=self._se.row_data_df,
            column_data=self._se.column_data_df,
            row_names=self._se.row_names,
            column_names=self._se.column_names,
            metadata=dict(self._se.metadata),
        )
    
    # =========================================================================
    # Model fitting (returns LimmaModel)
    # =========================================================================
    
    def lm_fit(
        self,
        design: pd.DataFrame,
        assay: str = "log_expr",
        ndups: Optional[int] = None,
        method: Literal["ls", "robust"] = "ls",
        **kwargs: Any,
    ) -> LimmaModel:
        """
        Fit linear model to expression data.
        
        Args:
            design: Design matrix (samples × covariates) as pandas DataFrame.
            assay: Expression assay to use.
            ndups: Number of technical replicates.
            method: Fitting method ("ls" or "robust").
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            LimmaModel: Container with fitted model.
        """
        from ..edger.utils import pandas_to_r_matrix
        from .lm_fit import LimmaModel
        
        exprs_r = self._se.assay_r(assay)
        design_r = pandas_to_r_matrix(design)
        
        # Handle weights
        if "weights" in self._se.assay_names:
            weights = self._se.assay_r("weights")
        else:
            weights = self._renv.ro.NULL
        
        ndups_r = self._renv.ro.NULL if ndups is None else ndups
        
        fit = self._limma.lmFit(
            exprs_r,
            design_r,
            weights=weights,
            method=method,
            **kwargs,
        )
        
        return LimmaModel(
            sample_names=self._se.column_names,
            feature_names=self._se.row_names,
            lm_fit=fit,
            design=design,
            ndups=ndups,
            method=method,
        )
    
    def contrasts_fit(
        self,
        lm_obj: LimmaModel,
        contrast: Sequence[Union[int, float]],
    ) -> LimmaModel:
        """
        Apply contrast to fitted linear model.
        
        Args:
            lm_obj: LimmaModel from lm_fit.
            contrast: 1D contrast vector.
        
        Returns:
            LimmaModel: With contrast_fit slot set.
        """
        from dataclasses import replace
        from .lm_fit import LimmaModel
        
        assert lm_obj.lm_fit is not None, "lm_fit must be set"
        
        contrast_arr = np.asarray(contrast, dtype=float)
        contrast_r = self._renv.ro.FloatVector(contrast_arr)
        
        fit_r = self._limma.contrasts_fit(lm_obj.lm_fit, contrast=contrast_r)
        
        return replace(lm_obj, contrast_fit=fit_r)
    
    def e_bayes(
        self,
        lm_obj: LimmaModel,
        proportion: float = 0.01,
        trend: bool = False,
        robust: bool = False,
        **kwargs: Any
    ) -> LimmaModel:
        """
        Compute empirical Bayes moderated statistics.
        
        Args:
            lm_obj: LimmaModel from lm_fit or contrasts_fit.
            proportion: Assumed proportion of DE genes.
            trend: Fit mean-variance trend.
            robust: Use robust empirical Bayes.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            LimmaModel: With ebayes slot set.
        """
        from dataclasses import replace
        
        r_fit = lm_obj.contrast_fit if lm_obj.contrast_fit is not None else lm_obj.lm_fit
        assert r_fit is not None, "lm_fit or contrast_fit must be set"
        
        call_kwargs = {"proportion": proportion, "trend": trend, "robust": robust}
        call_kwargs.update(kwargs)
        
        eb = self._limma.eBayes(r_fit, **call_kwargs)
        
        return replace(lm_obj, ebayes=eb)
    
    def top_table(
        self,
        lm_obj: LimmaModel,
        coef: Optional[Union[int, str]] = None,
        n: Optional[int] = None,
        adjust_method: str = "BH",
        sort_by: str = "PValue",
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Extract top-ranked genes from differential expression analysis.
        
        Args:
            lm_obj: LimmaModel (will run eBayes if not done).
            coef: Coefficient to extract.
            n: Number of top genes (None = all).
            adjust_method: Multiple testing method.
            sort_by: Sort column.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            pd.DataFrame: Results table with standardized column names.
        """
        # Use ebayes if available, otherwise compute it
        if lm_obj.ebayes is not None:
            eb = lm_obj.ebayes
        else:
            lm_obj = self.e_bayes(lm_obj)
            eb = lm_obj.ebayes
        
        if n is None:
            n = int(self._renv.r2py(self._renv.ro.baseenv["nrow"](eb)))
        
        # Handle coef for contrast fits
        if lm_obj.contrast_fit is not None and coef is None:
            coef = 1
        
        # Map sort_by
        if coef is None:
            sort_by_map = {"PValue": "F", "logFC": "F", "AveExpr": "F", "B": "F", "F": "F", "none": "none"}
            sort_by_r = sort_by_map.get(sort_by, "F")
            coef_r = self._renv.ro.NULL
        else:
            sort_by_map = {"PValue": "p", "logFC": "logFC", "AveExpr": "AveExpr", "B": "B", "F": "B", "none": "none"}
            sort_by_r = sort_by_map.get(sort_by, "p")
            coef_r = coef
        
        call_kwargs = {"number": n, "adjust.method": adjust_method, "coef": coef_r, "sort.by": sort_by_r}
        call_kwargs.update(kwargs)
        
        top_r = self._limma.topTable(eb, **call_kwargs)
        
        with self._renv.localconverter(self._renv.default_converter + self._renv.pandas2ri.converter):
            df = self._renv.get_conversion().rpy2py(top_r)
        
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
        lm_obj: LimmaModel,
        method: str = "separate",
        adjust_method: str = "BH",
        p_value: float = 0.05,
        lfc: float = 0,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Classify genes as significantly up, down, or not significant.
        
        Args:
            lm_obj: LimmaModel (will run eBayes if not done).
            method: Testing method ("separate", "global", etc.).
            adjust_method: Multiple testing method.
            p_value: Significance threshold.
            lfc: Log-fold-change threshold.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            pd.DataFrame: Values -1 (down), 0 (not sig), 1 (up).
        """
        # Use ebayes if available, otherwise compute it
        if lm_obj.ebayes is not None:
            eb = lm_obj.ebayes
        else:
            lm_obj = self.e_bayes(lm_obj)
            eb = lm_obj.ebayes
        
        call_kwargs = {
            "method": method,
            "adjust.method": adjust_method,
            "p.value": p_value,
            "lfc": lfc
        }
        call_kwargs.update(kwargs)
        
        decide_r = self._limma.decideTests(eb, **call_kwargs)
        
        r_rownames = self._renv.ro.baseenv["rownames"](decide_r)
        r_colnames = self._renv.ro.baseenv["colnames"](decide_r)
        
        rownames = list(self._renv.r2py(r_rownames)) if r_rownames is not self._renv.ro.NULL else None
        colnames = list(self._renv.r2py(r_colnames)) if r_colnames is not self._renv.ro.NULL else None
        
        arr = self._renv.ro.baseenv["as.matrix"](decide_r)
        with self._renv.localconverter(self._renv.default_converter + self._renv.pandas2ri.converter):
            matrix_arr = self._renv.r2py(arr)
        
        return pd.DataFrame(matrix_arr, index=rownames, columns=colnames)
    
    def treat(
        self,
        lm_obj: LimmaModel,
        lfc: float = 1.0,
        robust: bool = False,
        trend: bool = False,
        **kwargs: Any
    ) -> LimmaModel:
        """
        Test for differential expression relative to a fold-change threshold.
        
        Args:
            lm_obj: LimmaModel from lm_fit or contrasts_fit.
            lfc: Log-fold-change threshold.
            robust: Use robust empirical Bayes.
            trend: Fit mean-variance trend.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            LimmaModel: With TREAT fit.
        """
        from dataclasses import replace
        
        r_fit = lm_obj.contrast_fit if lm_obj.contrast_fit is not None else lm_obj.lm_fit
        assert r_fit is not None, "lm_fit or contrast_fit must be set"
        
        call_kwargs = {"lfc": lfc, "robust": robust, "trend": trend}
        call_kwargs.update(kwargs)
        
        treat_fit = self._limma.treat(r_fit, **call_kwargs)
        
        return replace(lm_obj, lm_fit=treat_fit, method="treat")


def activate():
    """
    Called on module import to register the Limma accessor.
    
    Registration actually happens via the @register_rese_accessor decorator
    when the class is defined, so this function exists primarily for
    documentation and as an explicit hook point.
    """
    pass  # Registration happens via decorator at class definition
