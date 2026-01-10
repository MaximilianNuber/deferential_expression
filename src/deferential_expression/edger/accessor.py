"""
EdgeR accessor for RESummarizedExperiment.

Provides edgeR differential expression methods via the accessor pattern.

Usage:
    import deferential_expression.edger  # Triggers accessor registration
    
    se = RESummarizedExperiment(...)
    se_norm = se.edger.calc_norm_factors(method="TMM")
    se_cpm = se_norm.edger.cpm()
    mask = se.edger.filter_by_expr()

All methods return new objects (functional/immutable style).
Matrix results go to assays via RMatrixAdapter.
Vector results go to column_data after Python conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union
import numpy as np
import pandas as pd

from ..extensions import register_rese_accessor
from ..resummarizedexperiment import RMatrixAdapter
from .utils import _prep_edger, pandas_to_r_matrix, numpy_to_r_matrix

if TYPE_CHECKING:
    from ..resummarizedexperiment import RESummarizedExperiment


@register_rese_accessor("edger")
class EdgeRAccessor:
    """
    Accessor providing edgeR differential expression methods.
    
    This accessor encapsulates all R FFI calls for edgeR in OOP methods.
    All results are written directly to the RESummarizedExperiment:
    - Matrix results → assays via RMatrixAdapter
    - Vector results → column_data after Python conversion
    
    Attributes:
        _se: Reference to the parent RESummarizedExperiment.
        _r: Cached rpy2 environment.
        _pkg: Cached edgeR R package.
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
    def _edger(self):
        """Lazy import of edgeR R package and environment."""
        if self._pkg is None:
            self._r, self._pkg = _prep_edger()
        return self._pkg
    
    @property
    def _renv(self):
        """Lazy access to rpy2 environment."""
        if self._r is None:
            self._r, self._pkg = _prep_edger()
        return self._r
    
    # =========================================================================
    # Methods that modify column_data (1D results)
    # =========================================================================
    
    def calc_norm_factors(
        self,
        assay: str = "counts",
        method: str = "TMM",
        refColumn: Optional[int] = None,
        logratioTrim: float = 0.3,
        sumTrim: float = 0.05,
        doWeighting: bool = True,
        Acutoff: float = -1e10,
        p: float = 0.75,
        **kwargs
    ) -> RESummarizedExperiment:
        """
        Compute normalization factors and store in column_data.
        
        Wraps ``edgeR::calcNormFactors``. The normalization factors are
        converted to a Python array and stored in column_data['norm.factors'].
        
        Args:
            assay: Name of the counts assay.
            method: Normalization method: "TMM", "RLE", "upperquartile", or "none".
            refColumn: Reference column for normalization (0-indexed, or None for auto).
            logratioTrim: Amount of trimming of the log-ratios (TMM only).
            sumTrim: Amount of trimming of intensity values (TMM only).
            doWeighting: Whether to use weighted trimmed mean (TMM only).
            Acutoff: Cutoff on average log-expression (TMM only).
            p: Quantile for upperquartile normalization.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            New RESummarizedExperiment with 'norm.factors' in column_data.
        
        Example:
            >>> se_norm = se.edger.calc_norm_factors(method="TMM")
            >>> se_norm.column_data["norm.factors"]
        """
        rmat = self._se.assay_r(assay)
        
        # Handle None -> R NULL
        refColumn_r = self._renv.ro.NULL if refColumn is None else refColumn
        
        r_factors = self._edger.calcNormFactors(
            rmat,
            method=method,
            refColumn=refColumn_r,
            logratioTrim=logratioTrim,
            sumTrim=sumTrim,
            doWeighting=doWeighting,
            Acutoff=Acutoff,
            p=p,
            **kwargs
        )
        
        # Convert to Python and store in column_data
        norm_factors = np.asarray(r_factors)
        coldata = self._se.get_column_data()
        new_coldata = coldata.set_column("norm.factors", norm_factors)
        
        return self._se.set_column_data(new_coldata)
    
    # =========================================================================
    # Methods that add assays (2D results)
    # =========================================================================
    
    def cpm(
        self,
        assay: str = "counts",
        log: bool = False,
        prior_count: float = 2.0,
        normalized_lib_sizes: bool = True,
        **kwargs
    ) -> RESummarizedExperiment:
        """
        Compute counts per million and add as new assay.
        
        Wraps ``edgeR::cpm``. The result is stored as an R-backed assay
        via RMatrixAdapter.
        
        Args:
            assay: Name of the input counts assay.
            log: Whether to compute log2-CPM.
            prior_count: Prior count to add before log transformation.
            normalized_lib_sizes: Whether to use normalized library sizes.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            New RESummarizedExperiment with 'cpm' (or 'logcpm' if log=True) assay.
        
        Example:
            >>> se_cpm = se.edger.cpm(log=True)
            >>> cpm_values = se_cpm.assay("cpm", as_numpy=True)
        """
        rmat = self._se.assay_r(assay)
        
        cpm_r = self._edger.cpm(
            rmat,
            log=log,
            prior_count=prior_count,
            normalized_lib_sizes=normalized_lib_sizes,
            **kwargs
        )
        
        # Store as R-backed assay
        assay_name = "logcpm" if log else "cpm"
        return self._se.set_assay(name=assay_name, value=RMatrixAdapter(cpm_r, self._renv))
    
    # =========================================================================
    # Methods that return masks/arrays (no SE mutation)
    # =========================================================================
    
    def filter_by_expr(
        self,
        assay: str = "counts",
        group: Optional[Sequence[str]] = None,
        design: Optional[pd.DataFrame] = None,
        lib_size: Optional[Sequence] = None,
        min_count: float = 10,
        min_total_count: float = 15,
        large_n: int = 10,
        min_prop: float = 0.7,
        **kwargs
    ) -> np.ndarray:
        """
        Compute expression filter mask using edgeR's filterByExpr.
        
        Returns a boolean mask indicating which genes pass the expression
        filter. Does NOT modify the SE - use the mask to subset manually.
        
        Args:
            assay: Name of the counts assay.
            group: Optional group factor for filtering.
            design: Optional design matrix as pandas DataFrame.
            lib_size: Optional library sizes.
            min_count: Minimum count required in at least some samples.
            min_total_count: Minimum total count across all samples.
            large_n: Number of samples where min_count must be exceeded.
            min_prop: Minimum proportion of samples with counts above min_count.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            Boolean numpy array mask (True = keep gene).
        
        Example:
            >>> mask = se.edger.filter_by_expr(min_count=10)
            >>> se_filtered = se[mask, :]
        """
        rmat = self._se.assay_r(assay)
        
        # Convert Python args to R
        group_r = self._renv.StrVector(group) if group is not None else self._renv.ro.NULL
        design_r = pandas_to_r_matrix(design) if design is not None else self._renv.ro.NULL
        lib_size_r = self._renv.ro.NULL if lib_size is None else self._renv.FloatVector(np.asarray(lib_size, dtype=float))
        
        mask_r = self._edger.filterByExpr(
            rmat,
            group=group_r,
            design=design_r,
            lib_size=lib_size_r,
            min_count=min_count,
            min_total_count=min_total_count,
            large_n=large_n,
            min_prop=min_prop,
            **kwargs
        )
        
        return np.asarray(mask_r, dtype=bool)
    
    # =========================================================================
    # GLM model fitting
    # =========================================================================
    
    def glm_ql_fit(
        self,
        design: pd.DataFrame,
        assay: str = "counts",
        dispersion: Optional[Union[pd.DataFrame, np.ndarray, float]] = None,
        offset: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        lib_size: Optional[Sequence] = None,
        weights: Optional[Union[float, Sequence[float], pd.DataFrame, np.ndarray]] = None,
        legacy: bool = False,
        top_proportion: float = 0.1,
        **kwargs
    ):
        """
        Fit quasi-likelihood GLM using edgeR::glmQLFit.
        
        Returns an EdgeRModel containing the fit object for downstream testing.
        
        Args:
            design: Design matrix (samples × covariates) as pandas DataFrame.
            assay: Counts assay name.
            dispersion: Optional dispersion values/matrix.
            offset: Optional offset matrix (e.g., log library sizes).
            lib_size: Optional library sizes per sample.
            weights: Optional observation/sample weights.
            legacy: Legacy mode flag for R function.
            top_proportion: Proportion parameter for robust fitting.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            EdgeRModel: Container with the fitted model.
        
        Example:
            >>> design = pd.DataFrame({'Intercept': [1]*6, 'Condition': [0,0,0,1,1,1]})
            >>> model = se.edger.glm_ql_fit(design)
            >>> results = model.test(coef=2)
        """
        from .glm_ql_fit import EdgeRModel, GlmQlFitConfig
        
        rmat = self._se.assay_r(assay)
        design_r = pandas_to_r_matrix(design)
        
        # Handle dispersion
        if dispersion is not None:
            if isinstance(dispersion, pd.DataFrame):
                dispersion_r = pandas_to_r_matrix(dispersion)
            elif isinstance(dispersion, np.ndarray):
                dispersion_r = numpy_to_r_matrix(dispersion)
            elif isinstance(dispersion, (int, float)):
                dispersion_r = self._renv.FloatVector([float(dispersion)])
            else:
                dispersion_r = dispersion
        else:
            dispersion_r = self._renv.ro.NULL
        
        # Handle offset - check for norm.factors in column_data
        if offset is not None:
            if isinstance(offset, pd.DataFrame):
                offset_r = pandas_to_r_matrix(offset)
            elif isinstance(offset, np.ndarray):
                offset_r = numpy_to_r_matrix(offset)
            else:
                offset_r = offset
        else:
            # Use norm.factors if available
            if "norm.factors" in self._se.column_data.column_names:
                nf = np.asarray(self._se.column_data["norm.factors"], dtype=float)
                offset_r = self._renv.FloatVector(nf)
            else:
                offset_r = self._renv.ro.NULL
        
        # Handle lib_size
        lib_size_r = self._renv.ro.NULL if lib_size is None else self._renv.FloatVector(np.asarray(lib_size, dtype=float))
        
        # Handle weights
        if weights is not None:
            if isinstance(weights, pd.DataFrame):
                weights_r = pandas_to_r_matrix(weights)
            elif isinstance(weights, np.ndarray):
                weights_r = numpy_to_r_matrix(weights)
            elif isinstance(weights, (int, float)):
                weights_r = self._renv.FloatVector([float(weights)])
            elif isinstance(weights, Sequence):
                weights_r = self._renv.FloatVector(np.asarray(weights, dtype=float))
            else:
                weights_r = weights
        else:
            weights_r = self._renv.ro.NULL
        
        # Call glmQLFit_default
        fit_obj = self._edger.glmQLFit_default(
            rmat,
            design=design_r,
            dispersion=dispersion_r,
            offset=offset_r,
            weights=weights_r,
            legacy=legacy,
            top_proportion=top_proportion,
            **kwargs
        )
        
        config = GlmQlFitConfig(
            dispersion=dispersion,
            offset=offset,
            lib_size=lib_size,
            weights=weights,
            legacy=legacy,
            top_proportion=top_proportion,
            assay=assay,
            user_kwargs=kwargs
        )
        
        return EdgeRModel(
            sample_names=self._se.column_names,
            feature_names=self._se.row_names,
            fit=fit_obj,
            fit_config=config,
            design=design,
        )
    
    def glm_ql_ftest(
        self,
        model,
        coef: Optional[Union[str, int]] = None,
        contrast: Optional[Sequence] = None,
        poisson_bound: bool = True,
        adjust_method: str = "BH",
    ) -> pd.DataFrame:
        """
        Run quasi-likelihood F-test on a fitted model.
        
        Args:
            model: EdgeRModel from glm_ql_fit.
            coef: Coefficient name (str) or index (int) to test.
            contrast: Contrast vector (alternative to coef).
            poisson_bound: Whether to apply Poisson bound.
            adjust_method: Multiple testing adjustment method.
        
        Returns:
            pd.DataFrame: Results table with log fold-change, p-values, etc.
        """
        assert hasattr(model, "fit") and model.fit is not None, "Model must have a valid fit"
        
        # Handle coef
        if coef is not None:
            if isinstance(coef, int):
                coef_r = self._renv.IntVector([coef])
            else:
                coef_r = self._renv.StrVector([str(coef)])
        else:
            coef_r = self._renv.ro.NULL
        
        # Handle contrast
        if contrast is not None:
            contrast_r = self._renv.IntVector(np.asarray(contrast, dtype=int))
        else:
            contrast_r = self._renv.ro.NULL
        
        poisson_bound_r = self._renv.BoolVector([poisson_bound])
        
        # Run test
        res = self._edger.glmQLFTest(
            model.fit,
            coef=coef_r,
            contrast=contrast_r,
            poisson_bound=poisson_bound_r
        )
        
        # Extract results via topTags
        res = self._edger.topTags(
            res,
            n=self._renv.ro.r("Inf"),
            adjust_method=adjust_method,
            sort_by=self._renv.ro.NULL,
            p_value=self._renv.IntVector([1])
        )
        
        # Convert to pandas
        res = self._renv.ro.baseenv["as.data.frame"](res)
        with self._renv.localconverter(self._renv.default_converter + self._renv.pandas2ri.converter):
            df = self._renv.get_conversion().rpy2py(res)
        
        return df
    
    def top_tags(
        self,
        lrt_obj: Any,
        n: Optional[int] = None,
        adjust_method: str = "BH",
        sort_by: str = "PValue",
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract top-ranked genes from a test result.
        
        Args:
            lrt_obj: R object from glmQLFTest or similar.
            n: Number of top genes to return (None = all).
            adjust_method: Multiple testing correction method.
            sort_by: Column to sort by.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            pd.DataFrame: Results table with standardized column names.
        """
        if n is None:
            n_genes = int(self._renv.r2py(self._renv.ro.baseenv["nrow"](lrt_obj)))
            n = n_genes
        
        top_r = self._edger.topTags(
            lrt_obj,
            n=n,
            **{"adjust.method": adjust_method, "sort.by": sort_by},
            **kwargs
        )
        
        table_r = self._renv.ro.baseenv["$"](top_r, "table")
        
        with self._renv.localconverter(self._renv.default_converter + self._renv.pandas2ri.converter):
            df = self._renv.r2py(table_r)
        
        df = df.reset_index(names="gene")
        df = df.rename(columns={
            "PValue": "p_value",
            "FDR": "adj_p_value",
            "logFC": "log_fc",
            "logCPM": "log_cpm",
            "LR": "lr_statistic",
        })
        
        return df


def activate():
    """
    Called on module import to register the EdgeR accessor.
    
    Registration actually happens via the @register_rese_accessor decorator
    when the class is defined, so this function exists primarily for
    documentation and as an explicit hook point.
    """
    pass  # Registration happens via decorator at class definition
