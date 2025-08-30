from __future__ import annotations

from typing import Optional, Sequence, Union, Tuple, Literal
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Union, Tuple
from dataclasses import dataclass
from dataclasses import replace

# rpy2 manager and converters
from pyrtools.lazy_r_env import get_r_environment, r
from pyrtools.r_converters import RConverters
_rmana = get_r_environment()


# Your adapter & converters (assumes you defined these elsewhere in the package)
# from deferential_expression.edger_RESummarizedExperiment import RMatrixAdapter
from .resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter, _df_to_r_matrix, _df_to_r_df

# Helper to lazy-load limma
_limma_pkg: Any = None
def _limma() -> Any:
    """Lazily import and return the R `limma` package via rpy2.

    Returns:
        Any: An object handle to the imported R `limma` package as exposed by rpy2.

    Notes:
        The package is imported only once and cached in a module-level variable
        for subsequent calls.
    """
    global _limma_pkg
    if _limma_pkg is None:
        _r = _rmana
        _limma_pkg = _r.importr("limma")
    return _limma_pkg
#------
# Limma RESummarizedExperiment
#------
class Limma(RESummarizedExperiment):
    """An `RESummarizedExperiment` subclass that stores limma fit artifacts.

    This class augments an `RESummarizedExperiment` with slots for common
    limma analysis outputs (e.g., `lm_fit`, `contrast_fit`, `ebayes`) to
    enable fluent workflows while preserving assay and metadata containers.

    Attributes:
        lm_fit: R object produced by `limma::lmFit`.
        contrast_fit: R object produced by `limma::contrasts.fit`.
        ebayes: R object produced by `limma::eBayes`.
    """
    __slots__ = ("dge", "disp", "glm", "lrt")  # add others as you need

    def __init__(self, *, assays = None, row_data=None, column_data=None,
                 row_names=None, column_names=None, metadata=None, **kwargs):
        """Initialize a `Limma` object.

        Args:
            assays: Mapping of assay name to matrix-like objects. R-backed
                matrices should be wrapped in `RMatrixAdapter`.
            row_data: Optional row (feature) annotations; typically a pandas
                DataFrame or BiocFrame-compatible object.
            column_data: Optional column (sample) annotations; typically a pandas
                DataFrame or BiocFrame-compatible object.
            row_names: Optional sequence of row (feature) names.
            column_names: Optional sequence of column (sample) names.
            metadata: Optional free-form dictionary to store analysis metadata.
            **kwargs: Forwarded to the base `RESummarizedExperiment` constructor.

        Notes:
            Initializes limma-related slots (`lm_fit`, `contrast_fit`, `ebayes`)
            to `None`. Assays and annotations are delegated to the base class.
        """
        # initialize R object slots
        self.lm_fit = None
        self.contrast_fit = None
        self.ebayes = None
        super().__init__(assays=assays,
                         row_data=row_data,
                         column_data=column_data,
                         row_names=row_names,
                         column_names=column_names,
                         metadata=metadata,
                         **kwargs)

    # ---------- internal helper to clone ----------
    def _clone(self,
               *,
               assays=None,
               row_data=None,
               column_data=None,
               metadata=None,
               lm_fit = None,
               contrast_fit = None,
               ebayes = None) -> Limma:
        """Create a new `Limma` with updated fields while copying the rest.

        Args:
            assays: Optional replacement assays mapping.
            row_data: Optional replacement row annotations.
            column_data: Optional replacement column annotations.
            metadata: Optional replacement metadata dictionary.
            lm_fit: Optional replacement for the stored `lm_fit` R object.
            contrast_fit: Optional replacement for the stored `contrast_fit` R object.
            ebayes: Optional replacement for the stored `ebayes` R object.

        Returns:
            Limma: A new `Limma` instance with requested replacements applied,
            preserving any existing values for unspecified fields.
        """
        return type(self)(
            assays=assays if assays is not None else dict(self.assays),
            row_data=row_data if row_data is not None else self.row_data,
            column_data=column_data if column_data is not None else self.column_data,
            row_names=self.row_names,
            column_names=self.column_names,
            metadata=metadata if metadata is not None else dict(self.metadata)
        )._set_r_objs(lm_fit if lm_fit is not None else self.lm_fit,
                      contrast_fit if contrast_fit is not None else self.contrast_fit,
                      ebayes if ebayes is not None else self.ebayes)

    def _set_r_objs(self, lm_fit, contrast_fit, ebayes) -> Limma:
        """Set limma-related R objects on the instance.

        Args:
            lm_fit: R object from `limma::lmFit`.
            contrast_fit: R object from `limma::contrasts.fit`.
            ebayes: R object from `limma::eBayes`.

        Returns:
            Limma: The same instance (for fluent chaining).
        """
        self.lm_fit = lm_fit
        self.contrast_fit = contrast_fit
        self.ebayes = ebayes
        return self

    @property
    def samples(self) -> pd.DataFrame:
        """Return column (sample) annotations as a pandas DataFrame.

        Returns:
            pandas.DataFrame: The `column_data` converted to a pandas DataFrame.
        """
        return self.column_data.to_pandas()


# ——— 1) voom.default → log-expression + weights ———
def voom(
    se: RESummarizedExperiment,
    design: pd.DataFrame,
    lib_size: Optional[Union[pd.Series, Sequence, np.ndarray]] = None,
    block: Optional[Union[pd.Series, Sequence, np.ndarray, pd.Categorical]] = None,
    log_expr_assay: str = "log_expr",
    weights_assay: str = "weights",
    plot: bool = False,
    **kwargs
) -> RESummarizedExperiment:
    """Run `limma::voom` (default) on counts to compute log-CPM and weights.

    This wraps the R `voom` default method using an R-backed counts assay and a
    pandas design matrix. It writes two new assays into the returned object:
    `log_expr_assay` (log-CPM matrix) and `weights_assay` (precision weights).

    Args:
        se: Input `RESummarizedExperiment` containing a `"counts"` R-backed assay.
        design: Design matrix (samples × covariates) as a pandas DataFrame.
        lib_size: Optional library sizes per sample (length = n_samples). If not
            provided, attempts to fall back to `column_data['norm.factors']` if present.
        block: Optional blocking factor (e.g., batch) as array-like or pandas
            Categorical for voom’s correlation/duplicateCorrelation workflows.
        log_expr_assay: Name for the output log-expression assay.
        weights_assay: Name for the output weights assay.
        plot: Whether to enable voom’s diagnostic plot in R.
        **kwargs: Additional keyword arguments forwarded to `limma::voom`.

    Returns:
        RESummarizedExperiment: A new object with added assays for log-CPM and weights.

    Notes:
        This function assumes the `"counts"` assay is an R matrix accessible
        via `se.assay_r("counts")`. New assays are stored as `RMatrixAdapter`s.
    """
    limma_pkg = _limma()
    r = _rmana
    # extract raw counts R matrix
    counts_r = se.assay_r("counts")
    # R design
    design_r = _df_to_r_matrix(design)

    # convert lib_size to R if provided
    if lib_size is not None:
        lib_size = np.asarray(lib_size, dtype=float)
        assert lib_size.ndim == 1, "lib_size must be a 1D array-like"
        lib_size = r.FloatVector(lib_size)
    else:
        if "norm.factors" in se.column_data.column_names:
            # use norm factors as lib_size if available
            lib_size = se.column_data["norm.factors"]
            lib_size = np.asarray(lib_size, dtype=float)
            
            lib_size = r.FloatVector(lib_size)
        else:
            lib_size = r.ro.NULL


    if block is not None:
        if isinstance(block, pd.Categorical):
            with r.localconverter(r.default_converter + r.pandas2ri.converter):
                block = r.get_conversion().py2rpy(block)
        else:
            block = np.asarray(block, dtype=str)
            block = r.ro.StrVector(block)
    else:
        block = r.ro.NULL

    

    # call voom.default
    voom_fn = limma_pkg.voom  # direct .default
    voom_out = voom_fn(counts_r, design_r, plot=plot, lib_size = lib_size, block = block, **kwargs)
    
    # voom_out is an EList-like list with $E and $weights slots
    E_r       = r.ro.baseenv["[["](voom_out, "E")
    weights_r = r.ro.baseenv["[["](voom_out, "weights")
    
    # wrap and insert into assays
    assays = dict(se.assays)
    assays[log_expr_assay]    = RMatrixAdapter(E_r, r)
    assays[weights_assay]     = RMatrixAdapter(weights_r, r)
    
    return RESummarizedExperiment(
        assays=assays,
        row_data=se.row_data_df,
        column_data=se.column_data_df,
        row_names=se.row_names,
        column_names=se.column_names,
        metadata=dict(se.metadata),
    )

# ——— 2) voomWithQualityWeights.default — same pattern ———
def voom_quality_weights(
    se: RESummarizedExperiment,
    design: pd.DataFrame,
    log_expr_assay: str = "log_expr_qw",
    weights_assay: str = "weights",
    plot: bool = False,
    **kwargs
) -> RESummarizedExperiment:
    """Run `limma::voomWithQualityWeights` (default) to compute log-CPM and QW weights.

    Args:
        se: Input `RESummarizedExperiment` containing a `"counts"` R-backed assay.
        design: Design matrix (samples × covariates) as a pandas DataFrame.
        log_expr_assay: Name for the output log-expression assay.
        weights_assay: Name for the output weights assay.
        plot: Whether to enable voom’s diagnostic plot in R.
        **kwargs: Additional keyword arguments forwarded to `voomWithQualityWeights`.

    Returns:
        RESummarizedExperiment: A new object with `log_expr_assay` and `weights_assay`
        assays added as `RMatrixAdapter`s.
    """
    limma = _limma()
    _r = _rmana
    counts_r = se.assay_r("counts")
    design_r = _df_to_r_matrix(design)
    voom_qw_fn = limma.voomWithQualityWeights
    out = voom_qw_fn(counts_r, design_r, plot=plot, **kwargs)
    E_r       = _r.ro.baseenv["[["](out, "E")
    weights_r = _r.ro.baseenv["[["](out, "weights")
    assays = dict(se.assays)
    assays[log_expr_assay]    = RMatrixAdapter(E_r, _r)
    assays[weights_assay]     = RMatrixAdapter(weights_r, _r)
    return RESummarizedExperiment(
        assays=assays,
        row_data=se.row_data_df,
        column_data=se.column_data_df,
        row_names=se.row_names,
        column_names=se.column_names,
        metadata=dict(se.metadata),
    )

# ——— 3) normalizeBetweenArrays.default ———
def normalize_between_arrays(
    se: RESummarizedExperiment,
    exprs_assay: str = "log_expr",           # which assay to normalize
    normalized_assay: str = "log_expr_norm",
    method: str = "quantile",
    **kwargs
) -> RESummarizedExperiment:
    """Run `limma::normalizeBetweenArrays` on an expression assay.

    Args:
        se: Input `RESummarizedExperiment` with an R-backed expression assay.
        exprs_assay: Name of the input expression assay to normalize.
        normalized_assay: Name for the output normalized assay.
        method: Normalization method (e.g., `"quantile"`, `"scale"`, etc.).
        **kwargs: Additional keyword arguments forwarded to `normalizeBetweenArrays`.

    Returns:
        RESummarizedExperiment: A new object with the normalized assay stored
        under `normalized_assay` as an `RMatrixAdapter`.
    """
    limma = _limma()
    _r = _rmana
    # get the E matrix (either R mat or python)
    exprs_r = se.assay_r(exprs_assay)
    norm_fn = limma.normalizeBetweenArrays
    out_r = norm_fn(exprs_r, method=method, **kwargs)
    assays = dict(se.assays)
    assays[normalized_assay] = RMatrixAdapter(out_r, _r)
    return RESummarizedExperiment(
        assays=assays,
        row_data=se.row_data_df,
        column_data=se.column_data_df,
        row_names=se.row_names,
        column_names=se.column_names,
        metadata=dict(se.metadata),
    )

# ——— 4) removeBatchEffect.default ———
def remove_batch_effect(
    se: RESummarizedExperiment,
    batch: Union[pd.Series, Sequence, np.ndarray],
    exprs_assay: str = "log_expr_norm",
    corrected_assay: str = "log_expr_bc",
    design: Optional[pd.DataFrame] = None,
    **kwargs
) -> RESummarizedExperiment:
    """Run `limma::removeBatchEffect` to correct an expression assay.

    Args:
        se: Input `RESummarizedExperiment` with an R-backed expression assay.
        batch: Batch labels per sample (length = n_samples).
        exprs_assay: Name of the input expression assay to correct.
        corrected_assay: Name for the output batch-corrected assay.
        design: Optional design matrix (samples × covariates) used as covariates
            to protect biological signal during batch correction.
        **kwargs: Additional keyword arguments forwarded to `removeBatchEffect`.

    Returns:
        RESummarizedExperiment: A new object with the batch-corrected assay
        stored under `corrected_assay` as an `RMatrixAdapter`.
    """
    limma = _limma()
    _r = _rmana
    E_r = se.assay_r(exprs_assay)
    # batch → R
    batch = np.asarray(batch, dtype=str)
    
    batch_r = _r.StrVector(batch)

    # design optional
    design_r = _df_to_r_matrix(design) if design is not None else _r.ro.NULL
    rbe = limma.removeBatchEffect
    out_r = rbe(E_r, batch=batch_r, design=design_r, **kwargs)
    assays = dict(se.assays)
    assays[corrected_assay] = RMatrixAdapter(out_r, _r)
    return RESummarizedExperiment(
        assays=assays,
        row_data=se.row_data_df,
        column_data=se.column_data_df,
        row_names=se.row_names,
        column_names=se.column_names,
        metadata=dict(se.metadata),
    )

# ——— 5) lmFit.default — store coefficients etc. in metadata ———

@dataclass
class LimmaModel:
    """Container for limma results including fit, coefficients, and metadata.

    This class encapsulates the results of fitting a linear model using limma,
    storing the R objects produced by `lmFit`, `contrasts.fit`, and `eBayes`.
    It also holds sample and feature names, the design matrix, fitting method,
    and extracted coefficients as a pandas DataFrame.

    Args:
        sample_names: Optional sequence of sample names (column names).
        feature_names: Optional sequence of feature names (row names).
        lm_fit: Optional R object from `limma::lmFit`.
        contrast_fit: Optional R object from `limma::contrasts.fit`.
        ebayes: Optional R object from `limma::eBayes`.
        design: Optional design matrix (samples × covariates) as a pandas DataFrame.
        ndups: Optional number of technical replicates (if applicable).
        method: Fitting method used, e.g., `"ls"` (least squares) or `"robust"`.
        coefficients: Optional pandas DataFrame of extracted coefficients.
        metadata: Optional free-form dictionary for additional metadata.
    """
    sample_names: Optional[Sequence[str]] = None  # Sample names (column names)
    feature_names: Optional[Sequence[str]] = None  # Feature names (row names)
    lm_fit: Optional[Any] = None  # R object from lmFit
    contrast_fit: Optional[Any] = None  # R object from contrasts.fit
    ebayes: Optional[Any] = None  # R object from eBayes    

    design: Optional[pd.DataFrame] = None  # Design matrix used for fitting
    ndups: Optional[int] = None  # Number of technical replicates (if applicable
    method: Literal["ls", "robust"] = "ls"  # Fitting method used

    coefficients: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None

    def get_sample_names(self) -> Sequence[str]:
        """Get sample names from the design matrix or fit object."""
        if self.sample_names is not None:
            return self.sample_names

        if self.lm_fit is not None:
            r = get_r_environment()
            # return list(r.ro.baseenv["colnames"](r.ro.baseenv["(self.lm_fit)))
            return list(r.ro.baseenv["colnames"](self.lm_fit))
        return ()
    def get_feature_names(self) -> Sequence[str]:
        """Get feature names from the design matrix or fit object."""
        if self.feature_names is not None:
            return self.feature_names

        if self.lm_fit is not None:
            r = get_r_environment()
            return list(r.ro.baseenv["rownames"](self.lm_fit))
        return ()
    def get_lmfit_names(self) -> Sequence[str]:
        """Get the slot names of the lm_fit object for access and conversion."""
        r = get_r_environment()
        return list(r.ro.baseenv["names"](self.lm_fit))
    def get_coefficients(self) -> pd.DataFrame:
        """Extract coefficients as a pandas DataFrame from the limma lm_fit object."""
        assert self.lm_fit is not None, "lm_fit must be set to extract coefficients"
        _r = get_r_environment()
        if self.lm_fit is not None:
            coefs_r = self.lm_fit.rx2("coefficients")
            
        return RConverters.rmatrix_to_pandas(coefs_r)
    
    def e_bayes(self) -> LimmaModel:
        """Run eBayes on the lm_fit object and store the result."""
        assert self.lm_fit is not None, "lm_fit must be set to run eBayes"
        _r = get_r_environment()
        limma_pkg = _limma()
        eb = limma_pkg.eBayes(self.lm_fit)
        return replace(
            self,
            ebayes = eb
        )


def lm_fit(
    se: RESummarizedExperiment,
    design: pd.DataFrame,
    ndups: int | None = None,
    method: Literal["ls", "robust"] = "ls",  # or "robust" etc.
    return_result_object: bool = False,
    **kwargs
) -> RESummarizedExperiment:
    """Run `limma::lmFit` on an expression assay and store the fit on the object.

    Args:
        se: Input `RESummarizedExperiment` or `Limma` instance containing a
            `"log_expr"` assay (by convention) and optionally a `"weights"` assay.
        design: Design matrix (samples × covariates) as a pandas DataFrame.
        ndups: Number of technical replicates per unique sample (or `None`).
        method: Fitting method, e.g., `"ls"` (least squares) or `"robust"`.
        **kwargs: Additional keyword arguments forwarded to `limma::lmFit`.

    Returns:
        LimmaModel: An instance of `LimmaModel` with the `lm_fit` R object set.

    Raises:
        AssertionError: If `ndups` is provided but not an integer.
        TypeError: If `se` is not `RESummarizedExperiment` or `Limma`.
    """
    
    _r = _rmana
    limma_pkg = _limma()

    lmres = LimmaModel(method=method, sample_names=se.column_names, feature_names=se.row_names)

    exprs_r = se.assay_r("log_expr")  # or whichever assay
    design_r = _df_to_r_matrix(design)
    lmres.design = design

    if "weights" in se.assay_names:
        weights = se.assay_r("weights")
    else:
        weights = _r.ro.NULL

    if ndups is None:
        ndups = _r.ro.NULL
    else:
        assert isinstance(ndups, int), "ndups must be an integer or None"

    fit = limma_pkg.lmFit(exprs_r, design_r, weights = weights, method = method, **kwargs)
    lmres.lm_fit = fit

    return lmres


# ——— 6) contrasts.fit.default & eBayes.default & topTable.default — similar pattern ———
# You can call contrasts.fit_default(), then eBayes_default(), then topTable_default()
# and store their returned matrices/data.frames into assays or metadata columns.

def contrasts_fit(
    lm_obj: LimmaModel,
    contrast: Sequence[int | float],
):
    """Apply `limma::contrasts.fit` to an existing `lm_fit` in a `Limma` object.

    Args:
        se: A `Limma` instance with `lm_fit` set.
        contrast: 1D array-like contrast vector (length = number of columns in the design).
            Can be created using formulaic_contrasts or manually.

    Returns:
        LimmaModel: A new `LimmaModel` instance with the `contrast_fit` R object set.

    Raises:
        AssertionError: If `se` is not `Limma`, or `lm_fit` is missing, or
            `contrast` is not 1D.
    """

    assert isinstance(lm_obj, LimmaModel), "se must be a Limma instance"
    assert lm_obj.lm_fit is not None, "lm_fit must be set in the Limma instance"
    assert isinstance(contrast, (list, np.ndarray, pd.Series)), "contrast must be a list or numpy array"

    _r = _rmana
    limma_pkg = _limma()
    # convert contrast vector to R
    contrast = np.asarray(contrast, dtype=float)
    assert contrast.ndim == 1, "Contrast must be a 1D array-like"
    contrast_r = _r.ro.FloatVector(contrast)
    
    # apply contrasts.fit
    fit_r = limma_pkg.contrasts_fit(lm_obj.lm_fit, contrast=contrast_r)
    
    # return LimmaModel object with contrasts fit set
    return replace(
        lm_obj,
        contrast_fit = fit_r
    )

def top_table(
    lm_obj: LimmaModel,
    coef: str|int|None = None,
    n: int | None = None,
    adjust_method: str = "BH",
    sort_by: str = "PValue",
    **kwargs
) -> pd.DataFrame:
    """Extract top results using `limma::topTable` after `eBayes` on a fit.

    Args:
        se: A `Limma` instance with either `contrast_fit` or `lm_fit` set.
        coef: Either the name of the coefficient in the design matrix, or its index. 
            Indexing starts at 1 (1-based indexing), due to R's conventions.
        n: Number of top rows to return. If `None`, defaults to the number of rows
            in the object (i.e., all features).
        adjust_method: Multiple testing method (e.g., `"BH"`).
        sort_by: Sorting option passed to `topTable` (e.g., `"PValue"`, `"none"`).
        **kwargs: Additional keyword arguments forwarded to `topTable`.

    Returns:
        pandas.DataFrame: A DataFrame of top-ranked features with limma columns
        converted to `p_value`, `log_fc`, and `adj_p_value`, plus a `gene` index.

    Raises:
        AssertionError: If neither `contrast_fit` nor `lm_fit` is set.
    """
    """
    Apply topTable.default to the contrast_fit object in a Limma instance.
    Returns a pandas DataFrame with the top results.
    """
    assert isinstance(lm_obj, LimmaModel), "se must be a Limma instance"
    # assert se.contrast_fit is not None, "contrast_fit must be set in the Limma instance"

    r_fit = lm_obj.contrast_fit if lm_obj.contrast_fit is not None else lm_obj.lm_fit
    assert r_fit is not None, "lm_fit or contrast_fit must be set in the Limma instance"

    _r = _rmana
    limma_pkg = _limma()

    eb = limma_pkg.eBayes(r_fit)

    if n is None:
        
        n = int(_r.r2py(_r.ro.baseenv["nrow"](r_fit)))
    
    if coef is None:
        # default to the first coefficient (1-based index)
        coef = _r.ro.NULL
    
    # call topTable.default
    top_r = limma_pkg.topTable(eb,coef = coef, n=n, adjust_method=adjust_method, sort_by="none", **kwargs)
    
    # convert to pandas DataFrame
    with _r.localconverter(_r.default_converter + _r.pandas2ri.converter):
        df = _r.get_conversion().rpy2py(top_r)
    
    return df.reset_index(names="gene").rename(columns={'P.Value':'p_value', 'logFC':'log_fc', 'adj.P.Val':'adj_p_value'})

__all__ = [n for n in dir() if not n.startswith("_")]