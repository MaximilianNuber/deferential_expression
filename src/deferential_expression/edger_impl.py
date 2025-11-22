from __future__ import annotations
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd

from biocframe import BiocFrame
# from deferential_expression.rpy2_manager import Rpy2Manager
from functools import lru_cache

from pyrtools.lazy_r_env import get_r_environment, r
from pyrtools.r_converters import RConverters
from .resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter, _df_to_r_matrix, _df_to_r_df
from dataclasses import dataclass



@lru_cache(maxsize = 1)
def _prep_edger():
    """Lazily prepare the edgeR runtime.

    Returns:
        Tuple[Any, Any]: A tuple ``(r_env, edgeR_pkg)`` where
        ``r_env`` is the lazy rpy2 environment from ``pyrtools.lazy_r_env.r``,
        and ``edgeR_pkg`` is the imported R ``edgeR`` package (lazy import).

    Notes:
        The result is cached (LRU) to avoid repeated imports.
    """
    # r = Rpy2Manager()
    edger_pkg = r.lazy_import_r_packages("edgeR")

    return r, edger_pkg


from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


from dataclasses import replace

# ------------------------------------------------------------------
# R manager & converters (adapt to your imports)
# ------------------------------------------------------------------
_rmana = r
_r = get_r_environment()

def edgeR_pkg():
    """Import and return the R ``edgeR`` package via the global rpy2 manager.

    Returns:
        Any: The rpy2 handle to the R ``edgeR`` package.
    """
    # use cached_property on manager instead if you like
    return _rmana.importr("edgeR")



# ------------------------------------------------------------------
# EdgeR class: stores R fit objects + fluent ops returning new EdgeR
# ------------------------------------------------------------------
class EdgeR(RESummarizedExperiment):
    """``RESummarizedExperiment`` subclass adding slots for edgeR model state.

    Attributes:
        dge: R ``DGEList`` object (after construction/normalization).
        disp: R object holding dispersion estimates (e.g., from ``estimateDisp``).
        glm: R GLM fit object from ``edgeR::glmQLFit``.
        lrt: R test object from ``edgeR::glmQLFTest``.
    """
    __slots__ = ("dge", "disp", "glm", "lrt")  # add others as you need

    def __init__(self, *, assays = None, row_data=None, column_data=None,
                 row_names=None, column_names=None, metadata=None, **kwargs):
        """Initialize an ``EdgeR`` container with standard SE fields.

        Args:
            assays: Mapping of assay names to arrays or R-backed matrices.
            row_data: Optional feature annotations.
            column_data: Optional sample annotations.
            row_names: Optional feature names.
            column_names: Optional sample names.
            metadata: Optional free-form metadata dictionary.
            **kwargs: Forwarded to ``RESummarizedExperiment``.
        """
        # initialize R object slots
        self.dge = None
        self.disp = None
        self.glm = None
        self.lrt = None
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
               dge=None,
               disp=None,
               glm=None,
               lrt=None):
        """Clone the object with optional replacements and updated edgeR slots.

        Args:
            assays: Optional replacement assays mapping.
            row_data: Optional replacement row annotations.
            column_data: Optional replacement column annotations.
            metadata: Optional replacement metadata dictionary.
            dge: Optional replacement DGE object.
            disp: Optional replacement dispersion object.
            glm: Optional replacement GLM fit.
            lrt: Optional replacement test object.

        Returns:
            EdgeR: A new ``EdgeR`` with fields replaced as requested.
        """
        return EdgeR(
            assays=assays if assays is not None else dict(self.assays),
            row_data=row_data if row_data is not None else self.row_data,
            column_data=column_data if column_data is not None else self.column_data,
            row_names=self.row_names,
            column_names=self.column_names,
            metadata=metadata if metadata is not None else dict(self.metadata)
        )._set_r_objs(dge if dge is not None else self.dge,
                      disp if disp is not None else self.disp,
                      glm if glm is not None else self.glm,
                      lrt if lrt is not None else self.lrt)

    def _set_r_objs(self, dge, disp, glm, lrt):
        """Set edgeR-related R objects on the instance (fluent).

        Args:
            dge: R ``DGEList`` object.
            disp: R dispersion fit object.
            glm: R GLM fit object from ``glmQLFit``.
            lrt: R test object from ``glmQLFTest``.

        Returns:
            EdgeR: The same instance for chaining.
        """
        self.dge, self.disp, self.glm, self.lrt = dge, disp, glm, lrt
        return self

    @property
    def samples(self) -> pd.DataFrame:
        """pandas.DataFrame: Column (sample) annotations as pandas."""
        return self.column_data.to_pandas()

    # ------------------- edgeR ops -------------------

    def filter_by_expr(self,
                       design: Optional[pd.DataFrame] = None,
                       assay: str = "counts",
                       **kwargs) -> Tuple["EdgeR", np.ndarray]:
        """Compute an expression filter mask using ``edgeR::filterByExpr``.

        Args:
            design: Optional design matrix (samples × covariates) as pandas.
            assay: Name of the counts assay to evaluate.
            **kwargs: Additional args forwarded to ``filterByExpr``.

        Returns:
            np.ndarray: Boolean mask of rows to keep.

        Notes:
            The code currently returns only the boolean mask. Subsetting and
            DGE updates below are unreachable due to the early ``return``.
        """
        edger = edgeR_pkg()
        counts_r = self.assay_r(assay)
        # samples_r = _df_to_r_df(self.column_data.to_pandas() or pd.DataFrame(index=np.arange(counts_r.ncol)))
        # dge = edger.DGEList(counts=counts_r, samples=samples_r)

        design_r = _r.ro.NULL if design is None else _df_to_r_matrix(design)
        mask_r = edger.filterByExpr(counts_r, design=design_r, **kwargs)
        mask = np.array(mask_r, dtype=bool)

        cd = self.column_data

        return mask

        # subset all assays row-wise via BaseSE slicing (mask length = rows)
        new_se = self[mask, :]
        # keep same R objects? We built a temp dge above; update dge to filtered
        bracket = _r.ro.baseenv["["]
        dge_filtered = edger.DGEList(counts=bracket(counts_r, mask_r, _r.ro.NULL),
                                     samples=samples_r)
        return new_se._clone(dge=dge_filtered), mask

    def calc_norm_factors(self,
                          assay: str = "counts",
                          **kwargs) -> "EdgeR":
        """Compute TMM normalization factors and store them in column data.

        Args:
            assay: Name of the counts assay.
            **kwargs: Additional args forwarded to ``edgeR::calcNormFactors``
                when called on a ``DGEList``.

        Returns:
            EdgeR: A cloned object with updated ``column_data`` containing
            ``edgeR_norm_factors`` and with ``dge`` set to the updated DGEList.
        """
        edger = edgeR_pkg()
        counts_r = self.assay_r(assay)
        samples_r = _df_to_r_df(self.column_data_df() or pd.DataFrame(index=np.arange(counts_r.ncol)))
        dge = edger.DGEList(counts=counts_r, samples=samples_r)
        dge = edger.calcNormFactors(dge, **kwargs)

        # pull norm factors
        factors = np.array(dge.rx2("samples").rx2("norm.factors"))
        col_df = self.column_data_df().copy() if self.column_data_df() is not None else pd.DataFrame(index=list(_r.ro.baseenv["colnames"](counts_r)))
        col_df["edgeR_norm_factors"] = factors
        return self._clone(column_data=col_df, dge=dge)

    def cpm(self,
            assay: str = "counts",
            log: bool = True,
            prior_count: float = 1.0,
            out_name: str = "cpm",
            **kwargs) -> "EdgeR":
        """Compute CPM (optionally log-CPM) using ``edgeR::cpm`` and add as assay.

        Args:
            assay: Source assay name (counts).
            log: Whether to compute log-CPM.
            prior_count: Prior count for log-CPM.
            out_name: Name for the output assay.
            **kwargs: Additional arguments forwarded to ``edgeR::cpm``.

        Returns:
            EdgeR: A cloned object with the new CPM assay (R-backed).
        """
        edger = edgeR_pkg()
        counts_r = self.assay_r(assay)
        cpm_r = edger.cpm(counts_r, log=log, prior_count=prior_count, **kwargs)
        # wrap as R matrix adapter and drop straight into assays
        assays = dict(self.assays)
        assays[out_name] = RMatrixAdapter(cpm_r, _r)
        return self._clone(assays=assays)

    def estimate_disp(self,
                      design: pd.DataFrame,
                      assay: str = "counts",
                      **kwargs) -> "EdgeR":
        """Estimate dispersion using ``edgeR::estimateDisp``.

        Args:
            design: Design matrix (samples × covariates) as pandas.
            assay: Counts assay name.
            **kwargs: Additional arguments forwarded to ``estimateDisp``.

        Returns:
            EdgeR: A cloned object with ``dge`` and ``disp`` set.
        """
        edger = edgeR_pkg()
        counts_r = self.assay_r(assay)
        samples_r = _df_to_r_df(self.col_data_df() or pd.DataFrame(index=np.arange(counts_r.ncol)))
        dge = edger.DGEList(counts=counts_r, samples=samples_r)
        dge = edger.calcNormFactors(dge)  # ensure norm factors
        design_r = _df_to_r_matrix(design)
        disp = edger.estimateDisp(dge, design=design_r, **kwargs)
        return self._clone(dge=dge, disp=disp)

    def glm_ql_fit(self,
                   design: Optional[pd.DataFrame] = None,
                   **kwargs) -> "EdgeR":
        """Fit the quasi-likelihood GLM via ``edgeR::glmQLFit``.

        Args:
            design: Optional design matrix. If ``disp`` is missing, this is
                required to compute it on-the-fly via ``estimate_disp``.
            **kwargs: Additional args forwarded to ``glmQLFit``.

        Returns:
            EdgeR: A cloned object with ``glm`` set.

        Raises:
            ValueError: If ``disp`` is not present and no ``design`` is provided.
        """
        if self.disp is None:
            if design is None:
                raise ValueError("No dispersion in object and no design provided.")
            # fallback: compute disp on the fly
            tmp = self.estimate_disp(design, **kwargs)
            return tmp.glm_ql_fit(design=None, **kwargs)

        edger = edgeR_pkg()
        glm = edger.glmQLFit(self.disp, design=self.disp.rx2("design"), **kwargs)
        return self._clone(glm=glm)

    def glm_qlf_test(self,
                     contrast: Sequence[float] | None = None,
                     coef: Optional[Union[int, str]] = None,
                     **kwargs) -> "EdgeR":
        """Run ``edgeR::glmQLFTest`` on the stored GLM fit.

        Args:
            contrast: Optional numeric contrast vector.
            coef: Optional coefficient index/name to test (alternative to ``contrast``).
            **kwargs: Additional args forwarded to ``glmQLFTest``.

        Returns:
            EdgeR: A cloned object with ``lrt`` set.

        Raises:
            ValueError: If neither ``contrast`` nor ``coef`` is provided.
            ValueError: If ``glm`` has not been fitted yet.
        """
        if self.glm is None:
            raise ValueError("Run glm_ql_fit first.")
        edger = edgeR_pkg()
        if contrast is not None:
            with _r.localconverter(_r.default_converter + _r.numpy2ri.converter):
                contrast_r = _r.get_conversion().py2rpy(np.asarray(contrast, dtype=float))
            lrt = edger.glmQLFTest(self.glm, contrast=contrast_r, **kwargs)
        else:
            # coef-based test
            if coef is None:
                raise ValueError("Provide either `contrast` or `coef`.")
            lrt = edger.glmQLFTest(self.glm, coef=coef, **kwargs)
        return self._clone(lrt=lrt)

    def top_tags(self,
                 n: Optional[int] = None,
                 adjust_method: str = "BH",
                 sort_by: str = "PValue",
                 **kwargs) -> Tuple["EdgeR", pd.DataFrame]:
        """Extract top results using ``edgeR::topTags`` and return a pandas table.

        Args:
            n: Number of rows to return; if ``None``, uses ``Inf``.
            adjust_method: Multiple-testing method (e.g., ``"BH"``).
            sort_by: Sorting key (e.g., ``"PValue"``).
            **kwargs: Additional args forwarded to ``topTags``.

        Returns:
            Tuple[EdgeR, pandas.DataFrame]: A cloned object (with metadata note)
            and a DataFrame with columns standardized to ``gene``, ``p_value``,
            ``log_fc``, and ``adj_p_value``.

        Raises:
            ValueError: If ``glm_qlf_test`` has not been run.
        """
        if self.lrt is None:
            raise ValueError("Run glm_qlf_test first.")
        edger = edgeR_pkg()
        n_val = _r.ro.r("Inf") if n is None else int(n)
        tt = edger.topTags(self.lrt, n=n_val, adjust_method=adjust_method, sort_by=sort_by, **kwargs)
        # extract table
        table_r = tt.rx2("table") if "table" in list(tt.names) else _r.ro.baseenv["as.data.frame"](tt)
        with _r.localconverter(_r.default_converter + _r.pandas2ri.converter):
            df = _r.get_conversion().rpy2py(table_r)
        df = df.reset_index().rename(columns={
            "index": "gene",
            "PValue": "p_value",
            "logFC": "log_fc",
            "FDR": "adj_p_value"
        })
        # store R table? you can put into metadata if desired
        meta = dict(self.metadata)
        meta["edgeR_topTags_last"] = {"n": n, "adjust_method": adjust_method, "sort_by": sort_by}
        return self._clone(metadata=meta), df
    
def _filter_by_expr_impl(
    rmat, 
    group = None, 
    design = None, 
    lib_size = None,
    min_count = 10,
    min_total_count = 15,
    large_n = 10, 
    min_prop = 0.7,
    **kwargs
):
    pkg = edgeR_pkg()
    r = get_r_environment()
    group = r.ro.NULL if group is None else group
    design = r.ro.NULL if design is None else design
    lib_size = r.ro.NULL if lib_size is None else lib_size

    return pkg.filterByExpr(
        rmat, 
        group = group, 
        design = design, 
        lib_size = lib_size,
        min_count = min_count,
        min_total_count = min_total_count,
        large_n = large_n,
        min_prop = min_prop,
        **kwargs
        )
    

def filter_by_expr(obj: RESummarizedExperiment, group: Sequence[str] | None = None, design: pd.DataFrame | None = None, 
                   assay = "counts", **kwargs):
    """Functional wrapper for ``edgeR::filterByExpr`` on an ``EdgeR`` container.

    Args:
        obj: ``EdgeR`` instance holding at least the counts assay.
        group: Optional group labels (factor) for filtering.
        design: Optional design matrix as pandas DataFrame.
        assay: Assay name containing counts.
        **kwargs: Additional args forwarded to ``filterByExpr``.

    Returns:
        np.ndarray: Boolean mask indicating rows to retain.
    """
    pkg = edgeR_pkg()
    r = _rmana
    if group is not None:
        group = r.StrVector(group)
    else:
        group = r.ro.NULL
        
    if design is not None:
        design = RConverters.pandas_to_r_matrix(design)
    else:
        design = r.ro.NULL

    # mask_r = pkg.filterByExpr(obj.assay_r(assay), group = group, design = design, **kwargs)
    mask_r = _filter_by_expr_impl(
        obj.assay_r(assay), 
        group = group, 
        design = design, 
        **kwargs
    )
    mask = np.asarray(mask_r)
    return mask.astype(bool)

def _calc_norm_factors_impl(
    rmat, 
    method = "TMM", 
    refColumn=None,
    logratioTrim=.3,
    sumTrim=0.05,
    doWeighting=True,
    Acutoff=-1e10, 
    p=0.75,
    **kwargs
):
    pkg = edgeR_pkg()
    r = get_r_environment()
    refColumn = r.ro.NULL if refColumn is None else refColumn
    return pkg.calcNormFactors(
        rmat, 
        method = method, 
        refColumn=refColumn,
        logratioTrim=logratioTrim,
        sumTrim=sumTrim,
        doWeighting=doWeighting,
        Acutoff=Acutoff, 
        p=p,
        **kwargs
    )

def calc_norm_factors(
    obj: EdgeR, 
    assay = "counts", 
    method = "TMM", 
    refColumn=None,
    logratioTrim=.3,
    sumTrim=0.05,
    doWeighting=True,
    Acutoff=-1e10, 
    p=0.75,
    **kwargs
):
    """Functional TMM normalization: compute and store norm factors in colData.

    Args:
        obj: ``EdgeR`` instance.
        assay: Counts assay name.
        **kwargs: Additional args forwarded to ``edgeR::calcNormFactors``.

    Returns:
        EdgeR: New object with updated column data containing ``norm.factors``.
    """
    pkg = edgeR_pkg()
    r = _rmana

    rmat = obj.assay_r(assay)

    # r_factors = pkg.calcNormFactors(rmat, **kwargs)
    r_factors = _calc_norm_factors_impl(
        rmat, 
        method = method, 
        refColumn=refColumn,
        logratioTrim=logratioTrim,
        sumTrim=sumTrim,
        doWeighting=doWeighting,
        Acutoff=Acutoff, 
        p=p,
        **kwargs
        )
    norm_factors = np.asarray(r_factors)

    # new_cols = BiocFrame({"norm.factors": norm_factors})
    coldata = obj.get_column_data()

    import biocutils as ut

    # new_cols = ut.combine_columns(coldata, new_cols)
    new_cols = coldata.set_column("norm.factors", norm_factors)

    return obj.set_column_data(new_cols)


def cpm(obj: EdgeR, assay: str = "counts", **kwargs):
    """Functional CPM computation using ``edgeR::cpm`` and assay insertion.

    Args:
        obj: ``EdgeR`` instance.
        assay: Counts assay name.
        **kwargs: Additional args forwarded to ``edgeR::cpm``.

    Returns:
        EdgeR: New object with a ``"cpm"`` assay added (R-backed).
    """
    r, pkg = _prep_edger()

    rmat = obj.assay_r(assay)

    cpm_mat = pkg.cpm(rmat, **kwargs)
    cpm_mat = RMatrixAdapter(cpm_mat, r)

    return obj.set_assay(name = "cpm", assay = cpm_mat)

@dataclass
class EdgeRModel:
    """Container for limma results including fit, coefficients, and metadata."""
    sample_names: Optional[Sequence[str]] = None  # Sample names (column names)
    feature_names: Optional[Sequence[str]] = None  # Feature names (row names)
    fit: Optional[Any] = None  # R object from lmFit
    fit_config: Optional[Any] = None  # Configuration used for fitting

    design: Optional[pd.DataFrame] = None  # Design matrix used for fitting

    coefficients: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None



def _glm_ql_fit_impl(
    rmat,
    design_r,
    dispersion_r,
    offset,
    weights,
    legacy,
    top_proportion,
    **user_kwargs
):
    pkg = edgeR_pkg()
    return pkg.glmQLFit_default(
        rmat, 
        design = design_r,
        dispersion = dispersion_r,
        offset = offset,
        weights = weights, 
        legacy = legacy,
        top_proportion = top_proportion,
        **user_kwargs
    )

@dataclass
class GlmQlFitConfig:
    dispersion: Optional[Union[pd.DataFrame, np.ndarray]] = None
    offset: Optional[Union[pd.DataFrame, np.ndarray]] = None
    lib_size: Optional[Sequence] = None
    weights: Optional[Union[int, float, Sequence[float], pd.DataFrame, np.ndarray]] = None
    legacy: bool = False
    top_proportion: float = 0.1
    assay: str = "counts"
    user_kwargs: Dict[str, Any] = None

def glm_ql_fit(
    obj: RESummarizedExperiment, 
    design: pd.DataFrame, 
    *,
    dispersion: pd.DataFrame | np.ndarray | None = None, 
    offset: pd.DataFrame | np.ndarray | None = None,
    lib_size: Sequence | None = None,
    weights: int | float | Sequence[float] | pd.DataFrame | np.ndarray | None = None,
    legacy: bool = False,
    top_proportion: 'float' = 0.1,
    assay: 'str' = 'counts',
    **user_kwargs
):
    """Functional ``edgeR::glmQLFit`` with optional dispersion/offset/weights.

    Args:
        obj: ``EdgeR`` instance with a counts assay.
        design: Design matrix (samples × covariates) as pandas DataFrame.
        dispersion: Optional per-observation dispersion values/matrix.
        offset: Optional offset matrix (e.g., log library sizes).
        lib_size: Optional library sizes per sample.
        weights: Optional observation/sample weights.
        legacy: Pass-through boolean to the underlying R function (if supported).
        top_proportion: Pass-through numeric parameter for robust fitting.
        assay: Counts assay name.
        **user_kwargs: Additional args forwarded to the R implementation.

    Returns:
        EdgeR: New object with ``glm`` fit stored.

    Notes:
        Inputs that are pandas/NumPy are converted to R matrices/vectors as needed.
    """
    
    config = GlmQlFitConfig(
        dispersion = dispersion,
        offset = offset,
        lib_size = lib_size,
        weights = weights,
        legacy = legacy,
        top_proportion = top_proportion,
        assay = assay,
        user_kwargs = user_kwargs
    )
    

    r, pkg = _prep_edger()
    rmat = obj.assay_r(assay)

    design_r = RConverters.pandas_to_r_matrix(design)

    if dispersion is not None:
        if isinstance(dispersion, pd.DataFrame):
            dispersion = RConverters.pandas_to_r_matrix(dispersion)
        elif isinstance(dispersion, np.ndarray):
            dispersion = RConverters.numpy_to_r_matrix(dispersion)
    else:
        dispersion = r.ro.NULL

    if offset is not None:
        if isinstance(offset, pd.DataFrame):
            offset = RConverters.pandas_to_r_matrix(offset)
        elif isinstance(offset, np.ndarray):
            offset = RConverters.numpy_to_r_matrix(offset)
    else:
        if "norm.factors" in obj.column_data.column_names:
            # use norm factors as offset if available
            offset = obj.column_data["norm.factors"]
            offset = np.asarray(offset, dtype=float)
            offset = r.FloatVector(offset)
        else:
            offset = r.ro.NULL

    if lib_size is not None:
        lib_size = r.FloatVector(np.asarray(lib_size, dtype = float))
    else: 
        lib_size = r.ro.NULL

    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            weights = RConverters.pandas_to_r_matrix(weights)
        elif isinstance(weights, np.ndarray):
            weights = RConverters.numpy_to_r_matrix(weights)
        elif isinstance(weights, int):
            weights = r.ro.IntVector([weights])
        elif isinstance(weights, float):
            weights = r.ro.FloatVector([weights])
        elif isinstance(weights, Sequence):
            weights = r.ro.FloatVector(
                np.asarray(Sequence, dtype = float)
            )
    else:
        weights = r.ro.NULL


    # fit_obj = pkg.glmQLFit_default(
    #     rmat, 
    #     design = design_r,
    #     dispersion = dispersion,
    #     offset = offset,
    #     weights = weights, 
    #     legacy = legacy,
    #     top_proportion = top_proportion,
    #     **user_kwargs
    # )
    fit_obj = _glm_ql_fit_impl(
        rmat,
        design_r,
        dispersion,
        offset,
        weights,
        legacy,
        top_proportion,
        **user_kwargs
    )
    
    # return obj._clone(glm = fit_obj)
    return EdgeRModel(
        sample_names = obj.column_names,
        feature_names = obj.row_names,
        fit = fit_obj,
        fit_config = config,
        design = design,
        
    )


def glm_ql_ftest(obj: EdgeRModel, coef: str | None = None, contrast: Sequence | None = None, poisson_bound: bool = True,
                adjust_method = "BH"):
    """Functional quasi-likelihood F-test and table extraction via ``topTags``.

    Args:
        obj: ``EdgeR`` instance with a fitted ``glm``.
        coef: Optional coefficient name to test.
        contrast: Optional contrast vector.
        poisson_bound: Whether to apply the Poisson bound in the test.
        adjust_method: Multiple-testing method for the returned table.

    Returns:
        pandas.DataFrame: Results table with all rows (``n = Inf`` in R). Columns
        follow edgeR’s defaults.

    Raises:
        AssertionError: If ``glm`` has not been set on ``obj``.
    """
    assert hasattr(obj, "fit")
    assert obj.fit is not None

    r, pkg = _prep_edger()
    if coef is not None:
        coef = r.StrVector([coef])
    else:
        coef = r.ro.NULL
    if contrast is not None:
        contrast = np.asarray(contrast, dtype = int)
        contrast = r.IntVector(contrast)
    else:
        contrast = r.ro.NULL
    poisson_bound = r.BoolVector([poisson_bound])

    res = pkg.glmQLFTest(obj.fit, coef = coef, contrast = contrast, poisson_bound = poisson_bound)
    # topTags(object, n=10, adjust.method="BH", sort.by="PValue", p.value=1)

    res = pkg.topTags(
        res,
        n = r.ro.r("Inf"),
        adjust_method = adjust_method,
        sort_by = r.ro.NULL,
        p_value = r.IntVector([1])
    )

    res = r.ro.baseenv["as.data.frame"](res)
    with r.localconverter(
        r.default_converter + r.pandas2ri.converter
    ):
        res = r.get_conversion().rpy2py(res)

    return res

# __all__ = [n for n in dir() if not n.startswith("_")]
__all__ = [
    "EdgeRModel", 
    "GlmQlFitConfig", 
    "filter_by_expr", 
    "calc_norm_factors", 
    "cpm", 
    "glm_ql_fit", 
    "glm_ql_ftest"
    ]