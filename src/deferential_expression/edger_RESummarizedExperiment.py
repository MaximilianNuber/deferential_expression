from __future__ import annotations
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd

from biocframe import BiocFrame
# from deferential_expression.rpy2_manager import Rpy2Manager
from functools import lru_cache

from deferential_expression.edger_func import _RConverters
from pyrtools.lazy_r_env import get_r_environment, r


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
    from deferential_expression.rpy2_manager import Rpy2Manager
    # r = Rpy2Manager()
    edger_pkg = r.lazy_import_r_packages("edgeR")

    return r, edger_pkg


from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from summarizedexperiment import SummarizedExperiment
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

class RMatrixAdapter:
    """Thin wrapper around an rpy2 R matrix to preserve R backing.

    This adapter:
      * Stores a reference to the underlying R matrix (``rmat``).
      * Exposes ``shape`` without converting to NumPy.
      * Implements ``__getitem__`` to slice **in R** and return another
        ``RMatrixAdapter`` (prevents accidental NumPy materialization).
      * Provides ``to_numpy()`` for explicit conversion and ``__array__`` to
        allow NumPy coercion when needed.

    Attributes:
        _rmat: The underlying rpy2 SEXP matrix.
        _shape: Tuple ``(n_rows, n_cols)`` inferred from R ``dim``.
        _r: The rpy2 manager/environment used for conversions.
    """
    __slots__ = ("_rmat", "_shape", "_r")
    def __init__(self, rmat: Any, r_manager: Rpy2Manager):
        """Initialize the adapter.

        Args:
            rmat: An R matrix SEXP object (rpy2).
            r_manager: rpy2 manager/environment providing ``ro`` and converters.

        Notes:
            ``shape`` is computed from the R-level ``dim(rmat)`` at init time.
        """
        self._rmat = rmat
        self._r = _rmana
        dims = r_manager.ro.baseenv["dim"](rmat)
        self._shape = (int(dims[0]), int(dims[1]))
    @property
    def shape(self): 
        """Tuple[int, int]: Matrix shape (rows, cols) without conversion."""
        return self._shape
    @property
    def rmat(self):  
        """Any: Underlying rpy2 SEXP matrix (read-only reference)."""
        return self._rmat
    def to_numpy(self):
        """Convert the wrapped R matrix to a NumPy array.

        Returns:
            np.ndarray: Dense NumPy array with the same values as the R matrix.
        """
        with self._r.localconverter(self._r.default_converter + self._r.numpy2ri.converter):
            return self._r.get_conversion().rpy2py(self._rmat)
    # def __getitem__(self, idx):
    #     return self.to_numpy()[idx]
    # CRITICAL FIX: keep results as an RMatrixAdapter, not a NumPy array
    def __getitem__(self, key):
        """Slice the R matrix in R and return another ``RMatrixAdapter``.

        Args:
            key: Either ``rows`` or ``(rows, cols)`` using Python indexing
                (ints, slices, boolean masks, or integer arrays).

        Returns:
            RMatrixAdapter: Adapter wrapping the sliced R matrix.

        Raises:
            IndexError: If more than two indices are provided.
        """
        if not isinstance(key, tuple):
            rows, cols = key, slice(None)
        else:
            if len(key) != 2:
                raise IndexError("Use 2D indexing: [rows, cols].")
            rows, cols = key
        return _slice_rmat(self, rows, cols)

    def __array__(self, dtype=None):
        """Support implicit NumPy coercion (e.g., ``np.asarray(adapter)``).

        Args:
            dtype: Optional dtype to cast to.

        Returns:
            np.ndarray: The converted NumPy array (possibly view-cast).
        """
        arr = self.to_numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

def _df_to_r_matrix(df: pd.DataFrame) -> Any:
    """Convert a pandas DataFrame to an R matrix, preserving row/col names.

    Args:
        df: DataFrame with numeric values; index/columns become dimnames.

    Returns:
        Any: rpy2 SEXP matrix with ``rownames`` and ``colnames`` set.
    """
    with _r.localconverter(_r.default_converter + _r.numpy2ri.converter):
        r_mat = _r.get_conversion().py2rpy(df.values)
    r_mat = _r.ro.baseenv["rownames<-"](r_mat, _r.StrVector(df.index.astype(str).to_numpy()))
    r_mat = _r.ro.baseenv["colnames<-"](r_mat, _r.StrVector(df.columns.astype(str).to_numpy()))
    return r_mat

def _df_to_r_df(df: pd.DataFrame) -> Any:
    """Convert a pandas DataFrame to an R ``data.frame``.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Any: rpy2 SEXP data.frame object.
    """
    with _r.localconverter(_r.default_converter + _r.pandas2ri_converter):
        return _r.get_conversion().py2rpy(df)

# ------------------------------------------------------------------
# RESummarizedExperiment
# ------------------------------------------------------------------

def _to_r_index(idx, n, *, r=_r):
    """Build an R index (logical/integer) from a Python index for 1-based R.

    Args:
        idx: Python index (slice, int, sequence/array, or boolean mask).
        n: Length along the dimension being indexed.
        r: rpy2 environment providing vector constructors.

    Returns:
        Any: R index vector suitable for subsetting (logical or integer).

    Notes:
        * Slices are translated to 1-based sequences.
        * Boolean arrays are converted to R logical vectors.
        * Integer arrays are shifted by +1 for 1-based R indexing.
        * If ``idx`` is None/unsupported, returns ``1:n`` (all).
    """
    if isinstance(idx, slice):
        start, stop, step = idx.indices(n)
        return r.IntVector(list(range(start + 1, stop + 1, step)))  # R is 1-based
    if isinstance(idx, (list, np.ndarray, pd.Index)):
        idx = np.asarray(idx)
        if idx.dtype == bool:
            return r.BoolVector(idx.tolist())
        return r.IntVector((idx + 1).tolist())
    if isinstance(idx, int):
        return r.IntVector([idx + 1])
    return r.IntVector(list(range(1, n + 1)))  # fallback: ':'
    

def _slice_rmat(adapter: "RMatrixAdapter", rows, cols):
    """Slice an R-backed matrix using R subsetting and wrap the result.

    Args:
        adapter: The source ``RMatrixAdapter``.
        rows: Row indexer (slice/int/array/bool mask).
        cols: Column indexer (slice/int/array/bool mask).

    Returns:
        RMatrixAdapter: New adapter wrapping the sliced R matrix.
    """
    r = _r
    rm = adapter.rmat
    nrow = adapter.shape[0]
    ncol = adapter.shape[1]
    ridx = _to_r_index(rows, nrow, r=r)
    cidx = _to_r_index(cols, ncol, r=r)
    bracket = r.ro.baseenv["["]
    out = bracket(rm, ridx, cidx)
    return RMatrixAdapter(out, r)

class RESummarizedExperiment(SummarizedExperiment):
    """
    Drop-in subclass that auto-wraps R matrices (rpy2) with RMatrixAdapter,
    but keeps ctor-compatible signature so slicing works.
    """
    def __init__(
        self,
        *,
        assays: Optional[Dict[str, Any]] = None,
        row_data: Optional[pd.DataFrame] = None,
        column_data: Optional[pd.DataFrame] = None,
        row_names: Optional[Sequence[str]] = None,
        column_names: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize an ``RESummarizedExperiment`` with optional R-backed assays.

        Args:
            assays: Mapping from assay name to matrix-like object. R matrices
                may be provided directly and will be wrapped into ``RMatrixAdapter``.
            row_data: Optional feature annotations as a pandas DataFrame.
            column_data: Optional sample annotations as a pandas DataFrame.
            row_names: Optional feature names.
            column_names: Optional sample names.
            metadata: Optional dictionary of free-form metadata.
            **kwargs: Forwarded to base ``SummarizedExperiment`` constructor.

        Raises:
            AssertionError: If ``assays`` is provided but not a ``dict``.
        """
        assert isinstance(assays, (dict, type(None))), "Assays must be a dictionary of RMatrixAdapter or numpy arrays."

        assays = {k: self._ensure_array_like(v) for k, v in assays.items()} if assays is not None else None
        super().__init__(assays=assays,
                         row_data=row_data,
                         column_data=column_data,
                         row_names=row_names,
                         column_names=column_names,
                         metadata=metadata or {},
                         **kwargs)

    @staticmethod
    def _ensure_array_like(x: Any) -> Any:
        """Return an array-like object, wrapping R matrices with ``RMatrixAdapter``.

        Args:
            x: Candidate assay object (NumPy array, R matrix, adapter, etc.).

        Returns:
            Any: ``RMatrixAdapter`` if ``x`` is an R matrix; otherwise ``x`` unchanged.
        """
        try:
            from rpy2.rinterface import Sexp
            if hasattr(x, "__sexp__") and "matrix" in x.rclass:
                return RMatrixAdapter(x, _r)
        except Exception:
            pass
        return x

    # ------- convenience getters -------
    def assay(self, name: str, as_numpy=False, as_pandas=False):
        """Retrieve an assay by name with optional conversion.

        Args:
            name: Assay name to fetch.
            as_numpy: If ``True``, return a NumPy array.
            as_pandas: If ``True``, return a pandas DataFrame (uses R dimnames if R-backed).

        Returns:
            Any: The stored object (``RMatrixAdapter`` or array-like) unless
            conversion flags are set; then returns the converted representation.
        """
        obj = self.assays[name]

        if isinstance(obj, RMatrixAdapter):
            arr = obj.to_numpy()
        else:
            arr = obj
        if as_pandas:
            if isinstance(obj, RMatrixAdapter):
                rn = list(_r.ro.baseenv["rownames"](obj.rmat))
                cn = list(_r.ro.baseenv["colnames"](obj.rmat))
                return pd.DataFrame(arr, index=rn, columns=cn)
            return pd.DataFrame(arr)
        return arr if as_numpy or as_pandas else obj

    def assay_r(self, name: str):
        """Return the raw R matrix for an R-backed assay.

        Args:
            name: Assay name.

        Returns:
            Any: rpy2 SEXP matrix.

        Raises:
            TypeError: If the assay is not R-backed (i.e., not an ``RMatrixAdapter``).
        """
        obj = self.assays[name]
        if isinstance(obj, RMatrixAdapter):
            return obj.rmat
        raise TypeError(f"Assay '{name}' is not an R matrix adapter.")

    @property
    def row_data_df(self) -> Optional[pd.DataFrame]:
        """pandas.DataFrame | None: Row (feature) annotations as pandas."""
        return self.row_data.to_pandas()

    @property
    def column_data_df(self) -> Optional[pd.DataFrame]:
        """pandas.DataFrame | None: Column (sample) annotations as pandas."""
        return self.column_data.to_pandas()

    def __getitem__(self, args):
        """Slice the experiment and preserve R-backed assays.

        Args:
            args: Either ``rows`` or ``(rows, cols)`` indexers.

        Returns:
            RESummarizedExperiment: A new instance with sliced assays, annotations,
            and row/column names. R-backed assays remain R-backed via adapters.

        Raises:
            ValueError: If the number of indices is not 1 or 2.
        """
        if not isinstance(args, tuple):
            rows, cols = args, slice(None)
        else:
            if len(args) != 2:
                raise ValueError("Use obj[rows, cols].")
            rows, cols = args

        # Build sliced assays dict
        new_assays = {}
        for k, v in self.assays.items():
            if isinstance(v, RMatrixAdapter):
                new_assays[k] = _slice_rmat(v, rows, cols)
            else:
                new_assays[k] = v[rows, cols]

        # Slice row/column data
        rd = None if self.row_data is None else self.row_data_df.iloc[rows]
        cd = None if self.column_data is None else self.column_data_df.iloc[cols]

        # Slice names
        row_names = None
        col_names = None
        if self.row_names is not None:
            row_names = np.array(self.row_names)[rows].tolist()
        if self.column_names is not None:
            col_names = np.array(self.column_names)[cols].tolist()

        return self.__class__(
            assays=new_assays,
            row_data=rd,
            column_data=cd,
            row_names=row_names,
            column_names=col_names,
            metadata=dict(self.metadata)
        )

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

def filter_by_expr(obj: EdgeR, group: Sequence[str] | None = None, design: pd.DataFrame | None = None, 
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
        design = _RConverters.df_to_r_matrix(design)
    else:
        design = r.ro.NULL

    mask_r = pkg.filterByExpr(obj.assay_r(assay), group = group, design = design, **kwargs)
    mask = np.asarray(mask_r)
    return mask.astype(bool)

def calc_norm_factors(obj: EdgeR, assay = "counts", **kwargs):
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

    r_factors = pkg.calcNormFactors(rmat, **kwargs)
    norm_factors = np.asarray(r_factors)

    new_cols = BiocFrame({"norm.factors": norm_factors})
    coldata = obj.get_column_data()

    import biocutils as ut

    new_cols = ut.combine_columns(coldata, new_cols)

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


def glm_ql_fit(
    obj: EdgeR, 
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

    r, pkg = _prep_edger()
    rmat = obj.assay_r(assay)

    design_r = _RConverters.df_to_r_matrix(design)

    if dispersion is not None:
        if isinstance(dispersion, pd.DataFrame):
            dispersion = _RConverters.df_to_r_matrix(dispersion)
        elif isinstance(dispersion, np.ndarray):
            dispersion = _RConverters.np_to_r_matrix(dispersion)
    else:
        dispersion = r.ro.NULL

    if offset is not None:
        if isinstance(offset, pd.DataFrame):
            offset = _RConverters.df_to_r_matrix(offset)
        elif isinstance(offset, np.ndarray):
            offset = _RConverters.np_to_r_matrix(offset)
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
            weights = _RConverters.df_to_r_matrix(weights)
        elif isinstance(weights, np.ndarray):
            weights = _RConverters.np_to_r_matrix(weights)
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


    fit_obj = pkg.glmQLFit_default(
        rmat, 
        design = design_r,
        dispersion = dispersion,
        offset = offset,
        weights = weights, 
        legacy = legacy,
        top_proportion = top_proportion,
        **user_kwargs
    )
    
    return obj._clone(glm = fit_obj)


def glm_ql_ftest(obj: EdgeR, coef: str | None = None, contrast: Sequence | None = None, poisson_bound: bool = True,
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
    assert hasattr(obj, "glm")
    assert obj.glm is not None

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

    res = pkg.glmQLFTest(obj.glm, coef = coef, contrast = contrast, poisson_bound = poisson_bound)
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

__all__ = [n for n in dir() if not n.startswith("_")]