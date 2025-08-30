from typing import Any, Optional, Sequence, Union, Dict
import numpy as np
import pandas as pd

from biocframe import BiocFrame
from summarizedexperiment import SummarizedExperiment

from pyrtools.lazy_r_env import get_r_environment, r
from pyrtools.r_converters import RConverters


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
    def __init__(self, rmat: Any, r_manager):
        """Initialize the adapter.

        Args:
            rmat: An R matrix SEXP object (rpy2).
            r_manager: rpy2 manager/environment providing ``ro`` and converters.

        Notes:
            ``shape`` is computed from the R-level ``dim(rmat)`` at init time.
        """
        self._rmat = rmat
        self._r = get_r_environment()
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
    with r.localconverter(r.default_converter + r.numpy2ri.converter):
        r_mat = r.get_conversion().py2rpy(df.values)
    r_mat = r.ro.baseenv["rownames<-"](r_mat, r.StrVector(df.index.astype(str).to_numpy()))
    r_mat = r.ro.baseenv["colnames<-"](r_mat, r.StrVector(df.columns.astype(str).to_numpy()))
    return r_mat

def _df_to_r_df(df: pd.DataFrame) -> Any:
    """Convert a pandas DataFrame to an R ``data.frame``.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Any: rpy2 SEXP data.frame object.
    """
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        return r.get_conversion().py2rpy(df)

# ------------------------------------------------------------------
# RESummarizedExperiment
# ------------------------------------------------------------------

def _to_r_index(idx, n, *, r=get_r_environment()):
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
    r = get_r_environment()
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
                return RMatrixAdapter(x, get_r_environment())
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
                rn = list(r.ro.baseenv["rownames"](obj.rmat))
                cn = list(r.ro.baseenv["colnames"](obj.rmat))
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
    
    def to_summarized_experiment(self) -> SummarizedExperiment:
        """Convert to a base ``SummarizedExperiment``, converting R matrices to NumPy.

        Returns:
            SummarizedExperiment: New instance with all assays as NumPy arrays.
        """
        new_assays = {}
        assay_names = self.assay_names
        for assay in assay_names:
            new_assays[assay] = self.assays[assay].to_numpy()
        
        return SummarizedExperiment(
            assays=new_assays,
            row_data=self.row_data,
            column_data=self.col_data,
            row_names=self.row_names,
            column_names=self.column_names,
            metadata=dict(self.metadata)
        )
    
    @staticmethod
    def from_summarized_experiment(se: SummarizedExperiment) -> "RESummarizedExperiment":
        """Create an ``RESummarizedExperiment`` from a base ``SummarizedExperiment``.

        Args:
            se: Input ``SummarizedExperiment`` instance.

        Returns:
            RESummarizedExperiment: New instance with all assays wrapped as
            ``RMatrixAdapter`` if they are NumPy arrays.
        """
        re_assays = {}
        assay_names = se.assay_names

        colnames = se.colnames
        rownames = se.rownames

        for assay in assay_names:
            _assay = se.assays[assay]
            _assay = RConverters.numpy_to_r_matrix(_assay, rownames=rownames, colnames = colnames)

            re_assays[assay] = _assay

        return RESummarizedExperiment(
            assays=re_assays,
            row_data=se.row_data,
            column_data=se.column_data,
            row_names=se.rownames,
            column_names=se.colnames,
            metadata=dict(se.metadata)
        )