from typing import Any, Optional, Sequence, Union, Dict, no_type_check
import numpy as np
import pandas as pd

from summarizedexperiment import SummarizedExperiment
from bioc2ri.rutils import r_dim, is_r
from bioc2ri.rnames import set_rownames, set_colnames

from bioc2ri.lazy_r_env import get_r_environment, r
from bioc2ri import numpy_plugin
from .rpy2_manager import Rpy2ManagerProto

from numpy.typing import NDArray
NumericDType = np.integer[Any] | np.floating[Any]
NumericArray = NDArray[NumericDType]

IndexLike = Union[
    slice,
    int,
    Sequence[int],
    Sequence[bool],
    NDArray[NumericDType],
    pd.Index,
]

np_eng = numpy_plugin()


def numpy_to_r_matrix(
    mat: Any,
    rownames: Optional[Sequence[str]] = None,
    colnames: Optional[Sequence[str]] = None,
) -> Any:
    rmat = np_eng.py2r(mat)

    if rownames is not None:
        if not is_r(rownames):
            rn_arr = np.asarray(rownames, dtype=str)
            rmat = set_rownames(rmat, rn_arr)
        else:
            rmat = set_rownames(rmat, rownames)

    if colnames is not None:
        if not is_r(colnames):
            cn_arr = np.asarray(colnames, dtype=str)
            rmat = set_colnames(rmat, cn_arr)
        else:
            rmat = set_colnames(rmat, colnames)

    return rmat


def _to_r_index(
    idx: IndexLike,
    n: int,
    *,
    r: Any = None,
) -> Any:
    """Build an R index (logical/integer) from a Python index for 1-based R.

    Args:
        idx: Python index (slice, int, sequence/array, or boolean mask).
        n: Length along the dimension being indexed.
        r: rpy2 environment providing vector constructors.

    Returns:
        Any: R index vector suitable for subsetting (logical or integer).
    """
    if r is None:
        r = get_r_environment()

    if isinstance(idx, slice):
        start, stop, step = idx.indices(n)
        # R is 1-based
        return r.IntVector(list(range(start + 1, stop + 1, step)))

    if isinstance(idx, (list, np.ndarray, pd.Index)):
        idx_arr = np.asarray(idx)
        if idx_arr.dtype == bool:
            return r.BoolVector(idx_arr.tolist())
        return r.IntVector((idx_arr + 1).tolist())

    if isinstance(idx, int):
        return r.IntVector([idx + 1])

    # fallback: ':' (all indices)
    return r.IntVector(list(range(1, n + 1)))


def _is_r_matrix(value: Any) -> bool:
    """Check if value is an R matrix SEXP.
    
    Args:
        value: Object to check.
    
    Returns:
        bool: True if value is an R matrix, False otherwise.
    """
    try:
        return hasattr(value, "__sexp__") and "matrix" in value.rclass
    except (AttributeError, TypeError):
        return False
    
@no_type_check
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
    # CRITICAL: use drop=FALSE to preserve matrix structure when subsetting single rows/cols
    out = bracket(rm, ridx, cidx, drop=False)
    return RMatrixAdapter(out, r)

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
    def __init__(self, rmat: Any, r_manager: Optional[Rpy2ManagerProto] = None):
        """Initialize the adapter.

        Args:
            rmat: An R matrix SEXP object (rpy2).
            r_manager: Optional rpy2 manager/environment providing ``ro`` and converters.
                If None, uses the default environment from get_r_environment().

        Notes:
            ``shape`` is computed from the R-level ``dim(rmat)`` at init time.
        """
        self._rmat = rmat
        self._r = r_manager if r_manager is not None else get_r_environment()
        dims = r_dim(rmat)
        self._shape = (int(dims[0]), int(dims[1]))
    @property
    def shape(self) -> tuple[int, int]: 
        """Tuple[int, int]: Matrix shape (rows, cols) without conversion."""
        return self._shape
    @property
    def rmat(self) -> Any:  
        """Any: Underlying rpy2 SEXP matrix (read-only reference)."""
        return self._rmat
    def to_numpy(self) -> NDArray[NumericDType]:
        """Convert the wrapped R matrix to a NumPy array.

        Returns:
            np.ndarray: Dense NumPy array with the same values as the R matrix.
        """
        with self._r.localconverter(self._r.default_converter + self._r.numpy2ri.converter):
            npmat = self._r.get_conversion().rpy2py(self._rmat) 
        return npmat # type: ignore
    # def __getitem__(self, idx):
    #     return self.to_numpy()[idx]
    # CRITICAL FIX: keep results as an RMatrixAdapter, not a NumPy array
    def __getitem__(self, key: Union[IndexLike, tuple[IndexLike, IndexLike]]) -> "RMatrixAdapter":
        """Slice the R matrix in R and return another ``RMatrixAdapter``.

        Args:
            key: Either ``rows`` or ``(rows, cols)`` using Python indexing
                (ints, slices, boolean masks, or integer arrays).

        Returns:
            RMatrixAdapter: Adapter wrapping the sliced R matrix.

        Raises:
            IndexError: If more than two indices are provided.
        """

        rows: IndexLike
        cols: IndexLike

        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Use 2D indexing: [rows, cols].")
            rows, cols = key
        else:
            rows, cols = key, slice(None)

        return _slice_rmat(self, rows, cols) # type: ignore


    def __array__(self, dtype:Any=None) -> NDArray[NumericDType]:
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
    """Convert a pandas DataFrame to an R matrix, preserving dimnames."""
    # bioc2ri handles numpy conversion + dimnames; we just pass arrays and names
    mat = np_eng.py2r(df.to_numpy(copy=False))
    rownames = np.asarray(df.index).astype("str")
    mat = set_rownames(mat, np_eng.py2r(rownames))
    colnames = np.asarray(df.columns).astype("str")
    mat = set_colnames(mat, np_eng.py2r(colnames))
    return mat

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

class RESummarizedExperiment(SummarizedExperiment): # type: ignore[misc]
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
        **kwargs: Any
    ) -> None:
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
        # Cache for accessors (like xarray/pandas)
        self._cache: Dict[str, Any] = {}
        
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
        if _is_r_matrix(x):
            return RMatrixAdapter(x)
        return x

    # ------- convenience getters -------
    def assay(
        self, name: str, as_numpy: bool = False, as_pandas: bool = False
    ) -> Union[RMatrixAdapter, NDArray[NumericDType], pd.DataFrame, Any]:
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

    def assay_r(self, name: str) -> Any:
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
    
    def set_assay(
        self,
        name: str,
        value: Any,
        *,
        rownames: Optional[Sequence[str]] = None,
        colnames: Optional[Sequence[str]] = None,
    ) -> "RESummarizedExperiment":
        """
        Return a new RESummarizedExperiment with an updated assay.

        This is a functional-style "setter": it does NOT mutate the current
        object, but instead returns a new instance with the modified assays
        mapping. Python arrays are automatically converted to R-backed
        matrices via RMatrixAdapter.

        Parameters
        ----------
        name:
            Assay name (e.g. "counts", "logcounts").
        value:
            Matrix-like object. Can be:

            * np.ndarray (or array-like)
            * pandas.DataFrame
            * rpy2 R matrix (SEXP) – wrapped in RMatrixAdapter
            * RMatrixAdapter – stored as-is

        rownames, colnames:
            Optional explicit row/column names. If omitted, we fall back to
            self.row_names / self.column_names where available.
        """
        # Start from a shallow copy of the existing assays (functional style)
        new_assays: Dict[str, Any] = dict(self.assays)

        if isinstance(value, RMatrixAdapter):
            new_assays[name] = value
        else:
            if _is_r_matrix(value):
                new_assays[name] = RMatrixAdapter(value)
            else:
                # Python-side data -> NumPy
                rn: Optional[list[str]]
                cn: Optional[list[str]]
                
                if isinstance(value, pd.DataFrame):
                    arr = value.to_numpy(copy=False)

                    # rownames
                    if rownames is not None:
                        rn = list(rownames)
                    else:
                        rn = [str(x) for x in value.index.to_list()]

                    # colnames
                    if colnames is not None:
                        cn = list(colnames)
                    else:
                        cn = [str(x) for x in value.columns.to_list()]
                else:
                    arr = np.asarray(value)

                    if rownames is not None:
                        rn = list(rownames)
                    else:
                        rn = (
                            list(self.row_names)
                            if self.row_names is not None
                            else None
                        )

                    if colnames is not None:
                        cn = list(colnames)
                    else:
                        cn = (
                            list(self.column_names)
                            if self.column_names is not None
                            else None
                        )

                # sanity checks
                if rn is not None and len(rn) != arr.shape[0]:
                    raise ValueError(
                        f"Length of rownames ({len(rn)}) does not match "
                        f"number of rows in assay ({arr.shape[0]})."
                    )
                if cn is not None and len(cn) != arr.shape[1]:
                    raise ValueError(
                        f"Length of colnames ({len(cn)}) does not match "
                        f"number of columns in assay ({arr.shape[1]})."
                    )

                rmat = numpy_to_r_matrix(
                    arr,
                    rownames=rn,
                    colnames=cn,
                )
                new_assays[name] = RMatrixAdapter(rmat, get_r_environment())

        return self.__class__(
            assays=new_assays,
            row_data=self.row_data_df if self.row_data is not None else None,
            column_data=self.column_data_df if self.column_data is not None else None,
            row_names=self.row_names,
            column_names=self.column_names,
            metadata=dict(self.metadata),
        )


    @property
    def row_data_df(self) -> Optional[pd.DataFrame]:
        """pandas.DataFrame | None: Row (feature) annotations as pandas."""
        if self.row_data is None:
            return None
        return self.row_data.to_pandas() # type: ignore

    @property
    def column_data_df(self) -> Optional[pd.DataFrame]:
        """pandas.DataFrame | None: Column (sample) annotations as pandas."""
        if self.column_data is None:
            return None
        return self.column_data.to_pandas() # type: ignore

    @no_type_check
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
                # For non-RMatrixAdapter assays (numpy arrays, etc), we need to handle
                # single-integer indexing carefully to preserve 2D structure
                sliced = v[rows, cols]
                # If slicing resulted in 1D array, reshape to keep 2D
                if hasattr(sliced, 'ndim') and sliced.ndim == 1:
                    # Determine which dimension was collapsed
                    # Check if rows is a single int
                    if isinstance(rows, int):
                        sliced = sliced.reshape(1, -1)
                    # Check if cols is a single int
                    elif isinstance(cols, int):
                        sliced = sliced.reshape(-1, 1)
                new_assays[k] = sliced

        # Slice row/column data - ensure we always get DataFrames, not Series
        rd = None
        if self.row_data is not None:
            rd_sliced = self.row_data_df.iloc[rows]
            # If slicing with single int returns Series, convert back to DataFrame
            if isinstance(rd_sliced, pd.Series):
                rd = pd.DataFrame([rd_sliced])
            else:
                rd = rd_sliced
        
        cd = None
        if self.column_data is not None:
            cd_sliced = self.column_data_df.iloc[cols]
            # If slicing with single int returns Series, convert back to DataFrame
            if isinstance(cd_sliced, pd.Series):
                cd = pd.DataFrame([cd_sliced])
            else:
                cd = cd_sliced

        # Slice names
        row_names = None
        col_names = None
        if self.row_names is not None:
            row_names = np.array(self.row_names)[rows].tolist()
            # Ensure list for single-element case
            if not isinstance(row_names, list):
                row_names = [row_names]
        if self.column_names is not None:
            col_names = np.array(self.column_names)[cols].tolist()
            # Ensure list for single-element case
            if not isinstance(col_names, list):
                col_names = [col_names]

        return self.__class__(
            assays=new_assays,
            row_data=rd,
            column_data=cd,
            row_names=row_names,
            column_names=col_names,
            metadata=dict(self.metadata)
        )

    def propagate_dimnames_to_assays(self) -> "RESummarizedExperiment":
        """The current row names and column names of the RESummarizedExperiment are set for each R-matrix in assays."""
        
        row_names = self.get_row_names()
        col_names = self.get_column_names()

        assays = self._assays()
        assay_names = list(assays.keys())

        new_assays = {}
        for assay in assay_names:
            mat = assays[assay].rmat
            new_mat = set_colnames(mat, colnames)
            new_mat = set_rownames(new_mat, row_names)

            new_assays[assay] = new_mat
        
        out = self.copy()
        out._assays = new_assays

        return out
    
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
            column_data=self.column_data,
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
            _assay = numpy_to_r_matrix(_assay, rownames=rownames, colnames = colnames)

            re_assays[assay] = _assay

        return RESummarizedExperiment(
            assays=re_assays,
            row_data=se.row_data,
            column_data=se.column_data,
            row_names=se.rownames,
            column_names=se.colnames,
            metadata=dict(se.metadata)
        )