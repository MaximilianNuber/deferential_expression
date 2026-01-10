"""
RMatrixAdapter: A numpy array-compatible wrapper around R matrices.

This module implements an RMatrixAdapter class that fulfills the numpy array
contract, allowing it to be used wherever numpy arrays are expected while
keeping the underlying data in R for efficiency.

The array contract includes:
- __array__: np.asarray() support
- __array_ufunc__: numpy ufunc support (np.add, np.sum, etc.)
- __array_function__: numpy function dispatch
- shape, ndim, size, dtype, T: standard array attributes
- __getitem__: slicing and indexing
- __len__, __iter__: iteration support
- Arithmetic operators: +, -, *, /, //, **, @
- Comparison operators: <, <=, ==, !=, >, >=
- Mathematical methods: sum, mean, std, var, min, max, etc.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Union, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .rpy2_manager import Rpy2ManagerProto

# Type aliases
NumericDType = np.integer[Any] | np.floating[Any]
NumericArray = NDArray[NumericDType]

IndexLike = Union[
    slice,
    int,
    Sequence[int],
    Sequence[bool],
    NDArray[NumericDType],
]


def _get_r_environment():
    """Get the R environment lazily."""
    from bioc2ri.lazy_r_env import get_r_environment
    return get_r_environment()


def _r_dim(rmat: Any) -> tuple[int, ...]:
    """Get the dimensions of an R matrix."""
    from bioc2ri.rutils import r_dim
    return tuple(int(x) for x in r_dim(rmat))


def _to_r_index(idx: IndexLike, n: int, r: Any) -> Any:
    """Convert Python index to R index (1-based)."""
    if isinstance(idx, slice):
        start, stop, step = idx.indices(n)
        return r.IntVector(list(range(start + 1, stop + 1, step)))
    
    if isinstance(idx, (list, np.ndarray)):
        idx_arr = np.asarray(idx)
        if idx_arr.dtype == bool:
            return r.BoolVector(idx_arr.tolist())
        return r.IntVector((idx_arr + 1).tolist())
    
    if isinstance(idx, int):
        # Handle negative indexing
        if idx < 0:
            idx = n + idx
        return r.IntVector([idx + 1])
    
    # fallback: all indices
    return r.IntVector(list(range(1, n + 1)))


class RMatrixAdapter:
    """
    A numpy-compatible wrapper around R matrices via rpy2.
    
    This adapter implements the numpy array protocol, making it usable
    wherever numpy arrays are expected. The underlying data stays in R
    until explicitly converted.
    
    Implements:
    - Array protocol: __array__, __array_ufunc__, __array_function__
    - Standard attributes: shape, ndim, size, dtype, T
    - Indexing: __getitem__, __setitem__
    - Iteration: __len__, __iter__, __contains__
    - Arithmetic: +, -, *, /, //, %, **, @, and unary -, +, abs
    - Comparison: <, <=, ==, !=, >, >=
    - Reduction: sum, mean, std, var, min, max, prod, any, all
    - Other: copy, flatten, ravel, reshape, astype
    
    Attributes:
        _rmat: The underlying rpy2 R matrix object.
        _shape: Cached tuple of dimensions.
        _r: The rpy2 environment/manager.
    
    Example:
        >>> adapter = RMatrixAdapter(r_matrix)
        >>> adapter.shape
        (100, 10)
        >>> np.asarray(adapter)  # Convert to numpy
        array([[...]])
        >>> adapter + 1  # Arithmetic (materializes to numpy)
        array([[...]])
        >>> adapter[:10, :5]  # Slice in R, returns new adapter
        <RMatrixAdapter (10, 5)>
    """
    
    __slots__ = ("_rmat", "_shape", "_r", "_dtype_cache")
    
    # Tell numpy that we implement the array interface
    __array_priority__ = 10.0
    
    def __init__(
        self,
        rmat: Any,
        r_manager: Optional["Rpy2ManagerProto"] = None,
    ) -> None:
        """
        Initialize an RMatrixAdapter.
        
        Args:
            rmat: An R matrix object (rpy2 SEXP).
            r_manager: Optional rpy2 manager/environment. If None, uses default.
        """
        self._rmat = rmat
        self._r = r_manager if r_manager is not None else _get_r_environment()
        self._shape = _r_dim(rmat)
        self._dtype_cache: Optional[np.dtype] = None
    
    # =========================================================================
    # Core Properties
    # =========================================================================
    
    @property
    def rmat(self) -> Any:
        """The underlying R matrix object (read-only)."""
        return self._rmat
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of array dimensions."""
        return self._shape
    
    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return len(self._shape)
    
    @property
    def size(self) -> int:
        """Number of elements in the array."""
        result = 1
        for dim in self._shape:
            result *= dim
        return result
    
    @property
    def dtype(self) -> np.dtype:
        """Data type of the array elements."""
        if self._dtype_cache is None:
            # Sample a small portion to determine dtype
            arr = self.to_numpy()
            self._dtype_cache = arr.dtype
        return self._dtype_cache
    
    @property
    def T(self) -> "RMatrixAdapter":
        """Transposed matrix (computed in R)."""
        r = self._r
        t_func = r.ro.baseenv["t"]
        transposed = t_func(self._rmat)
        return RMatrixAdapter(transposed, r)
    
    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the array elements."""
        return self.size * self.dtype.itemsize
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_numpy(self) -> NumericArray:
        """
        Convert the R matrix to a NumPy array.
        
        Returns:
            np.ndarray: Dense NumPy array with the same values.
        """
        with self._r.localconverter(
            self._r.default_converter + self._r.numpy2ri.converter
        ):
            return self._r.get_conversion().rpy2py(self._rmat)
    
    def __array__(self, dtype: Any = None) -> NumericArray:
        """
        Support np.asarray(adapter) and np.array(adapter).
        
        Args:
            dtype: Optional dtype to cast to.
        
        Returns:
            np.ndarray: The converted NumPy array.
        """
        arr = self.to_numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr
    
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs,
        **kwargs,
    ) -> Any:
        """
        Handle numpy universal functions (ufuncs).
        
        This enables operations like np.add(adapter, 1), np.sin(adapter), etc.
        The adapter is converted to numpy for the operation.
        
        Args:
            ufunc: The numpy ufunc being called.
            method: The ufunc method ('__call__', 'reduce', etc.).
            *inputs: Input arrays.
            **kwargs: Additional arguments.
        
        Returns:
            The result of the ufunc operation.
        """
        # Convert RMatrixAdapter inputs to numpy
        converted_inputs = []
        for inp in inputs:
            if isinstance(inp, RMatrixAdapter):
                converted_inputs.append(inp.to_numpy())
            else:
                converted_inputs.append(inp)
        
        return getattr(ufunc, method)(*converted_inputs, **kwargs)
    
    def __array_function__(self, func, types, args, kwargs):
        """
        Handle numpy functions via __array_function__ protocol (NEP 18).
        
        This enables operations like np.sum(adapter), np.mean(adapter), etc.
        """
        # Convert RMatrixAdapter arguments to numpy
        def convert(x):
            if isinstance(x, RMatrixAdapter):
                return x.to_numpy()
            return x
        
        converted_args = tuple(convert(arg) for arg in args)
        converted_kwargs = {k: convert(v) for k, v in kwargs.items()}
        
        return func(*converted_args, **converted_kwargs)
    
    # =========================================================================
    # Indexing and Slicing
    # =========================================================================
    
    def __getitem__(
        self,
        key: Union[IndexLike, tuple[IndexLike, ...]],
    ) -> Union["RMatrixAdapter", np.ndarray]:
        """
        Slice the R matrix in R and return a new adapter.
        
        For 2D indexing, returns RMatrixAdapter.
        For single element access, returns the scalar value.
        
        Args:
            key: Index or slice specification.
        
        Returns:
            RMatrixAdapter for slices, scalar for single element.
        """
        r = self._r
        
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Use 2D indexing: [rows, cols]")
            rows, cols = key
        else:
            rows, cols = key, slice(None)
        
        # Check for single element access
        single_row = isinstance(rows, int)
        single_col = isinstance(cols, int)
        
        nrow, ncol = self._shape
        ridx = _to_r_index(rows, nrow, r)
        cidx = _to_r_index(cols, ncol, r)
        
        bracket = r.ro.baseenv["["]
        
        if single_row and single_col:
            # Single element: return scalar
            result = bracket(self._rmat, ridx, cidx)
            return float(result[0])
        else:
            # Slice: return new adapter with drop=FALSE
            result = bracket(self._rmat, ridx, cidx, drop=False)
            return RMatrixAdapter(result, r)
    
    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return self._shape[0]
    
    def __iter__(self):
        """Iterate over rows."""
        for i in range(len(self)):
            yield self[i, :]
    
    def __contains__(self, item) -> bool:
        """Check if item is in the array."""
        return item in self.to_numpy()
    
    # =========================================================================
    # Arithmetic Operators
    # =========================================================================
    
    def __add__(self, other) -> np.ndarray:
        return self.to_numpy() + np.asarray(other)
    
    def __radd__(self, other) -> np.ndarray:
        return np.asarray(other) + self.to_numpy()
    
    def __sub__(self, other) -> np.ndarray:
        return self.to_numpy() - np.asarray(other)
    
    def __rsub__(self, other) -> np.ndarray:
        return np.asarray(other) - self.to_numpy()
    
    def __mul__(self, other) -> np.ndarray:
        return self.to_numpy() * np.asarray(other)
    
    def __rmul__(self, other) -> np.ndarray:
        return np.asarray(other) * self.to_numpy()
    
    def __truediv__(self, other) -> np.ndarray:
        return self.to_numpy() / np.asarray(other)
    
    def __rtruediv__(self, other) -> np.ndarray:
        return np.asarray(other) / self.to_numpy()
    
    def __floordiv__(self, other) -> np.ndarray:
        return self.to_numpy() // np.asarray(other)
    
    def __rfloordiv__(self, other) -> np.ndarray:
        return np.asarray(other) // self.to_numpy()
    
    def __mod__(self, other) -> np.ndarray:
        return self.to_numpy() % np.asarray(other)
    
    def __rmod__(self, other) -> np.ndarray:
        return np.asarray(other) % self.to_numpy()
    
    def __pow__(self, other) -> np.ndarray:
        return self.to_numpy() ** np.asarray(other)
    
    def __rpow__(self, other) -> np.ndarray:
        return np.asarray(other) ** self.to_numpy()
    
    def __matmul__(self, other) -> np.ndarray:
        return self.to_numpy() @ np.asarray(other)
    
    def __rmatmul__(self, other) -> np.ndarray:
        return np.asarray(other) @ self.to_numpy()
    
    def __neg__(self) -> np.ndarray:
        return -self.to_numpy()
    
    def __pos__(self) -> np.ndarray:
        return +self.to_numpy()
    
    def __abs__(self) -> np.ndarray:
        return np.abs(self.to_numpy())
    
    # =========================================================================
    # Comparison Operators
    # =========================================================================
    
    def __lt__(self, other) -> np.ndarray:
        return self.to_numpy() < np.asarray(other)
    
    def __le__(self, other) -> np.ndarray:
        return self.to_numpy() <= np.asarray(other)
    
    def __eq__(self, other) -> np.ndarray:  # type: ignore[override]
        if isinstance(other, RMatrixAdapter):
            return self.to_numpy() == other.to_numpy()
        return self.to_numpy() == np.asarray(other)
    
    def __ne__(self, other) -> np.ndarray:  # type: ignore[override]
        if isinstance(other, RMatrixAdapter):
            return self.to_numpy() != other.to_numpy()
        return self.to_numpy() != np.asarray(other)
    
    def __gt__(self, other) -> np.ndarray:
        return self.to_numpy() > np.asarray(other)
    
    def __ge__(self, other) -> np.ndarray:
        return self.to_numpy() >= np.asarray(other)
    
    # =========================================================================
    # Reduction Methods (computed in R)
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None, **kwargs) -> Union[float, np.ndarray]:
        """Sum of array elements (computed in R)."""
        r = self._r
        if axis is None:
            # Total sum
            result = r.ro.baseenv["sum"](self._rmat)
            return float(result[0])
        elif axis == 0:
            # Column sums
            result = r.ro.baseenv["colSums"](self._rmat)
            return np.asarray(result)
        elif axis == 1:
            # Row sums
            result = r.ro.baseenv["rowSums"](self._rmat)
            return np.asarray(result)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def mean(self, axis: Optional[int] = None, **kwargs) -> Union[float, np.ndarray]:
        """Mean of array elements (computed in R)."""
        r = self._r
        if axis is None:
            # Total mean
            result = r.ro.baseenv["mean"](self._rmat)
            return float(result[0])
        elif axis == 0:
            # Column means
            result = r.ro.baseenv["colMeans"](self._rmat)
            return np.asarray(result)
        elif axis == 1:
            # Row means
            result = r.ro.baseenv["rowMeans"](self._rmat)
            return np.asarray(result)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def std(self, axis: Optional[int] = None, ddof: int = 0, **kwargs) -> Union[float, np.ndarray]:
        """Standard deviation of array elements (computed in R).
        
        Note: R uses n-1 (ddof=1) by default, this uses ddof=0 for numpy compatibility.
        """
        r = self._r
        # sd is in stats, access via globalenv
        sd_func = r.ro.globalenv.find("sd")
        if axis is None:
            # Total sd - R's sd uses n-1, adjust if needed
            result = sd_func(r.ro.baseenv["as.vector"](self._rmat))
            sd_val = float(result[0])
            if ddof == 0:
                # Adjust from n-1 to n
                n = self.size
                sd_val = sd_val * np.sqrt((n - 1) / n)
            return sd_val
        elif axis == 0:
            # Column sds using apply
            result = r.ro.baseenv["apply"](self._rmat, 2, sd_func)
            arr = np.asarray(result)
            if ddof == 0:
                n = self._shape[0]
                arr = arr * np.sqrt((n - 1) / n)
            return arr
        elif axis == 1:
            # Row sds using apply
            result = r.ro.baseenv["apply"](self._rmat, 1, sd_func)
            arr = np.asarray(result)
            if ddof == 0:
                n = self._shape[1]
                arr = arr * np.sqrt((n - 1) / n)
            return arr
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def var(self, axis: Optional[int] = None, ddof: int = 0, **kwargs) -> Union[float, np.ndarray]:
        """Variance of array elements (computed in R).
        
        Note: R uses n-1 (ddof=1) by default, this uses ddof=0 for numpy compatibility.
        """
        r = self._r
        # var is in stats, access via globalenv
        var_func = r.ro.globalenv.find("var")
        if axis is None:
            result = var_func(r.ro.baseenv["as.vector"](self._rmat))
            var_val = float(result[0])
            if ddof == 0:
                n = self.size
                var_val = var_val * (n - 1) / n
            return var_val
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, var_func)
            arr = np.asarray(result)
            if ddof == 0:
                n = self._shape[0]
                arr = arr * (n - 1) / n
            return arr
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, var_func)
            arr = np.asarray(result)
            if ddof == 0:
                n = self._shape[1]
                arr = arr * (n - 1) / n
            return arr
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def min(self, axis: Optional[int] = None, **kwargs) -> Union[float, np.ndarray]:
        """Minimum of array elements (computed in R)."""
        r = self._r
        if axis is None:
            result = r.ro.baseenv["min"](self._rmat)
            return float(result[0])
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, r.ro.baseenv["min"])
            return np.asarray(result)
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, r.ro.baseenv["min"])
            return np.asarray(result)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def max(self, axis: Optional[int] = None, **kwargs) -> Union[float, np.ndarray]:
        """Maximum of array elements (computed in R)."""
        r = self._r
        if axis is None:
            result = r.ro.baseenv["max"](self._rmat)
            return float(result[0])
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, r.ro.baseenv["max"])
            return np.asarray(result)
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, r.ro.baseenv["max"])
            return np.asarray(result)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def prod(self, axis: Optional[int] = None, **kwargs) -> Union[float, np.ndarray]:
        """Product of array elements (computed in R)."""
        r = self._r
        if axis is None:
            result = r.ro.baseenv["prod"](self._rmat)
            return float(result[0])
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, r.ro.baseenv["prod"])
            return np.asarray(result)
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, r.ro.baseenv["prod"])
            return np.asarray(result)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def any(self, axis: Optional[int] = None, **kwargs) -> Union[bool, np.ndarray]:
        """Test if any element is True (computed in R)."""
        r = self._r
        if axis is None:
            result = r.ro.baseenv["any"](self._rmat)
            return bool(result[0])
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, r.ro.baseenv["any"])
            return np.asarray(result, dtype=bool)
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, r.ro.baseenv["any"])
            return np.asarray(result, dtype=bool)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def all(self, axis: Optional[int] = None, **kwargs) -> Union[bool, np.ndarray]:
        """Test if all elements are True (computed in R)."""
        r = self._r
        if axis is None:
            result = r.ro.baseenv["all"](self._rmat)
            return bool(result[0])
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, r.ro.baseenv["all"])
            return np.asarray(result, dtype=bool)
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, r.ro.baseenv["all"])
            return np.asarray(result, dtype=bool)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def argmin(self, axis: Optional[int] = None, **kwargs) -> Union[int, np.ndarray]:
        """Indices of minimum values (computed in R, 0-indexed)."""
        r = self._r
        which_min = r.ro.baseenv["which.min"]
        if axis is None:
            # Flatten and find min index
            result = which_min(r.ro.baseenv["as.vector"](self._rmat))
            return int(result[0]) - 1  # R is 1-indexed
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, which_min)
            return np.asarray(result) - 1
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, which_min)
            return np.asarray(result) - 1
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def argmax(self, axis: Optional[int] = None, **kwargs) -> Union[int, np.ndarray]:
        """Indices of maximum values (computed in R, 0-indexed)."""
        r = self._r
        which_max = r.ro.baseenv["which.max"]
        if axis is None:
            result = which_max(r.ro.baseenv["as.vector"](self._rmat))
            return int(result[0]) - 1
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, which_max)
            return np.asarray(result) - 1
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, which_max)
            return np.asarray(result) - 1
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def cumsum(self, axis: Optional[int] = None, **kwargs) -> np.ndarray:
        """Cumulative sum of elements (computed in R)."""
        r = self._r
        if axis is None:
            result = r.ro.baseenv["cumsum"](r.ro.baseenv["as.vector"](self._rmat))
            return np.asarray(result)
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, r.ro.baseenv["cumsum"])
            return np.asarray(result)
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, r.ro.baseenv["cumsum"])
            return np.asarray(result).T
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    def cumprod(self, axis: Optional[int] = None, **kwargs) -> np.ndarray:
        """Cumulative product of elements (computed in R)."""
        r = self._r
        if axis is None:
            result = r.ro.baseenv["cumprod"](r.ro.baseenv["as.vector"](self._rmat))
            return np.asarray(result)
        elif axis == 0:
            result = r.ro.baseenv["apply"](self._rmat, 2, r.ro.baseenv["cumprod"])
            return np.asarray(result)
        elif axis == 1:
            result = r.ro.baseenv["apply"](self._rmat, 1, r.ro.baseenv["cumprod"])
            return np.asarray(result).T
        else:
            raise ValueError(f"Invalid axis {axis} for 2D array")
    
    # =========================================================================
    # Shape Manipulation (computed in R where possible)
    # =========================================================================
    
    def copy(self) -> "RMatrixAdapter":
        """Return a copy (in R, creates a new R matrix)."""
        r = self._r
        # R's as.matrix creates a copy
        copy_mat = r.ro.baseenv["as.matrix"](self._rmat)
        return RMatrixAdapter(copy_mat, r)
    
    def flatten(self, order: str = "C") -> np.ndarray:
        """Return a flattened copy of the array (computed in R).
        
        Note: R uses column-major (Fortran) order by default.
        For row-major (C) order, we transpose first.
        """
        r = self._r
        from bioc2ri import numpy_plugin
        np_eng = numpy_plugin()
        
        if order.upper() == "F":
            # R's as.vector uses column-major order (Fortran)
            result = r.ro.baseenv["as.vector"](self._rmat)
        else:
            # For C order: transpose first, then flatten
            result = r.ro.baseenv["as.vector"](r.ro.baseenv["t"](self._rmat))
        return np.asarray(result)
    
    def ravel(self, order: str = "C") -> np.ndarray:
        """Return a flattened array (computed in R)."""
        return self.flatten(order=order)
    
    def reshape(self, *shape, order: str = "C") -> "RMatrixAdapter":
        """Return reshaped array (computed in R).
        
        For 2D shapes, returns RMatrixAdapter. Otherwise returns numpy.
        """
        r = self._r
        
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        
        if len(shape) != 2:
            # R matrices are 2D, fall back to numpy for other shapes
            from bioc2ri import numpy_plugin
            np_eng = numpy_plugin()
            arr = np.asarray(self.to_numpy())
            return arr.reshape(shape, order=order)
        
        nrow, ncol = shape
        
        if order.upper() == "F":
            # R uses column-major by default
            reshaped = r.ro.baseenv["matrix"](
                r.ro.baseenv["as.vector"](self._rmat),
                nrow=nrow,
                ncol=ncol,
                byrow=False
            )
        else:
            # For C order: flatten in C order, then reshape with byrow=TRUE
            vec = r.ro.baseenv["as.vector"](r.ro.baseenv["t"](self._rmat))
            reshaped = r.ro.baseenv["matrix"](
                vec,
                nrow=nrow,
                ncol=ncol,
                byrow=True
            )
        
        return RMatrixAdapter(reshaped, r)
    
    def squeeze(self, axis: Optional[int] = None) -> Union["RMatrixAdapter", np.ndarray]:
        """Remove axes of length 1 (computed in R).
        
        R's drop() removes dimensions of length 1.
        """
        r = self._r
        result = r.ro.baseenv["drop"](self._rmat)
        
        # Check if result is still a matrix
        try:
            dims = r.ro.baseenv["dim"](result)
            if dims is not r.ro.NULL and len(dims) == 2:
                return RMatrixAdapter(result, r)
        except:
            pass
        
        # Result is a vector, convert to numpy
        return np.asarray(result)
    
    def swapaxes(self, axis1: int, axis2: int) -> "RMatrixAdapter":
        """Swap two axes (for 2D matrix, equivalent to transpose in R)."""
        r = self._r
        if self.ndim != 2:
            raise ValueError("swapaxes only supported for 2D matrices")
        if (axis1 == 0 and axis2 == 1) or (axis1 == 1 and axis2 == 0):
            # Swap rows and cols = transpose
            return self.T
        elif axis1 == axis2:
            return self.copy()
        else:
            raise ValueError(f"Invalid axes {axis1}, {axis2} for 2D matrix")
    
    def transpose(self, *axes) -> "RMatrixAdapter":
        """Permute dimensions (computed in R).
        
        For 2D matrices, uses R's t() function.
        """
        r = self._r
        if len(axes) == 0 or axes == (None,):
            # Default transpose
            return self.T
        
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        
        if self.ndim != 2:
            raise ValueError("transpose only supported for 2D matrices")
        
        if axes == (0, 1):
            return self.copy()
        elif axes == (1, 0):
            return self.T
        else:
            raise ValueError(f"Invalid axes {axes} for 2D matrix")
    
    # =========================================================================
    # Type Conversion
    # =========================================================================
    
    def astype(self, dtype, order: str = "K", casting: str = "unsafe", copy: bool = True) -> np.ndarray:
        """Cast to specified dtype."""
        return self.to_numpy().astype(dtype, order=order, casting=casting, copy=copy)
    
    def tolist(self) -> list:
        """Return as nested Python list."""
        return self.to_numpy().tolist()
    
    def tobytes(self, order: str = "C") -> bytes:
        """Return raw bytes."""
        return self.to_numpy().tobytes(order=order)
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return f"<RMatrixAdapter shape={self._shape} dtype={self.dtype}>"
    
    def __str__(self) -> str:
        arr = self.to_numpy()
        if self.size <= 100:
            return f"RMatrixAdapter:\n{arr}"
        return f"RMatrixAdapter {self._shape}:\n{arr[:5, :5]}..."
    
    # =========================================================================
    # Boolean Context
    # =========================================================================
    
    def __bool__(self) -> bool:
        """Evaluate truthiness (raises for size > 1, like numpy)."""
        if self.size > 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )
        return bool(self.to_numpy().item())
    
    # =========================================================================
    # R-side Operations (stay in R)
    # =========================================================================
    
    def rowSums(self) -> "RMatrixAdapter":
        """Compute row sums in R (stays in R)."""
        r = self._r
        rowSums = r.ro.baseenv["rowSums"]
        result = rowSums(self._rmat)
        # Wrap as 1D "matrix" for consistency
        return result  # Returns R vector, caller can convert
    
    def colSums(self) -> "RMatrixAdapter":
        """Compute column sums in R (stays in R)."""
        r = self._r
        colSums = r.ro.baseenv["colSums"]
        result = colSums(self._rmat)
        return result
    
    def rowMeans(self) -> Any:
        """Compute row means in R (stays in R)."""
        r = self._r
        rowMeans = r.ro.baseenv["rowMeans"]
        return rowMeans(self._rmat)
    
    def colMeans(self) -> Any:
        """Compute column means in R (stays in R)."""
        r = self._r
        colMeans = r.ro.baseenv["colMeans"]
        return colMeans(self._rmat)
    
    # =========================================================================
    # R-specific Methods
    # =========================================================================
    
    def get_rownames(self) -> Optional[list[str]]:
        """Get R row names if available."""
        r = self._r
        rn = r.ro.baseenv["rownames"](self._rmat)
        if rn is r.ro.NULL or rn == r.ro.NULL:
            return None
        return list(rn)
    
    def get_colnames(self) -> Optional[list[str]]:
        """Get R column names if available."""
        r = self._r
        cn = r.ro.baseenv["colnames"](self._rmat)
        if cn is r.ro.NULL or cn == r.ro.NULL:
            return None
        return list(cn)
