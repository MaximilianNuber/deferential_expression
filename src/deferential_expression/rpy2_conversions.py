from functools import singledispatch
from typing import Sequence, Union, Any, Tuple
# import types

from rpy2 import robjects as ro
from rpy2.robjects import vectors, methods, default_converter, conversion
from rpy2.robjects.vectors import ListVector
from rpy2.robjects.methods import RS4
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import get_conversion
from anndata2ri import scipy2ri
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
# from .exceptions import RPackageNotLoadedError

# from rpy_conversions.exceptions import RPackageNotLoadedError

def lazy_import_r_packages(packages: Union[str, Sequence[str]]) -> Tuple:
    """
    Lazily import one or more R packages via rpy2.robjects.packages.importr.
    Raises RPackageNotLoadedError on failure.
    """
    from rpy2.robjects.packages import importr

    if isinstance(packages, str):
        try:
            return importr(packages)
        except Exception as e:
            raise RPackageNotLoadedError(f"Could not load R pkg '{packages}': {e}") from e

    modules = []
    failures = []
    for pkg in packages:
        try:
            modules.append(importr(pkg))
        except Exception:
            modules.append(pkg)
            failures.append(pkg)
    if failures:
        raise RPackageNotLoadedError(f"Failed to load R packages: {', '.join(failures)}")
    return tuple(modules)


# ----------------------
# Python -> R converters
# ----------------------
@singledispatch
def _py_to_r(obj) -> ro.RObject:
    """Convert a generic Python object to an R object.

    Uses rpy2’s default conversion rules to map:
      - Atomic Python sequences (uniform lists/tuples) → R atomic vectors
      - Mixed‐type lists or dicts → R lists (ListVector)
      - Primitive types via their registered overloads

    Args:
        obj: Any Python object.

    Returns:
        ro.RObject: An R object representing `obj`.
    """
    with localconverter(default_converter):
        return conversion.py2rpy(obj)

@_py_to_r.register(type(None))
def _(obj) -> ro.NULL:    # noqa: F811
    """Convert Python None to R NULL.

    Args:
        obj (None): The Python None value.

    Returns:
        ro.NULL: The R NULL singleton.
    """
    return ro.NULL

@_py_to_r.register(bool)
def _(obj: bool) -> vectors.BoolVector:    # noqa: F811
    """Convert a Python boolean to a length‐1 R logical vector.

    Args:
        obj (bool): A Python boolean.

    Returns:
        BoolVector: An R logical vector of length 1 containing `obj`.
    """
    return vectors.BoolVector([obj])

@_py_to_r.register(int)
def _(obj: int) -> vectors.IntVector:    # noqa: F811
    """Convert a Python integer to a length‐1 R integer vector.

    Args:
        obj (int): A Python integer.

    Returns:
        IntVector: An R integer vector of length 1 containing `obj`.
    """
    return vectors.IntVector([obj])

@_py_to_r.register(float)
def _(obj: float) -> vectors.FloatVector:    # noqa: F811
    """Convert a Python float to a length‐1 R numeric vector.

    Args:
        obj (float): A Python float.

    Returns:
        FloatVector: An R numeric vector of length 1 containing `obj`.
    """
    return vectors.FloatVector([obj])

@_py_to_r.register(str)
def _(obj: str) -> vectors.StrVector:    # noqa: F811
    """Convert a Python string to a length‐1 R character vector.

    Args:
        obj (str): A Python string.

    Returns:
        StrVector: An R character vector of length 1 containing `obj`.
    """
    return vectors.StrVector([obj])

@_py_to_r.register(dict)
def _(obj: dict) -> ListVector:    # noqa: F811
    """Convert a Python dict to an R named list.

    Recursively applies `_py_to_r` to each value, and uses the dict keys
    as names in the resulting R list.

    Args:
        obj (dict): A mapping of keys to Python values.

    Returns:
        ListVector: An R named list with the same keys and converted values.
    """
    rdict = {str(k): _py_to_r(v) for k, v in obj.items()}
    return ListVector(rdict)

@_py_to_r.register(np.ndarray)
def _(obj: np.ndarray) -> ro.Matrix:    # noqa: F811
    """Convert a NumPy array to an R array or vector.

    Uses numpy2ri for seamless handling of:
      - 1D arrays → R atomic vectors
      - 2D arrays → R matrices
      - Higher‐dimensional arrays → R arrays

    Args:
        obj (np.ndarray): A NumPy array of any shape and dtype.

    Returns:
        ro.Matrix or Vector: The equivalent R array/matrix/vector.
    """
    with localconverter(default_converter + numpy2ri.converter):
        return numpy2ri.py2rpy(obj)

@_py_to_r.register(pd.DataFrame)
def _(obj: pd.DataFrame) -> ro.DataFrame:    # noqa: F811
    """Convert a pandas DataFrame to an R data.frame.

    Preserves column names, row names, and dtypes via pandas2ri.

    Args:
        obj (pd.DataFrame): A pandas DataFrame.

    Returns:
        ro.DataFrame: The equivalent R data.frame.
    """
    with localconverter(default_converter + pandas2ri.converter):
        return conversion.py2rpy(obj)

@_py_to_r.register(pd.Series)
def _(obj: pd.Series) -> vectors.Vector:    # noqa: F811
    """Convert a pandas Series to an R vector.

    Preserves the Series name as an R vector name if available.

    Args:
        obj (pd.Series): A pandas Series.

    Returns:
        Vector: The equivalent R atomic vector.
    """
    with localconverter(default_converter + pandas2ri.converter):
        return conversion.py2rpy(obj)

@_py_to_r.register(csr_matrix)
def _(obj: csr_matrix) -> ro.Matrix:    # noqa: F811
    """Convert a SciPy CSR sparse matrix to an R sparse matrix.

    Args:
        obj (csr_matrix): A SciPy CSR sparse matrix.

    Returns:
        ro.Matrix: An R dgCMatrix or similar sparse S4 object.
    """
    with localconverter(default_converter + scipy2ri.converter):
        return conversion.py2rpy(obj)

@_py_to_r.register(csc_matrix)
def _(obj: csc_matrix) -> ro.Matrix:    # noqa: F811
    """Convert a SciPy CSC sparse matrix to an R sparse matrix.

    Args:
        obj (csc_matrix): A SciPy CSC sparse matrix.

    Returns:
        ro.Matrix: An R dgCMatrix or similar sparse S4 object.
    """
    with localconverter(default_converter + scipy2ri.converter):
        return conversion.py2rpy(obj)

# ----------------------
# R -> Python converters
# ----------------------
@singledispatch
def _r_to_py(obj: ro.RObject): # noqa: F811
    """
    Generic R to Python conversion using rpy2's default converter.

    Args:
        obj: An R object of unknown type.

    Returns:
        Python equivalent of the R object via generic rpy2 conversion.
    """
    with localconverter(default_converter):
        return conversion.rpy2py(obj)

@_r_to_py.register(type(ro.NULL))
def _(obj) -> None: # noqa: F811
    """
    Convert R NULL to Python None.

    Args:
        obj: R NULL object.

    Returns:
        None
    """
    return None

@_r_to_py.register(vectors.BoolVector)
def _(obj): # noqa: F811
    """
    Convert an R logical vector to Python bool or list of bools.

    Args:
        obj: rpy2.robjects.vectors.BoolVector

    Returns:
        Single bool if length 1, else list of bools.
    """
    vals = list(obj)
    return bool(vals[0]) if len(vals) == 1 else [bool(x) for x in vals]

@_r_to_py.register(vectors.IntVector)
def _(obj): # noqa: F811
    """
    Convert an R integer vector to Python int or list of ints.

    Args:
        obj: rpy2.robjects.vectors.IntVector

    Returns:
        Single int if length 1, else list of ints.
    """
    vals = list(obj)
    return int(vals[0]) if len(vals) == 1 else vals

@_r_to_py.register(vectors.FloatVector)
def _(obj): # noqa: F811
    """
    Convert an R numeric (float) vector to Python float or list of floats.

    Args:
        obj: rpy2.robjects.vectors.FloatVector

    Returns:
        Single float if length 1, else list of floats.
    """
    vals = list(obj)
    return float(vals[0]) if len(vals) == 1 else vals

@_r_to_py.register(vectors.StrVector)
def _(obj): # noqa: F811
    """
    Convert an R string vector to Python str or list of str.

    Args:
        obj: rpy2.robjects.vectors.StrVector

    Returns:
        Single str if length 1, else list of str.
    """
    vals = list(obj)
    return str(vals[0]) if len(vals) == 1 else vals

@_r_to_py.register(ListVector)
def _(obj): # noqa: F811
    """
    Convert an R ListVector to Python dict or list.

    Args:
        obj: rpy2.robjects.vectors.ListVector

    Returns:
        dict if ListVector has names, else list.
    """
    items = [_r_to_py(obj[i]) for i in range(len(obj))]
    names = list(obj.names) if obj.names != ro.NULL else None
    return dict(zip(names, items)) if names else items

@_r_to_py.register(vectors.Matrix)
def _(obj: vectors.Matrix): # noqa: F811
    """
    Convert an R matrix to a NumPy array.

    Args:
        obj: rpy2.robjects.vectors.Matrix

    Returns:
        numpy.ndarray via numpy2ri converter.
    """
    with localconverter(default_converter + numpy2ri.converter):
        return conversion.rpy2py(obj)

@_r_to_py.register(ro.DataFrame)
def _(obj: ro.DataFrame): # noqa: F811
    """
    Convert an R DataFrame to a pandas.DataFrame.

    Args:
        obj: rpy2.robjects.DataFrame

    Returns:
        pandas.DataFrame via pandas2ri converter.
    """
    with localconverter(default_converter + pandas2ri.converter):
        return conversion.rpy2py(obj)

@_r_to_py.register(vectors.Vector)
def _(obj: vectors.Vector): # noqa: F811
    """
    Convert a generic R atomic vector to a Python list.

    Args:
        obj: rpy2.robjects.vectors.Vector

    Returns:
        list of elements.
    """
    return list(obj)

@_r_to_py.register(RS4)
def _(obj: RS4): # noqa: F811
    """
    Convert an R S4 object.

    Handles sparse Matrix S4 classes via scipy2ri; otherwise, returns
    a dict of slot values.

    Args:
        obj: rpy2.robjects.methods.RS4

    Returns:
        dict or sparse matrix conversion.
    """
    # Handle sparse S4 via scipy2ri
    cls = obj.rclass[0]
    if cls in ("dgCMatrix", "dgRMatrix"):
        with localconverter(default_converter + scipy2ri.converter):
            cv = get_conversion()
            return cv.rpy2py(obj)
    # Otherwise, map slots to dict
    return {name: _r_to_py(obj.slots[name]) for name in obj.slotnames()}