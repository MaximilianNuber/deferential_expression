from functools import lru_cache
from bioc2ri.lazy_r_env import r
import numpy as np

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

def numpy_to_r_matrix(mat, rownames=None, colnames=None):
    from bioc2ri.rutils import is_r
    from bioc2ri.rnames import set_rownames, set_colnames
    from bioc2ri import numpy_plugin
    np_eng = numpy_plugin()

    rmat = np_eng.py2r(mat)
    if rownames is not None:
        if not is_r(rownames):
            rownames = np.asarray(rownames, dtype = str)
        rmat = set_rownames(rmat, rownames)
    if colnames is not None:
        if not is_r(colnames):
            colnames = np.asarray(colnames, dtype = str)
        rmat = set_colnames(rmat, colnames)
    return rmat

def pandas_to_r_matrix(df):
    _mat = df.values
    mat = numpy_to_r_matrix(_mat, rownames = df.index.to_list(), colnames = df.columns.to_list())
    return mat