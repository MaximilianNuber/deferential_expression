from typing import Any
from bioc2ri.lazy_r_env import get_r_environment

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
        _r = get_r_environment()
        _limma_pkg = _r.importr("limma")
    return _limma_pkg