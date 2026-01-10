"""deferential_expression: Python-first access to R/Bioconductor DE tools.

This package provides Python wrappers for edgeR, limma, and sva with
lazy loading of submodules to avoid R dependency checks until needed.

Usage:
    >>> from deferential_expression import RESummarizedExperiment
    >>> # edgeR is NOT loaded yet - no R dependency check
    >>> 
    >>> import deferential_expression.edger  # NOW edgeR is checked/installed
    >>> rse.edger.calc_norm_factors()
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

# Core exports that don't require R
from .resummarizedexperiment import RESummarizedExperiment
from .rmatrixadapter import RMatrixAdapter
from .r_init import initialize_r, check_r_initialized, is_r_initialized, get_rmat
from .r_utils import (
    ensure_r_dependencies,
    check_renv,
    is_renv_installed,
    activate_renv,
    has_renv,
    create_renv,
    install_base_dependencies,
    BASE_R_PACKAGES,
)
from .volcano_plot import volcano_plot

__all__ = [
    "RESummarizedExperiment",
    "RMatrixAdapter",
    "initialize_r",
    "check_r_initialized",
    "is_r_initialized",
    "get_rmat",
    "ensure_r_dependencies",
    "check_renv",
    "is_renv_installed",
    "activate_renv",
    "has_renv",
    "create_renv",
    "install_base_dependencies",
    "BASE_R_PACKAGES",
    "volcano_plot",
    # Lazy-loaded submodules
    "edger",
    "limma", 
    "sva",
]

# Submodules to be lazily loaded (PEP 810 style)
_LAZY_SUBMODULES = {"edger", "limma", "sva"}


def __getattr__(name: str):
    """Lazy loading of submodules per PEP 562/810."""
    if name in _LAZY_SUBMODULES:
        # Import the submodule on first access
        module = importlib.import_module(f".{name}", __name__)
        # Cache it in the module globals
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy submodules in dir() output."""
    return list(__all__)