"""Limma: Linear Models for Microarray and RNA-Seq Data.

This module provides Python wrappers for the R limma package, enabling
differential expression analysis with proper R-backing via rpy2.

Functional API:
    >>> import deferential_expression.limma as limma
    >>> rse_voom = limma.voom(rse, design)
    >>> model = limma.lm_fit(rse_voom, design)
    >>> results = model.e_bayes().top_table()

Accessor API:
    >>> import deferential_expression.limma
    >>> rse_voom = rse.limma.voom(design)
    >>> model = rse_voom.limma.lm_fit(design)
"""

# Check/install limma R package on module import
from ..r_utils import ensure_r_dependencies
ensure_r_dependencies(["limma"])

# Functional API exports
from .voom import voom
from .normalize_between_arrays import normalize_between_arrays
from .remove_batch_effect import remove_batch_effect
from .lm_fit import lm_fit, LimmaModel
from .contrasts_fit import contrasts_fit
from .e_bayes import e_bayes
from .top_table import top_table
from .decide_tests import decide_tests
from .treat import treat
from .utils import _limma

# Register Limma accessor on RESummarizedExperiment
from .accessor import activate, LimmaAccessor
activate()

__all__ = [
    # Functional API
    "voom",
    "normalize_between_arrays",
    "remove_batch_effect",
    "lm_fit",
    "contrasts_fit",
    "e_bayes",
    "top_table",
    "decide_tests",
    "treat",
    # Model classes
    "LimmaModel",
    # Accessor
    "LimmaAccessor",
    # Utilities
    "_limma",
]
