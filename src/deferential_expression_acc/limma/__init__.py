"""Limma: Linear Models for Microarray and RNA-Seq Data.

This module provides Python wrappers for the R limma package, enabling
differential expression analysis with proper R-backing via rpy2.
"""

from .voom import voom
from .voom_with_quality_weights import voom_with_quality_weights
from .normalize_between_arrays import normalize_between_arrays
from .remove_batch_effect import remove_batch_effect
from .lm_fit import lm_fit, LimmaModel
from .contrasts_fit import contrasts_fit
from .e_bayes import e_bayes
from .top_table import top_table
from .decide_tests import decide_tests
from .treat import treat
from .duplicate_correlation import duplicate_correlation
from .utils import _limma

# Register Limma accessor on RESummarizedExperiment
# This happens when the module is imported
from .accessor import activate, LimmaAccessor
activate()

__all__ = [
    "voom",
    "voom_with_quality_weights",
    "normalize_between_arrays",
    "remove_batch_effect",
    "lm_fit",
    "LimmaModel",
    "contrasts_fit",
    "e_bayes",
    "top_table",
    "decide_tests",
    "treat",
    "duplicate_correlation",
    "_limma",
    "LimmaAccessor",
]

