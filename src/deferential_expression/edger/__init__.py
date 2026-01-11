"""EdgeR: Analysis of Differential Expression in Genomic Data.

This module provides Python wrappers for the R edgeR package, enabling
differential expression analysis with proper R-backing via rpy2.

Functional API:
    >>> import deferential_expression.edger as edger
    >>> rse_norm = edger.calc_norm_factors(rse, method="TMM")
    >>> mask = edger.filter_by_expr(rse, min_count=10)
    >>> model = edger.glm_ql_fit(rse, design)
    >>> results = edger.glm_ql_ftest(model, coef=2)

Accessor API:
    >>> import deferential_expression.edger
    >>> rse_norm = rse.edger.calc_norm_factors(method="TMM")
    >>> model = rse.edger.glm_ql_fit(design)
"""

# Check/install edgeR R package on module import
from ..r_utils import ensure_r_dependencies
ensure_r_dependencies(["edgeR"])

# Functional API exports
from .calc_norm_factors import calc_norm_factors
from .cpm import cpm
from .filter_by_expr import filter_by_expr
from .glm_ql_fit import glm_ql_fit, EdgeRModel, GlmQlFitConfig
from .glm_ql_ftest import glm_ql_ftest
from .top_tags import top_tags
from .utils import _prep_edger

# Register EdgeR accessor on RESummarizedExperiment
from .accessor import activate, EdgeRAccessor
activate()

__all__ = [
    # Functional API
    "calc_norm_factors",
    "cpm",
    "filter_by_expr",
    "glm_ql_fit",
    "glm_ql_ftest",
    "top_tags",
    # Model classes
    "EdgeRModel",
    "GlmQlFitConfig",
    # Accessor
    "EdgeRAccessor",
    # Utilities
    "_prep_edger",
]