from .calc_norm_factors import calc_norm_factors
from .glm_ql_fit import glm_ql_fit
from .glm_ql_ftest import glm_ql_ftest
from .cpm import cpm
from .filter_by_expr import filter_by_expr
from .estimate_disp import estimate_disp
from .top_tags import top_tags
from .EdgeR import EdgeR

# Register EdgeR accessor on RESummarizedExperiment
# This happens when the module is imported
from .accessor import activate, EdgeRAccessor
activate()

__all__ = [
    "calc_norm_factors",
    "glm_ql_fit",
    "glm_ql_ftest",
    "cpm",
    "filter_by_expr",
    "estimate_disp",
    "top_tags",
    "EdgeR",
    "EdgeRAccessor",
]