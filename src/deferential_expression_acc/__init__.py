# Note: edger and limma modules are NOT imported by default.
# Users must explicitly import them to register the accessors:
#   import deferential_expression_acc.edger
#   import deferential_expression_acc.limma

from .resummarizedexperiment import RESummarizedExperiment
from .r_utils import ensure_r_dependencies
from .volcano_plot import volcano_plot

__all__ = [
    "RESummarizedExperiment",
    "ensure_r_dependencies",
    "volcano_plot",
]