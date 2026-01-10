"""SVA: Surrogate Variable Analysis.

This module provides Python wrappers for the R sva package, enabling
batch correction and surrogate variable analysis with proper R-backing via rpy2.

Functional API:
    >>> import deferential_expression.sva as sva
    >>> rse_combat = sva.combat(rse, batch="batch_col")
    >>> rse_combat = sva.combat_seq(rse, batch="batch_col")
    >>> rse_sva = sva.sva(rse, mod=design)
    >>> sv_df = sva.get_sv(rse_sva)

Accessor API:
    >>> import deferential_expression.sva
    >>> rse_combat = rse.sva.combat(batch="batch_col")
    >>> sv_df = rse.sva.get_sv()
"""

# Check/install sva R package on module import
from ..r_utils import ensure_r_dependencies
ensure_r_dependencies(["sva"])

# Functional API exports
from .combat import combat
from .combat_seq import combat_seq
from .sva_func import sva
from .get_sv import get_sv
from .utils import _prep_sva, _sva

# Register SVA accessor on RESummarizedExperiment
from .accessor import activate, SVAAccessor
activate()

__all__ = [
    # Functional API
    "combat",
    "combat_seq",
    "sva",
    "get_sv",
    # Accessor
    "SVAAccessor",
    # Utilities
    "_prep_sva",
    "_sva",
]
