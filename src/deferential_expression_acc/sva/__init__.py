"""SVA: Surrogate Variable Analysis.

This module provides Python wrappers for the R sva package, enabling
batch correction and surrogate variable analysis via rpy2.

Functions:
    - ComBat: Batch correction for continuous/normalized expression data
    - ComBat_seq: Batch correction for count data
    - sva: Surrogate variable analysis to capture hidden variation
"""

# Register SVA accessor on RESummarizedExperiment
from .accessor import activate, SVAAccessor

activate()

__all__ = [
    "SVAAccessor",
]
