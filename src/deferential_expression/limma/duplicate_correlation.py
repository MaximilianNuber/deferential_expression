from typing import Any, Union, Sequence

import numpy as np
import pandas as pd

from bioc2ri.lazy_r_env import get_r_environment
from ..resummarizedexperiment import RESummarizedExperiment
from .utils import _limma
from ..edger.utils import pandas_to_r_matrix


def duplicate_correlation(
    se: RESummarizedExperiment,
    design: pd.DataFrame,
    block: Union[pd.Series, Sequence, np.ndarray, pd.Categorical],
    exprs_assay: str = "log_expr",
    **kwargs: Any
) -> float:
    """Estimate correlation between duplicate spots or technical replicates.

    Wraps the R ``limma::duplicateCorrelation`` function to estimate the
    intra-block correlation for use with ``voom()`` or ``lmFit()``.

    Args:
        se: Input ``RESummarizedExperiment`` containing an expression assay.
        design: Design matrix (samples × covariates) as a pandas DataFrame.
        block: Blocking factor indicating which samples are related (e.g., technical
            replicates, repeated measures from same individual). Can be array-like
            or pandas Categorical.
        exprs_assay: Name of the expression assay to use. Default: ``"log_expr"``.
        **kwargs: Additional keyword arguments forwarded to ``limma::duplicateCorrelation``.

    Returns:
        float: Estimated consensus correlation between technical replicates.

    Notes:
        - Use the returned correlation as the ``block`` parameter in ``voom()``.
        - For RNA-seq, typically run after an initial ``voom()`` transformation.
        - The correlation is a consensus value across all genes.

    Examples:
        >>> # First voom without blocking
        >>> se_voom = voom(se, design=design_df)
        >>> # Estimate correlation
        >>> cor = duplicate_correlation(se_voom, design=design_df, block=batch_labels)
        >>> # Re-run voom with blocking
        >>> se_voom2 = voom(se, design=design_df, block=batch_labels)
    """
    r = get_r_environment()
    limma_pkg = _limma()

    exprs_r = se.assay_r(exprs_assay)
    design_r = pandas_to_r_matrix(design)

    # Convert block to R
    if isinstance(block, pd.Categorical):
        with r.localconverter(r.default_converter + r.pandas2ri.converter):
            block_r = r.get_conversion().py2rpy(block)
    else:
        block_arr = np.asarray(block, dtype=str)
        block_r = r.StrVector(block_arr)

    # Get weights if available
    if "weights" in se.assay_names:
        weights = se.assay_r("weights")
    else:
        weights = r.ro.NULL

    # Call duplicateCorrelation
    dupcor_result = limma_pkg.duplicateCorrelation(
        exprs_r,
        design_r,
        block=block_r,
        weights=weights,
        **kwargs
    )

    # Extract consensus correlation
    consensus_cor = float(r.r2py(dupcor_result.rx2("consensus.correlation")))

    return consensus_cor
