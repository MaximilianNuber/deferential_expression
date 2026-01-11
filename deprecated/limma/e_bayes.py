from dataclasses import replace
from typing import Any, Optional, Tuple

from bioc2ri.lazy_r_env import get_r_environment
from .lm_fit import LimmaModel
from .utils import _limma


def e_bayes(
    lm_obj: LimmaModel,
    proportion: float = 0.01,
    stdev_coef_lim: Optional[Tuple[float, float]] = None,
    trend: bool = False,
    robust: bool = False,
    winsor_tail_p: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> LimmaModel:
    """Compute empirical Bayes moderated statistics for differential expression.

    Wraps the R ``limma::eBayes`` function to compute moderated t-statistics,
    moderated F-statistics, and log-odds of differential expression by empirical
    Bayes moderation of standard errors.

    Args:
        lm_obj: ``LimmaModel`` instance with either ``lm_fit`` or ``contrast_fit`` set.
        proportion: Assumed proportion of genes that are differentially expressed.
            Used for computing the B-statistic (log-odds). Default: 0.01.
        stdev_coef_lim: Optional tuple (lower, upper) specifying limits for standard
            deviation coefficients. If ``None``, computed automatically.
        trend: If ``True``, fits a mean-variance trend to the standard deviations.
            Useful for RNA-seq count data. Default: ``False``.
        robust: If ``True``, uses robust empirical Bayes to protect against outliers.
            Default: ``False``.
        winsor_tail_p: Optional tuple (lower, upper) of tail probabilities for Winsorizing.
            Only used when ``robust=True``.
        **kwargs: Additional keyword arguments forwarded to ``limma::eBayes``.

    Returns:
        LimmaModel: New instance with the ``ebayes`` slot containing the R object
            returned by ``limma::eBayes``.

    Raises:
        AssertionError: If neither ``lm_fit`` nor ``contrast_fit`` is set.

    Notes:
        - Use the fit object from ``contrast_fit`` if available, otherwise ``lm_fit``.
        - The returned object can be passed to ``top_table()`` or ``decide_tests()``.

    Examples:
        >>> lm = lm_fit(se, design=design_df)
        >>> lm_eb = e_bayes(lm, robust=True)
        >>> results = top_table(lm_eb)
    """
    assert isinstance(lm_obj, LimmaModel), "lm_obj must be a LimmaModel instance"
    
    r_fit = lm_obj.contrast_fit if lm_obj.contrast_fit is not None else lm_obj.lm_fit
    assert r_fit is not None, "lm_fit or contrast_fit must be set in the LimmaModel instance"

    r = get_r_environment()
    limma_pkg = _limma()

    # Prepare optional arguments
    call_kwargs: dict[str, Any] = {"proportion": proportion, "trend": trend, "robust": robust}
    
    if stdev_coef_lim is not None:
        call_kwargs["stdev.coef.lim"] = r.FloatVector(stdev_coef_lim)
    
    if winsor_tail_p is not None:
        call_kwargs["winsor.tail.p"] = r.FloatVector(winsor_tail_p)
    
    call_kwargs.update(kwargs)

    eb = limma_pkg.eBayes(r_fit, **call_kwargs)
    
    return replace(lm_obj, ebayes=eb)
