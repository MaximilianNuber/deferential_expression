"""
Fit linear model using limma::lmFit.

This module provides the LimmaModel dataclass for storing fit results
and the lm_fit function for fitting the model.
"""

from __future__ import annotations
from typing import Any, Dict, Literal, Optional, Sequence, TypeVar, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .utils import _limma
from .checks import check_se, check_assay_exists, check_r_assay, check_design

# Type variable for SummarizedExperiment variants
SE = TypeVar("SE")


@dataclass
class LimmaModel:
    """Container for limma linear model fit results.
    
    This dataclass stores the R fit objects and associated metadata from
    lm_fit. Use with e_bayes(), top_table(), etc. for downstream analysis.
    
    Attributes:
        sample_names: Sample names (column names) from the input SE.
        feature_names: Feature names (row names) from the input SE.
        lm_fit: R object from lmFit.
        design: Design matrix used for fitting.
        contrast_fit: R object from contrasts.fit (optional).
        ebayes: R object from eBayes (optional).
        ndups: Number of technical replicates.
        method: Fitting method used.
        metadata: Additional metadata.
    """
    sample_names: Optional[Sequence[str]] = None
    feature_names: Optional[Sequence[str]] = None
    lm_fit: Optional[Any] = None
    design: Optional[pd.DataFrame] = None
    contrast_fit: Optional[Any] = None
    ebayes: Optional[Any] = None
    ndups: Optional[int] = None
    method: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def e_bayes(
        self,
        proportion: float = 0.01,
        trend: bool = False,
        robust: bool = False,
        **kwargs: Any
    ) -> "LimmaModel":
        """
        Apply empirical Bayes moderation.
        
        Convenience method that delegates to the e_bayes function.
        
        Returns:
            LimmaModel with ebayes slot set.
        """
        from .e_bayes import e_bayes as _e_bayes
        return _e_bayes(self, proportion=proportion, trend=trend, robust=robust, **kwargs)

    def contrasts_fit(
        self,
        contrast: Sequence[Union[int, float]],
    ) -> "LimmaModel":
        """
        Apply contrast to fitted model.
        
        Convenience method that delegates to the contrasts_fit function.
        
        Returns:
            LimmaModel with contrast_fit slot set.
        """
        from .contrasts_fit import contrasts_fit as _contrasts_fit
        return _contrasts_fit(self, contrast=contrast)

    def top_table(
        self,
        coef: Optional[Union[int, str]] = None,
        n: Optional[int] = None,
        adjust_method: str = "BH",
        sort_by: str = "PValue",
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Extract top-ranked genes.
        
        Convenience method that delegates to the top_table function.
        
        Returns:
            pd.DataFrame with DE results.
        """
        from .top_table import top_table as _top_table
        return _top_table(self, coef=coef, n=n, adjust_method=adjust_method, sort_by=sort_by, **kwargs)

    def decide_tests(
        self,
        method: str = "separate",
        adjust_method: str = "BH",
        p_value: float = 0.05,
        lfc: float = 0,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Classify genes as up/down/not significant.
        
        Convenience method that delegates to the decide_tests function.
        
        Returns:
            pd.DataFrame with -1 (down), 0 (not sig), 1 (up).
        """
        from .decide_tests import decide_tests as _decide_tests
        return _decide_tests(self, method=method, adjust_method=adjust_method, p_value=p_value, lfc=lfc, **kwargs)

    def treat(
        self,
        lfc: float = 1.0,
        robust: bool = False,
        trend: bool = False,
        **kwargs: Any
    ) -> "LimmaModel":
        """
        Apply TREAT (fold-change threshold testing).
        
        Convenience method that delegates to the treat function.
        
        Returns:
            LimmaModel with TREAT fit.
        """
        from .treat import treat as _treat
        return _treat(self, lfc=lfc, robust=robust, trend=trend, **kwargs)


def lm_fit(
    se: SE,
    design: pd.DataFrame,
    assay: str = "log_expr",
    ndups: Optional[int] = None,
    method: Literal["ls", "robust"] = "ls",
    **kwargs: Any,
) -> LimmaModel:
    """
    Fit linear model to expression data using limma::lmFit.
    
    Works with any BiocPy SummarizedExperiment variant (SE, RSE, SCE).
    The assay must be R-initialized using initialize_r() first.
    
    Args:
        se: Input SummarizedExperiment with R-initialized expression assay.
        design: Design matrix (samples × covariates) as pandas DataFrame.
        assay: Expression assay to use. Default: "log_expr".
        ndups: Number of technical replicates. Default: None.
        method: Fitting method ("ls" or "robust"). Default: "ls".
        **kwargs: Additional args forwarded to R function.
    
    Returns:
        LimmaModel: Container with fitted model.
    
    Raises:
        TypeError: If inputs are invalid.
        KeyError: If assay doesn't exist.
    
    Example:
        >>> from deferential_expression import initialize_r
        >>> import deferential_expression.limma as limma
        >>> import pandas as pd
        >>> se = initialize_r(se, assay="log_expr")
        >>> design = pd.DataFrame({'Intercept': [1]*6, 'Condition': [0,0,0,1,1,1]})
        >>> model = limma.lm_fit(se, design)
        >>> results = model.e_bayes().top_table()
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from ..edger.utils import pandas_to_r_matrix
    from ..r_init import get_rmat, is_r_initialized
    
    # Validate inputs
    check_se(se)
    check_assay_exists(se, assay)
    check_r_assay(se, assay)
    n_samples = len(se.column_names) if se.column_names else se.shape[1]
    check_design(design, n_samples)
    
    r = get_r_environment()
    limma_pkg = _limma()
    
    exprs_r = get_rmat(se, assay)
    design_r = pandas_to_r_matrix(design)
    
    # Handle weights
    if "weights" in se.assay_names and is_r_initialized(se, "weights"):
        weights = get_rmat(se, "weights")
    else:
        weights = r.ro.NULL
    
    ndups_r = r.ro.NULL if ndups is None else ndups
    
    fit = limma_pkg.lmFit(
        exprs_r,
        design_r,
        weights=weights,
        method=method,
        **kwargs,
    )
    
    return LimmaModel(
        sample_names=se.column_names,
        feature_names=se.row_names,
        lm_fit=fit,
        design=design,
        ndups=ndups,
        method=method,
    )
