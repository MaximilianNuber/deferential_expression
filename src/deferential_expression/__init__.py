# deferential_expression/__init__.py
from types import SimpleNamespace

# --- Limma namespace ---
from .limma_RESummarizedExperiment import (
    Limma,
    voom,
    voom_quality_weights,
    normalize_between_arrays,
    remove_batch_effect,
    lm_fit,
    contrasts_fit,
    top_table,
)

limma = SimpleNamespace(
    Limma=Limma,
    voom=voom,
    voom_quality_weights=voom_quality_weights,
    normalize_between_arrays=normalize_between_arrays,
    remove_batch_effect=remove_batch_effect,
    lm_fit=lm_fit,
    contrasts_fit=contrasts_fit,
    top_table=top_table,
)

# --- edgeR namespace ---
from .edger_RESummarizedExperiment import (
    EdgeR,
    RESummarizedExperiment,
    RMatrixAdapter,
    filter_by_expr,
    calc_norm_factors,
    cpm,
    glm_ql_fit,
    glm_ql_ftest,
)

edger = SimpleNamespace(
    EdgeR=EdgeR,
    filter_by_expr=filter_by_expr,
    calc_norm_factors=calc_norm_factors,
    cpm=cpm,
    glm_ql_fit=glm_ql_fit,
    glm_ql_ftest=glm_ql_ftest,
)

import deferential_expression.edger_RESummarizedExperiment as edg

__all__ = ["limma", "edger", "RESummarizedExperiment", "RMatrixAdapter", "edg"]