from typing import Any, Optional, Sequence, Union
from deferential_expression.rpy2_manager import Rpy2Manager
from .edger_func import _RConverters
from deferential_expression.edger_func import EdgeRState

import pandas as pd
import numpy as np
from anndata import AnnData


# ===================== LIMMA FUNCTIONAL PIPELINE ===================== #
# Reuses Rpy2Manager instance `_rmana` and _RConverters above.

_rmana = Rpy2Manager()

# Lazy-load limma once
_limma_pkg = None

def _limma():
    global _limma_pkg
    if _limma_pkg is None:
        _limma_pkg = _rmana.importr("limma")
    return _limma_pkg

from dataclasses import dataclass, field, replace
from typing import Tuple

@dataclass(frozen=True, slots=True)
class LimmaState:
    # Core R-side objects
    exprs: Any | None = None         # matrix or EList (after voom)
    voom_obj: Any | None = None      # voom/voomWithQualityWeights return (EList)
    fit: Any | None = None           # lmFit result
    fit_contrasts: Any | None = None # contrasts.fit result
    fit_ebayes: Any | None = None    # eBayes result
    top_table_r: Any | None = None   # R data.frame from topTable

    # R descriptors
    design_r: Any | None = None
    contrast_r: Any | None = None

    # Python side artifacts
    weights: Optional[pd.DataFrame] = None   # voom weights
    normalized_exprs: Optional[pd.DataFrame] = None
    batch_corrected_exprs: Optional[pd.DataFrame] = None

    meta: dict[str, Any] = field(default_factory=dict)

    # ----------- Getters ----------- #
    def get_exprs_df(self) -> Optional[pd.DataFrame]:
        if self.exprs is None:
            return None
        r = _rmana
        # If it's an EList, pull E slot
        robj = self.exprs.rx2('E') if 'E' in list(self.exprs.names) else self.exprs
        return _RConverters.rmatrix_to_df(robj)
    
    def get_exprs_anndata(self) -> Optional[AnnData]:
        if self.exprs is None:
            return None
        r = _rmana
        # If it's an EList, pull E slot
        robj = self.exprs.rx2('E') if 'E' in list(self.exprs.names) else self.exprs
        return _RConverters.rmatrix_to_anndata(robj, obs_names=self.exprs.rx2('colnames'), var_names=self.exprs.rx2('rownames'))

    def get_top_table(self) -> Optional[pd.DataFrame]:
        if self.top_table_r is None:
            return None
        r = _rmana
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            return r.get_conversion().rpy2py(self.top_table_r)

# --------------- Constructors ---------------- #

def limma_state_from_anndata(adata: AnnData, layer: Optional[str] = None) -> LimmaState:
    exprs_r = _RConverters.anndata_to_r_matrix(adata, layer)
    return LimmaState(exprs=exprs_r)

def limma_state_from_dfs(exprs: pd.DataFrame) -> LimmaState:
    exprs_r = _RConverters.df_to_r_matrix(exprs)
    return LimmaState(exprs=exprs_r)

def limma_state_from_numpy(exprs: np.ndarray, rownames: Sequence[str], colnames: Sequence[str]) -> LimmaState:
    exprs_r = _RConverters.np_to_r_matrix(exprs, rownames=rownames, colnames=colnames)
    return LimmaState(exprs=exprs_r)

# --------------- Helpers ---------------- #

def _extract_E(exprs_obj: Any) -> Any:
    """Return expression matrix from EList or matrix R object."""
    if hasattr(exprs_obj, 'names') and 'E' in list(exprs_obj.names):
        return exprs_obj.rx2('E')
    return exprs_obj

def _replace_E(exprs_obj: Any, new_E: Any) -> Any:
    """Return a new EList with E replaced (if EList), else return new_E."""
    r = _rmana
    if hasattr(exprs_obj, 'names') and 'E' in list(exprs_obj.names):
        # rebuild list
        new_list = {name: (new_E if name == 'E' else exprs_obj.rx2(name)) for name in exprs_obj.names}
        return r.ro.baseenv['list'](**new_list)
    return new_E

# --------------- Functional steps ---------------- #

def voom_step(state: LimmaState | EdgeRState, design_df: pd.DataFrame, plot: bool = False, **kwargs) -> LimmaState:
    r = _rmana
    limma = _limma()
    if isinstance(state, LimmaState):
        design_r = _RConverters.df_to_r_matrix(design_df)
        voom_obj = limma.voom(state.exprs, design=design_r, plot=plot, **kwargs)

        # extract weights to pandas
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            w_df = r.get_conversion().rpy2py(voom_obj.rx2('weights'))
        return replace(state, voom_obj=voom_obj, exprs=voom_obj, design_r=design_r, weights=w_df)

    elif isinstance(state, EdgeRState):
        # EdgeRState has exprs as DGEList, convert to matrix
        design_r = _RConverters.df_to_r_matrix(design_df)
        # E = r.ro.baseeenv["[["](state.dge, "counts")
        voom_obj = limma.voom(state.dge, design=design_r, plot=plot, **kwargs)

        # extract weights to pandas
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            w_df = r.get_conversion().rpy2py(voom_obj.rx2('weights'))
        return LimmaState(voom_obj=voom_obj, exprs=r.ro.baseenv["[["](voom_obj, "E"), design_r=design_r, weights=w_df)

    

def voom_qw_step(state: LimmaState, design_df: pd.DataFrame, plot: bool = False, **kwargs) -> LimmaState:
    r = _rmana
    limma = _limma()
    design_r = _RConverters.df_to_r_matrix(design_df)
    voom_obj = limma.voomWithQualityWeights(state.exprs, design=design_r, plot=plot, **kwargs)
    with r.localconverter(r.default_converter + r.pandas2ri_converter):
        w_df = r.get_conversion().rpy2py(voom_obj.rx2('weights'))
    return replace(state, voom_obj=voom_obj, exprs=voom_obj, design_r=design_r, weights=w_df)

def normalize_between_arrays_step(state: LimmaState, method: str = 'quantile', **kwargs) -> LimmaState:
    limma = _limma()
    E = _extract_E(state.exprs)
    E_norm = limma.normalizeBetweenArrays(E, method=method, **kwargs)
    exprs_new = _replace_E(state.exprs, E_norm)
    # python copy
    r = _rmana
    with r.localconverter(r.default_converter + r.pandas2ri_converter):
        norm_df = r.get_conversion().rpy2py(E_norm)
    return replace(state, exprs=exprs_new, normalized_exprs=norm_df)

def remove_batch_effect_step(state: LimmaState, batch: Union[pd.Series, Sequence, np.ndarray],
                             design_df: Optional[pd.DataFrame] = None, **kwargs) -> LimmaState:
    r = _rmana
    limma = _limma()
    # batch to R
    if isinstance(batch, pd.Series):
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            batch_r = r.get_conversion().py2rpy(batch)
    else:
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            batch_r = r.get_conversion().py2rpy(pd.Series(batch))
    design_r = _RConverters.df_to_r_matrix(design_df) if design_df is not None else r.ro.NULL

    E = _extract_E(state.exprs)
    E_corr = limma.removeBatchEffect(E, batch=batch_r, design=design_r, **kwargs)
    exprs_new = _replace_E(state.exprs, E_corr)

    with r.localconverter(r.default_converter + r.pandas2ri_converter):
        corr_df = r.get_conversion().rpy2py(E_corr)
    return replace(state, exprs=exprs_new, batch_corrected_exprs=corr_df)

def lm_fit_step(state: LimmaState, design_df: pd.DataFrame, **kwargs) -> LimmaState:
    limma = _limma()
    design_r = _RConverters.df_to_r_matrix(design_df)
    if state.voom_obj is not None:
        # If voom was run, use voom_object
        fit = limma.lmFit(state.voom_obj, design=design_r, **kwargs)
    else:
        # Otherwise use exprs directly
        fit = limma.lmFit(state.exprs, design=design_r, **kwargs)
    return replace(state, fit=fit, design_r=design_r)

def contrasts_fit_step(state: LimmaState, contrast: Union[pd.DataFrame, np.ndarray, Sequence[float]], **kwargs) -> LimmaState:
    limma = _limma()
    # build contrast matrix
    if isinstance(contrast, pd.DataFrame):
        contrast_r = _RConverters.df_to_r_matrix(contrast)
    else:
        arr = np.asarray(contrast, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        contrast_r = _RConverters.df_to_r_matrix(pd.DataFrame(arr))
    fit2 = limma.contrasts_fit(state.fit, contrast_r, **kwargs)
    return replace(state, fit_contrasts=fit2, contrast_r=contrast_r)

def e_bayes_step(state: LimmaState, robust: bool = True, trend: bool = False, **kwargs) -> LimmaState:
    limma = _limma()
    fit_input = state.fit_contrasts or state.fit
    fit_eb = limma.eBayes(fit_input, robust=robust, trend=trend, **kwargs)
    return replace(state, fit_ebayes=fit_eb)

def top_table_step(state: LimmaState, coef: Optional[Union[int, str]] = None, number: Optional[int] = None,
                   adjust_method: str = 'BH', sort_by: str = 'P', **kwargs) -> Tuple[LimmaState, pd.DataFrame]:
    limma = _limma()
    r = _rmana
    if number is None:
        number_r = r.ro.r('Inf')
    else:
        number_r = int(number)
    
    assert state.fit_ebayes is not None
    fit_input = state.fit_ebayes
    coef = coef if coef is not None else r.ro.NULL
    tt = limma.topTable(fit_input, coef=coef, number=number_r,
                        adjust_method=adjust_method, sort_by=sort_by, **kwargs)
    with r.localconverter(r.default_converter + r.pandas2ri_converter):
        df = r.get_conversion().rpy2py(tt)
    df = df.reset_index().rename(columns={'index':'gene', 'P.Value':'p_value', 'adj.P.Val':'adj_p_value', 'logFC':'log_fc'})
    return replace(state, top_table_r=tt), df

# Optional duplicateCorrelation step

def duplicate_correlation_step(state: LimmaState, design_df: pd.DataFrame, block: Sequence, **kwargs) -> LimmaState:
    limma = _limma()
    r = _rmana
    design_r = _RConverters.df_to_r_matrix(design_df)
    with r.localconverter(r.default_converter + r.pandas2ri_converter):
        block_r = r.get_conversion().py2rpy(pd.Series(block))
    dup = limma.duplicateCorrelation(state.exprs, design=design_r, block=block_r, **kwargs)
    # store correlation in meta
    meta_new = dict(state.meta)
    meta_new['duplicate_correlation'] = dup.rx2('consensus.correlation')[0]
    return replace(state, meta=meta_new)

# Tiny pipeline helper reused

def pipe_limma(state: LimmaState, *funcs):
    for f in funcs:
        state = f(state)
    return state


# -------------------- LIMMA fluent API -------------------- #

def _limma_voom(self: LimmaState, design_df: pd.DataFrame, plot: bool = False, **kwargs) -> LimmaState:
    return voom_step(self, design_df, plot=plot, **kwargs)

def _limma_voom_qw(self: LimmaState, design_df: pd.DataFrame, plot: bool = False, **kwargs) -> LimmaState:
    return voom_qw_step(self, design_df, plot=plot, **kwargs)

def _limma_normalize(self: LimmaState, method: str = 'quantile', **kwargs) -> LimmaState:
    return normalize_between_arrays_step(self, method=method, **kwargs)

def _limma_remove_batch(self: LimmaState, batch: Union[pd.Series, Sequence, np.ndarray], design_df: Optional[pd.DataFrame] = None, **kwargs) -> LimmaState:
    return remove_batch_effect_step(self, batch=batch, design_df=design_df, **kwargs)

def _limma_lm_fit(self: LimmaState, design_df: pd.DataFrame, **kwargs) -> LimmaState:
    return lm_fit_step(self, design_df, **kwargs)

def _limma_contrasts_fit(self: LimmaState, contrast: Union[pd.DataFrame, np.ndarray, Sequence[float]], **kwargs) -> LimmaState:
    return contrasts_fit_step(self, contrast, **kwargs)

def _limma_e_bayes(self: LimmaState, robust: bool = True, trend: bool = False, **kwargs) -> LimmaState:
    return e_bayes_step(self, robust=robust, trend=trend, **kwargs)

def _limma_top_table(self: LimmaState, coef: Optional[Union[int, str]] = None, number: Optional[int] = None,
                     adjust_method: str = 'BH', sort_by: str = 'P', **kwargs) -> Tuple[LimmaState, pd.DataFrame]:
    return top_table_step(self, coef=coef, number=number, adjust_method=adjust_method, sort_by=sort_by, **kwargs)

def _limma_dup_corr(self: LimmaState, design_df: pd.DataFrame, block: Sequence, **kwargs) -> LimmaState:
    return duplicate_correlation_step(self, design_df, block, **kwargs)

# attach methods
LimmaState.voom = _limma_voom                                 # type: ignore[attr-defined]
LimmaState.voom_qw = _limma_voom_qw                           # type: ignore[attr-defined]
LimmaState.normalize_between_arrays = _limma_normalize        # type: ignore[attr-defined]
LimmaState.remove_batch_effect = _limma_remove_batch          # type: ignore[attr-defined]
LimmaState.lm_fit = _limma_lm_fit                             # type: ignore[attr-defined]
LimmaState.contrasts_fit = _limma_contrasts_fit               # type: ignore[attr-defined]
LimmaState.e_bayes = _limma_e_bayes                           # type: ignore[attr-defined]
LimmaState.top_table = _limma_top_table                       # type: ignore[attr-defined]
LimmaState.duplicate_correlation = _limma_dup_corr            # type: ignore[attr-defined]