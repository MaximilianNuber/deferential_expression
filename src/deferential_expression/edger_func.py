from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Union, Tuple

import numpy as np
import pandas as pd

try:
    from anndata import AnnData
except ImportError:  # optional
    AnnData = None  # type: ignore

# ------------------------------------------------------------------
#  Rpy2Manager – assumed to exist in your code base. Import here.
# ------------------------------------------------------------------
from deferential_expression.rpy2_manager import Rpy2Manager  # your singleton
_rmana = Rpy2Manager()

# Lazy-load edgeR once
_edgeR_pkg = None

def _edgeR():
    global _edgeR_pkg
    if _edgeR_pkg is None:
        _edgeR_pkg = _rmana.importr("edgeR")
    return _edgeR_pkg

# ------------------------------------------------------------------
#  Converters
# ------------------------------------------------------------------
class _RConverters:
    @staticmethod
    def df_to_r_matrix(df: pd.DataFrame) -> Any:
        r = _rmana
        with r.localconverter(r.default_converter + r.numpy2ri_converter):
            r_mat = r.get_conversion().py2rpy(df.values)
        # dimnames
        r_mat = r.ro.baseenv["rownames<-"](r_mat, r.StrVector(df.index.astype(str).to_numpy()))
        r_mat = r.ro.baseenv["colnames<-"](r_mat, r.StrVector(df.columns.astype(str).to_numpy()))
        return r_mat

    @staticmethod
    def df_to_r_df(df: pd.DataFrame) -> Any:
        r = _rmana
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            return r.get_conversion().py2rpy(df)

    @staticmethod
    def np_to_r_matrix(arr: np.ndarray, rownames: Optional[Sequence[str]] = None,
                       colnames: Optional[Sequence[str]] = None) -> Any:
        r = _rmana
        with r.localconverter(r.default_converter + r.numpy2ri_converter):
            r_mat = r.get_conversion().py2rpy(arr)
        if rownames is not None:
            r_mat = r.ro.baseenv["rownames<-"](r_mat, r.StrVector(list(map(str, rownames))))
        if colnames is not None:
            r_mat = r.ro.baseenv["colnames<-"](r_mat, r.StrVector(list(map(str, colnames))))
        return r_mat

    @staticmethod
    def anndata_to_r_matrix(adata: "AnnData", layer: Optional[str] = None) -> Any:
        r = _rmana
        mat = adata.X if layer is None else adata.layers[layer]
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        mat = np.asarray(mat).T  # genes x samples
        return _RConverters.np_to_r_matrix(mat, rownames=adata.var_names, colnames=adata.obs_names)
    
    @staticmethod
    def rmatrix_to_df(r_mat: Any) -> pd.DataFrame:
        r = _rmana
        with r.localconverter(r.default_converter + r.numpy2ri_converter):
            pymat = r.get_conversion().rpy2py(r_mat)
        df = pd.DataFrame(pymat)
        # set rownames and colnames

        colnames = np.asarray(
            r.ro.baseenv["colnames"](r_mat), dtype=str
        )
        rownames = np.asarray(
            r.ro.baseenv["rownames"](r_mat), dtype=str
        )
        if colnames is not None:
            df.columns = colnames
        if rownames is not None:
            df.index = rownames

        return df
    
    @staticmethod
    def rmatrix_to_numpy(r_mat: Any) -> np.ndarray:
        r = _rmana
        with r.localconverter(r.default_converter + r.numpy2ri_converter):
            return r.get_conversion().rpy2py(r_mat)

    @staticmethod
    def rmatrix_to_anndata(r_mat: Any) -> AnnData:
        r = _rmana
        df = _RConverters.rmatrix_to_numpy(r_mat)
        rcolnames = r.ro.baseenv["colnames"](r_mat)
        rrownames = r.ro.baseenv["rownames"](r_mat)

        print(np.asarray(rcolnames))
        print(isinstance(np.asarray(rcolnames), type(r.ro.NULL)))

        colnames = None if isinstance((rcolnames), type(r.ro.NULL)) else np.asarray(rcolnames)
        rownames = None if isinstance((rrownames), type(r.ro.NULL)) else np.asarray(rrownames)

        adata = AnnData(df.T)
        print(colnames)
        print(rownames)
        if colnames is not None:
            adata.var_names = rownames
        if rownames is not None:
            adata.obs_names = colnames

        return adata

# ------------------------------------------------------------------
#  EdgeRState dataclass (immutable, slots, getters)
# ------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EdgeRState:
    # Core R objects
    dge: Any | None = None        # edgeR::DGEList
    disp: Any | None = None       # result of estimateDisp (often a DGEList with dispersion slots)
    fit: Any | None = None        # glmQLFit/glmFit object
    test: Any | None = None       # DGELRT (glmQLFTest) object
    top_table_r: Any | None = None  # TopTags object or data.frame in R

    # Python side artifacts
    mask: Optional[np.ndarray] = None
    norm_factors: Optional[np.ndarray] = None
    cpm: Optional[pd.DataFrame] = None

    # Design / contrast objects kept in R form for reproducibility
    design_r: Any | None = None
    contrast_r: Any | None = None

    # Catch-all metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------- Getters ---------------- #
    def get_counts(self) -> pd.DataFrame:
        if self.dge is None:
            raise ValueError("No DGEList stored.")
        r = _rmana
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            df = r.get_conversion().rpy2py(self.dge.rx2("counts"))
        return df

    def get_samples(self) -> pd.DataFrame:
        if self.dge is None:
            raise ValueError("No DGEList stored.")
        r = _rmana
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            df = r.get_conversion().rpy2py(self.dge.rx2("samples"))
        return df

    def get_genes_kept(self) -> Optional[np.ndarray]:
        return self.mask

    def get_norm_factors(self) -> Optional[np.ndarray]:
        return self.norm_factors

    def get_top_table(self) -> Optional[pd.DataFrame]:
        if self.top_table_r is None:
            return None
        r = _rmana
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            df = r.get_conversion().rpy2py(self.top_table_r)
        return df

# ------------------------------------------------------------------
#  Constructors for EdgeRState
# ------------------------------------------------------------------

def edger_state_from_anndata(adata: "AnnData", layer: Optional[str] = None,
                              group: Optional[pd.Series] = None) -> EdgeRState:
    r = _rmana
    counts_r = _RConverters.anndata_to_r_matrix(adata, layer)
    samples_r = _RConverters.df_to_r_df(adata.obs)
    group_r = r.ro.NULL
    if group is not None:
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            group_r = r.get_conversion().py2rpy(group)
    dge = _edgeR().DGEList(counts=counts_r, samples=samples_r, group=group_r)
    return EdgeRState(dge=dge)


def edger_state_from_dfs(counts: pd.DataFrame, samples: pd.DataFrame,
                         group: Optional[pd.Series] = None) -> EdgeRState:
    r = _rmana
    counts_r = _RConverters.df_to_r_matrix(counts)
    samples_r = _RConverters.df_to_r_df(samples)
    group_r = r.ro.NULL
    if group is not None:
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            group_r = r.get_conversion().py2rpy(group)
    dge = _edgeR().DGEList(counts=counts_r, samples=samples_r, group=group_r)
    return EdgeRState(dge=dge)


def edger_state_from_numpy(counts: np.ndarray,
                           samples: pd.DataFrame,
                           rownames: Optional[Sequence[str]] = None,
                           colnames: Optional[Sequence[str]] = None,
                           group: Optional[Sequence[str]] = None) -> EdgeRState:
    r = _rmana
    if counts.ndim != 2:
        raise ValueError("counts must be 2D")
    counts_r = _RConverters.np_to_r_matrix(counts.T,  # transpose to genes x samples
                                           rownames=rownames, colnames=colnames)
    samples_r = _RConverters.df_to_r_df(samples)
    group_r = r.ro.NULL
    if group is not None:
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            group_r = r.get_conversion().py2rpy(pd.Series(group))
    dge = _edgeR().DGEList(counts=counts_r, samples=samples_r, group=group_r)
    return EdgeRState(dge=dge)

# ------------------------------------------------------------------
#  Functional steps (pure: EdgeRState -> EdgeRState)
# ------------------------------------------------------------------

def filter_step(state: EdgeRState, design_df: Optional[pd.DataFrame] = None, **kwargs) -> EdgeRState:
    r = _rmana
    pkg = _edgeR()
    if design_df is None:
        design_r = r.ro.NULL
    else:
        design_r = _RConverters.df_to_r_matrix(design_df)
    mask_r = pkg.filterByExpr(state.dge, design=design_r, **kwargs)
    # subset
    bracket = r.ro.baseenv['[']

    n_samples = list(r.ro.baseenv["ncol"](state.dge))[0]
    samples_mask = np.ones(n_samples, dtype = bool)
    samples_mask = r.BoolVector(samples_mask)

    filtered_dge = bracket(state.dge, mask_r, samples_mask)
    
    mask_np = np.array(mask_r, dtype=bool)
    return replace(state, dge=filtered_dge, mask=mask_np, design_r=design_r)


def norm_step(state: EdgeRState, **kwargs) -> EdgeRState:
    pkg = _edgeR()
    dge_n = pkg.calcNormFactors(state.dge, **kwargs)
    factors = np.array(dge_n.rx2('samples').rx2('norm.factors'))
    return replace(state, dge=dge_n, norm_factors=factors)


def estimate_disp_step(state: EdgeRState, design_df: pd.DataFrame, **kwargs) -> EdgeRState:
    pkg = _edgeR()
    r_design = _RConverters.df_to_r_matrix(design_df)

    disp = pkg.estimateDisp(state.dge, design=r_design, **kwargs)
    return replace(state, disp=disp, design_r=r_design)


def fit_glmql_step(state: EdgeRState, design_df, **kwargs) -> EdgeRState:
    # if state.disp is None or state.design_r is None:
    #     raise ValueError("estimate_disp_step must be run before fit_glmql_step")
    pkg = _edgeR()
    r = _rmana
    edger = r.importr("edgeR")
    design_r = _RConverters.df_to_r_matrix(design_df)
    
    fit = edger.glmQLFit(state.dge, design=design_r, )
    return replace(state, fit=fit)


def test_contrast_step(state: EdgeRState, contrast: Sequence[float], **kwargs) -> EdgeRState:
    if state.fit is None:
        raise ValueError("fit_glmql_step must be run before test_contrast_step")
    r = _rmana
    pkg = _edgeR()
    with r.localconverter(r.default_converter + r.numpy2ri_converter):
        contrast_r = r.get_conversion().py2rpy(np.asarray(contrast, dtype=float))
    test = pkg.glmQLFTest(state.fit, contrast=contrast_r, **kwargs)
    return replace(state, test=test, contrast_r=contrast_r)


def top_tags_step(state: EdgeRState, n: Optional[int] = None,
                  adjust_method: str = "BH", sort_by: str = "PValue", **kwargs) -> Tuple[EdgeRState, pd.DataFrame]:
    if state.test is None:
        raise ValueError("test_contrast_step must be run before top_tags_step")
    r = _rmana
    pkg = _edgeR()
    if n is None:
        n_val = r.ro.r('Inf')
    else:
        n_val = int(n)
    top = pkg.topTags(state.test, n=n_val, adjust_method=adjust_method, sort_by=sort_by, **kwargs)
    # Extract the table slot and convert to pandas
    table_r = top.rx2('table') if 'table' in list(top.names) else r.ro.baseenv['as.data.frame'](top)
    with r.localconverter(r.default_converter + r.pandas2ri_converter):
        df = r.get_conversion().rpy2py(table_r)
    df = df.reset_index().rename(columns={'index': 'gene', 'PValue': 'p_value', 'logFC': 'log_fc', 'FDR': 'adj_p_value'})
    new_state = replace(state, top_table_r=table_r)
    return new_state


def cpm_step(state: EdgeRState, log: bool = True, prior_count: float = 1.0, **kwargs) -> Tuple[EdgeRState, pd.DataFrame]:
    pkg = _edgeR()
    cpm_r = pkg.cpm(state.dge, log=log, prior_count=prior_count, **kwargs)
    r = _rmana
    with r.localconverter(r.default_converter + r.pandas2ri_converter):
        cpm_df = r.get_conversion().rpy2py(cpm_r)
    return replace(state, cpm=cpm_df), cpm_df

# ------------------------------------------------------------------
#  Optional: tiny pipeline helper
# ------------------------------------------------------------------

def pipe(state: EdgeRState, *funcs):
    for f in funcs:
        state = f(state)
    return state


# ------------------------------------------------------------------
#  Fluent API bindings (chainable methods returning new immutable states)
# ------------------------------------------------------------------

def _edger_filter(self: EdgeRState, design_df: Optional[pd.DataFrame] = None, **kwargs) -> EdgeRState:
    return filter_step(self, design_df=design_df, **kwargs)

def _edger_norm(self: EdgeRState, **kwargs) -> EdgeRState:
    return norm_step(self, **kwargs)

def _edger_estimate_disp(self: EdgeRState, design_df: pd.DataFrame, **kwargs) -> EdgeRState:
    return estimate_disp_step(self, design_df, **kwargs)

def _edger_fit_glmql(self: EdgeRState, **kwargs) -> EdgeRState:
    return fit_glmql_step(self, **kwargs)

def _edger_test_contrast(self: EdgeRState, contrast: Sequence[float], **kwargs) -> EdgeRState:
    return test_contrast_step(self, contrast, **kwargs)

def _edger_top_tags(self: EdgeRState, n: Optional[int] = None, adjust_method: str = "BH", sort_by: str = "PValue", **kwargs) -> Tuple[EdgeRState, pd.DataFrame]:
    return top_tags_step(self, n=n, adjust_method=adjust_method, sort_by=sort_by, **kwargs)

def _edger_cpm(self: EdgeRState, log: bool = True, prior_count: float = 1.0, **kwargs) -> Tuple[EdgeRState, pd.DataFrame]:
    return cpm_step(self, log=log, prior_count=prior_count, **kwargs)

# attach methods
EdgeRState.filter = _edger_filter            # type: ignore[attr-defined]
EdgeRState.norm = _edger_norm                # type: ignore[attr-defined]
EdgeRState.estimate_disp = _edger_estimate_disp  # type: ignore[attr-defined]
EdgeRState.fit_glmql = _edger_fit_glmql      # type: ignore[attr-defined]
EdgeRState.test_contrast = _edger_test_contrast  # type: ignore[attr-defined]
EdgeRState.top_tags = _edger_top_tags        # type: ignore[attr-defined]
EdgeRState.cpm_matrix = _edger_cpm           # type: ignore[attr-defined]