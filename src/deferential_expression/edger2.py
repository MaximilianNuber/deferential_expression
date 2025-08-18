from typing import Any, Optional, Sequence, Union

import functools
from typing import Any, Callable, Dict, Optional, Type, Union


import pandas as pd
import numpy as np

from deferential_expression.design_matrix import DesignMixin
from formulaic_contrasts import FormulaicContrasts
from anndata import AnnData

# global manager instance
from deferential_expression.rpy2_manager import Rpy2Manager
_r_manager = Rpy2Manager()

# Conversion helper
class _EdgeRConverters:
    @staticmethod
    def df_to_r_matrix(df: pd.DataFrame) -> Any:
        r = _r_manager
        with r.localconverter(r.default_converter + r.numpy2ri_converter):
            r_mat = r.get_conversion().py2rpy(df.values)
        # set dimnames

        colnames = df.columns.to_numpy().astype(str)
        rownames = df.index.to_numpy().astype(str)

        r_mat = r.ro.baseenv["rownames<-"](r_mat, r.StrVector(rownames))
        r_mat = r.ro.baseenv["colnames<-"](r_mat, r.StrVector(colnames))
        return r_mat

    @staticmethod
    def df_to_r_dataframe(df: pd.DataFrame) -> Any:
        r = _r_manager
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            return r.get_conversion().py2rpy(df)
        
    @staticmethod
    def anndata_to_r_matrix(adata: AnnData, layer = None) -> Any:
        r = _r_manager

        mat = adata.X.T if layer is None else adata.layers[layer].T
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"Expected AnnData.X or layer '{layer}' to be a numpy array, got {type(mat)}")
        
        obs_names = adata.obs_names.to_numpy().astype(str)
        var_names = adata.var_names.to_numpy().astype(str)

        with r.localconverter(r.default_converter + r.numpy2ri_converter):
            r_mat = r.get_conversion().py2rpy(mat)
        
        # set dimnames
        r_mat = r.ro.baseenv["rownames<-"](r_mat, r.StrVector(var_names))
        r_mat = r.ro.baseenv["colnames<-"](r_mat, r.StrVector(obs_names))
        return r_mat

# Mixins
class FilterByExprMixin:
    def filter_by_expr(self, dge_obj: Any, design: pd.DataFrame | None = None, **kwargs) -> Any:
        """Apply edgeR::filterByExpr to return a new DGEList R object."""
        r = _r_manager
        if design is None:
            design = r.ro.NULL
        elif isinstance(design, pd.DataFrame):
            design = _EdgeRConverters.df_to_r_matrix(design)
        else:
            raise TypeError(f"Expected design to be a DataFrame or None, got {type(design)}")
            
        mask = self._pkg.filterByExpr(dge_obj, design=design, **kwargs)
        # subset counts by mask
        bracket = r.ro.baseenv['[']
        n_samples = r.ro.baseenv["ncol"](dge_obj)
        samples_mask = np.ones(n_samples, dtype = bool)
        samples_mask = r.BoolVector(samples_mask)
        filtered = bracket(dge_obj, mask, samples_mask)
        return filtered

class CalcNormFactorsMixin:
    def calc_norm_factors(self, dge_obj: Any) -> Any:
        """Apply edgeR::calcNormFactors and return updated DGEList R object."""
        updated = self._pkg.calcNormFactors(dge_obj)
        return updated
    
class CpmMixin:
    def cpm(self, obj: Any, log: bool = False, convert = True, **kwargs) -> pd.DataFrame:
        """Calculate counts per million (CPM) and return as pandas DataFrame."""
        r = _r_manager
        cpm_func = self._pkg.cpm
        if isinstance(obj, pd.DataFrame):
            # convert DataFrame to R matrix
            robj_EdgeRConverters.df_to_r_matrix(obj)
        elif isinstance(obj, AnnData):
            # convert AnnData to R matrix
            robj = _EdgeRConverters.anndata_to_r_matrix(obj)
        elif isinstance(obj, np.ndarray):
            # convert numpy array to R matrix
            with r.localconverter(r.default_converter + r.numpy2ri_converter):
                robj = r.get_conversion().py2rpy(obj)
        elif list(r.ro.baseenv["class"](obj))[0] == "DGEList":
            # assume obj is already a DGEList R object
            robj = obj
        elif list(r.ro.baseenv["is.matrix"](obj))[0]:
            # assume obj is already an R matrix
            robj = obj
        else:
            raise TypeError(f"Expected obj to be DataFrame, AnnData, or numpy array, got {type(obj)}")
        rcpm = cpm_func(robj, log=log, **kwargs)
        if convert:
            with r.localconverter(r.default_converter + r.pandas2ri_converter):
                cpm = r.get_conversion().rpy2py(rcpm)
            return cpm
        else:
            return rcpm

class EdgeRFitMixin:
    def fit_edgeR(self, dge_obj: Any, design_matrix: pd.DataFrame, **kwargs) -> Any:
        """Run estimateDisp and glmQLFit, returning the fitted R object."""
        r = _r_manager
        r_design = _EdgeRConverters.df_to_r_matrix(design_matrix)
        disp = self._pkg.estimateDisp(dge_obj, design=r_design)
        fit = self._pkg.glmQLFit(disp, design=r_design, **kwargs)
        return fit

class ContrastTestMixin:
    def test_contrast(self, fit_obj: Any, contrast: Sequence[float], **kwargs) -> pd.DataFrame:
        """Run glmQLFTest and topTags, returning pandas DataFrame."""
        r = _r_manager
        # convert contrast vector
        with r.localconverter(r.default_converter + r.numpy2ri_converter):
            cvec = r.get_conversion().py2rpy(np.asarray(contrast))
        test = self._pkg.glmQLFTest(fit_obj, contrast = cvec, **kwargs)
        top = self._pkg.topTags(test, n=np.inf, **kwargs)
        top = r.ro.baseenv["as.data.frame"](top)
        # extract table
        # table = top.rx2('table')
        with r.localconverter(r.default_converter + r.pandas2ri_converter):
            df = r.get_conversion().rpy2py(top)
        # df.index.name = 'gene'
        return df.reset_index(names = "gene").rename(columns={'PValue':'p_value','logFC':'log_fc','FDR':'adj_p_value'})

# Runner composing mixins
class EdgeRRunner(
    FilterByExprMixin, 
    CalcNormFactorsMixin, 
    EdgeRFitMixin, 
    ContrastTestMixin, 
    DesignMixin,
    CpmMixin
    ):
    def __init__(
            self, 
            counts: pd.DataFrame | AnnData, 
            samples: pd.DataFrame | None = None, 
            group: Optional[pd.Series] = None,
            layer: str | None = None
            ):
        r = _r_manager
        # import edgeR
        self._pkg = r.importr('edgeR')
        # create DGEList R object
        if isinstance(counts, pd.DataFrame):
            assert samples is not None, "samples DataFrame must be provided when counts is a DataFrame"
            r_counts = _EdgeRConverters.df_to_r_matrix(counts)
            r_samples = _EdgeRConverters.df_to_r_dataframe(samples)
        elif isinstance(counts, AnnData):
            r_counts = _EdgeRConverters.anndata_to_r_matrix(counts, layer=layer)
            r_samples = _EdgeRConverters.df_to_r_dataframe(counts.obs)
            
        r_group = r.ro.NULL        
        if group is not None:
            with r.localconverter(r.default_converter + r.pandas2ri_converter):
                r_group = r.get_conversion().py2rpy(group)
        self.dge = self._pkg.DGEList(counts=r_counts, samples=r_samples, group=r_group)

        self.formulaic: FormulaicContrasts | None = None
        self.design_matrix: pd.DataFrame | None = None
        self.edger_fit = None

    def run(self,
            design: pd.DataFrame,
            contrast: Sequence[float],
            min_count: int = 10,
            filter_kwargs: Optional[Dict[str, Any]] = None,
            normalize_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs) -> pd.DataFrame:
        # 1. Filter
        dge = self.filter_by_expr(self.dge, design=design, min_count=min_count, **(filter_kwargs or {}))
        # 2. Normalize
        dge = self.calc_norm_factors(dge, **(normalize_kwargs or {}))
        # 3. Fit
        dge = self.fit_edgeR(dge, design, **kwargs)
        # 4. Test
        return self.test_contrast(dge, contrast)