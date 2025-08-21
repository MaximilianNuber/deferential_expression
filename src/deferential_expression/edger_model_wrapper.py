# in diffexptools/dgelist_wrapper.py
import numpy as np
from typing import Any, Sequence
from pyrtools import RFunctionWrapper, RModelWrapper
from pyrtools.r_env import lazy_import_r_env

class DGEList(RModelWrapper):
    """
    A specialized wrapper around an edgeR DGEList object.
    """

    def __init__(self, r_model_object: Any):
        renv = lazy_import_r_env()
        r_class = renv.r2py(renv.ro.baseenv["class"](r_model_object))
        if r_class != "DGEList":
            raise TypeError(f"Expected a DGEList, got {r_class}")
        super().__init__(r_model_object)

    def subset(self, rows: Sequence | None = None, cols: Sequence | None = None):
        renv = lazy_import_r_env()
        dge = self._r_model
        nrows = renv.ro.baseenv["nrow"](dge)
        ncols = renv.ro.baseenv["ncol"](dge)
        if rows is None:
            rows = np.ones(nrows)
            rows = renv.py2r(rows)
        if not hasattr(rows, "__sexp__"):
            rows = np.asarray(rows)
            rows = renv.py2r(rows)
            
        if cols is None:
            cols = np.ones(ncols)
            cols = renv.py2r(cols)
        if not hasattr(cols, "__sexp__"):
            cols = np.asarray(cols)
            cols = renv.py2r(cols)
        
        dge = renv.ro.baseenv["["](dge, rows, cols)
        # dge = dge.rx(rows, cols)

        return DGEList(dge)

    @property
    def shape(self):
        renv = lazy_import_r_env()
        return tuple(renv.ro.baseenv["dim"](self._r_model))

    @property
    def rownames(self):
        renv = lazy_import_r_env()
        return renv.r2py(renv.ro.baseenv["rownames"](self._r_model))

    @rownames.setter
    def rownames(self, names: Sequence[str]):
        renv = lazy_import_r_env()
        vec = renv.ro.StrVector(names)
        # call R’s `rownames<-` and re-wrap
        out = RFunctionWrapper("rownames<-") \
                  .get_python_function(convert_output=False)(
                      self._r_model, vec
                  )
        return DGEList(out)


class DGEGLM(RModelWrapper):
    def __init__(self, obj):
        renv = lazy_import_r_env()
        r_class = renv.r2py(renv.ro.baseenv["class"](obj))
        if r_class != "DGEGLM":
            raise TypeError(f"Expected a DGEList, got {r_class}")
        super().__init__(obj)

class DGELRT(RModelWrapper):
    def __init__(self, obj):
        renv = lazy_import_r_env()
        r_class = renv.r2py(renv.ro.baseenv["class"](obj))
        if r_class != "DGELRT":
            raise TypeError(f"Expected a DGEList, got {r_class}")
        super().__init__(obj)

# # in diffexptools/api.py
# from pyrtools import RFunctionWrapper
# # from diffexptools.dgelist_wrapper import DGEList


# def make_dge_list(obj, **kwargs) -> DGEList:
#     dge = dge_list(counts, **kwargs)
#     return DGEList(dge)

# def filter_by_expr(dge: DGEList, design: Any) -> DGEList:
#     fn = RFunctionWrapper("filterByExpr",
#                           package="edgeR") \
#              .get_python_function(convert_output=False)
#     r_out = fn(dge._r_model, design=design)
#     return r_out

# def glmqlfit(dge: DGEList, design: Any) -> DGEList:
#     fn = RFunctionWrapper("glmQLFit",
#                           package="edgeR") \
#              .get_python_function(convert_output=False)
#     r_fit = fn(dge._r_model, design=design)
#     return DGEGLM(r_fit)

# def glmqlftest(fit: DGEGLM, **kwargs) -> DGEList:
#     fn = RFunctionWrapper("glmQLFTest",
#                           package="edgeR") \
#              .get_python_function(convert_output=False)
#     r_test = fn(fit._r_model, **kwargs)
#     return DGELRT(r_test)