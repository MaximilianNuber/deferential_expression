import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Union
from dataclasses import dataclass
from deferential_expression_acc.edger.utils import _prep_edger, numpy_to_r_matrix, pandas_to_r_matrix


def glm_ql_ftest(obj: "EdgeRModel", coef: Optional[Union[str, int]] = None, contrast: Optional[Sequence] = None, poisson_bound: bool = True,
                adjust_method: str = "BH"):
    """Functional quasi-likelihood F-test and table extraction via ``topTags``.

    Args:
        obj: ``EdgeR`` instance with a fitted ``glm``.
        coef: Optional coefficient name (str) or index (int) to test.
        contrast: Optional contrast vector.
        poisson_bound: Whether to apply the Poisson bound in the test.
        adjust_method: Multiple-testing method for the returned table.

    Returns:
        pandas.DataFrame: Results table with all rows (``n = Inf`` in R). Columns
        follow edgeR’s defaults.

    Raises:
        AssertionError: If ``glm`` has not been set on ``obj``.
    """
    assert hasattr(obj, "fit")
    assert obj.fit is not None

    r, pkg = _prep_edger()
    if coef is not None:
        if isinstance(coef, int):
            coef = r.IntVector([coef])
        else:
            coef = r.StrVector([str(coef)])
    else:
        coef = r.ro.NULL
    if contrast is not None:
        contrast = np.asarray(contrast, dtype = int)
        contrast = r.IntVector(contrast)
    else:
        contrast = r.ro.NULL
    poisson_bound = r.BoolVector([poisson_bound])

    res = pkg.glmQLFTest(obj.fit, coef = coef, contrast = contrast, poisson_bound = poisson_bound)
    # topTags(object, n=10, adjust.method="BH", sort.by="PValue", p.value=1)

    res = pkg.topTags(
        res,
        n = r.ro.r("Inf"),
        adjust_method = adjust_method,
        sort_by = r.ro.NULL,
        p_value = r.IntVector([1])
    )

    res = r.ro.baseenv["as.data.frame"](res)
    with r.localconverter(
        r.default_converter + r.pandas2ri.converter
    ):
        res = r.get_conversion().rpy2py(res)

    return res
