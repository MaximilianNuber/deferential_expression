import numpy as np
from .utils import _prep_edger


def calc_norm_factors(
    obj: "RESummarizedExperiment", 
    assay = "counts", 
    method = "TMM", 
    refColumn=None,
    logratioTrim=.3,
    sumTrim=0.05,
    doWeighting=True,
    Acutoff=-1e10, 
    p=0.75,
    **kwargs
):
    """Functional TMM normalization: compute and store norm factors in colData.

    Args:
        obj: ``EdgeR`` instance.
        assay: Counts assay name.
        **kwargs: Additional args forwarded to ``edgeR::calcNormFactors``.

    Returns:
        EdgeR: New object with updated column data containing ``norm.factors``.
    """
    r, pkg = _prep_edger()

    rmat = obj.assay_r(assay)

    # r_factors = pkg.calcNormFactors(rmat, **kwargs)
    r_factors = _calc_norm_factors_impl(
        rmat, 
        method = method, 
        refColumn=refColumn,
        logratioTrim=logratioTrim,
        sumTrim=sumTrim,
        doWeighting=doWeighting,
        Acutoff=Acutoff, 
        p=p,
        **kwargs
        )
    norm_factors = np.asarray(r_factors)

    # new_cols = BiocFrame({"norm.factors": norm_factors})
    coldata = obj.get_column_data()

    import biocutils as ut

    # new_cols = ut.combine_columns(coldata, new_cols)
    new_cols = coldata.set_column("norm.factors", norm_factors)

    return obj.set_column_data(new_cols)

def _calc_norm_factors_impl(
    rmat, 
    method = "TMM", 
    refColumn=None,
    logratioTrim=.3,
    sumTrim=0.05,
    doWeighting=True,
    Acutoff=-1e10, 
    p=0.75,
    **kwargs
):
    r, pkg = _prep_edger()
    refColumn = r.ro.NULL if refColumn is None else refColumn
    return pkg.calcNormFactors(
        rmat, 
        method = method, 
        refColumn=refColumn,
        logratioTrim=logratioTrim,
        sumTrim=sumTrim,
        doWeighting=doWeighting,
        Acutoff=Acutoff, 
        p=p,
        **kwargs
    )