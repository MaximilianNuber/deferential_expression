from deferential_expression.resummarizedexperiment import RMatrixAdapter
from deferential_expression.edger.utils import _prep_edger

def cpm(obj: "RESummarizedExperiment", assay: str = "counts", **kwargs):
    """Functional CPM computation using ``edgeR::cpm`` and assay insertion.

    Args:
        obj: ``EdgeR`` instance.
        assay: Counts assay name.
        **kwargs: Additional args forwarded to ``edgeR::cpm``.

    Returns:
        EdgeR: New object with a ``"cpm"`` assay added (R-backed).
    """
    renv, pkg = _prep_edger()

    rmat = obj.assay_r(assay)

    cpm_mat = pkg.cpm(rmat, **kwargs)
    cpm_mat = RMatrixAdapter(cpm_mat, renv)

    return obj.set_assay(name="cpm", value=cpm_mat)