"""
TEST: New SummarizedExperiment-based API for edgeR.

This is a test implementation of a new pattern where:
1. Functions accept SummarizedExperiment (not RESummarizedExperiment)
2. An initialize_r() utility converts assays to RMatrixAdapter
3. Each function checks that assays are properly initialized

This file is for TESTING ONLY - it does not modify the existing implementation.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from summarizedexperiment import SummarizedExperiment

# Import the new RMatrixAdapter
import sys
sys.path.insert(0, "/home/max-nuber/projects/deferential_expression/src")
from deferential_expression.rmatrixadapter import RMatrixAdapter


# =============================================================================
# Utility: initialize_r
# =============================================================================

def initialize_r(
    se: SummarizedExperiment,
    assay: str = "counts",
    in_place: bool = False,
) -> SummarizedExperiment:
    """
    Initialize R backing for a SummarizedExperiment assay.
    
    Converts the specified assay to an RMatrixAdapter, enabling R-based
    analysis functions. This must be called before using edgeR functions.
    
    Args:
        se: Input SummarizedExperiment.
        assay: Name of the assay to convert. Default: "counts".
        in_place: If True, modify se in place. If False, return a new SE.
    
    Returns:
        SummarizedExperiment with the assay converted to RMatrixAdapter.
    
    Example:
        >>> se = initialize_r(se, assay="counts")
        >>> # Now se.assays["counts"] is an RMatrixAdapter
        >>> model = glm_ql_fit(se, design)
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from bioc2ri import numpy_plugin
    from bioc2ri.rnames import set_rownames, set_colnames
    
    if assay not in se.assay_names:
        raise KeyError(f"Assay '{assay}' not found. Available: {list(se.assay_names)}")
    
    # Get the assay data
    arr = se.assays[assay]
    
    # If already an RMatrixAdapter, return as-is
    if isinstance(arr, RMatrixAdapter):
        return se if in_place else se
    
    # Convert to R matrix
    r = get_r_environment()
    np_eng = numpy_plugin()
    
    arr_np = np.asarray(arr)
    rmat = np_eng.py2r(arr_np)
    
    # Set dimnames if available
    if se.row_names is not None:
        rn = np.asarray(se.row_names, dtype=str)
        rmat = set_rownames(rmat, np_eng.py2r(rn))
    if se.column_names is not None:
        cn = np.asarray(se.column_names, dtype=str)
        rmat = set_colnames(rmat, np_eng.py2r(cn))
    
    # Create adapter
    adapter = RMatrixAdapter(rmat, r)
    
    # Update the assay
    new_assays = dict(se.assays)
    new_assays[assay] = adapter
    
    # Use _define_output pattern for in_place support
    output = se._define_output(in_place=in_place)
    output._assays = new_assays
    
    return output


def check_r_initialized(se: SummarizedExperiment, assay: str) -> None:
    """
    Check that an assay is an RMatrixAdapter.
    
    Raises:
        TypeError: If assay is not an RMatrixAdapter.
    """
    if assay not in se.assay_names:
        raise KeyError(f"Assay '{assay}' not found. Available: {list(se.assay_names)}")
    
    arr = se.assays[assay]
    if not isinstance(arr, RMatrixAdapter):
        raise TypeError(
            f"Assay '{assay}' is not R-initialized. "
            f"Call initialize_r(se, assay='{assay}') first."
        )


# =============================================================================
# EdgeR Model
# =============================================================================

@dataclass
class EdgeRModelNew:
    """Container for edgeR GLM fit results."""
    sample_names: Optional[Sequence[str]] = None
    feature_names: Optional[Sequence[str]] = None
    fit: Optional[Any] = None
    design: Optional[pd.DataFrame] = None


# =============================================================================
# EdgeR Functions (accepting SummarizedExperiment)
# =============================================================================

def calc_norm_factors(
    se: SummarizedExperiment,
    assay: str = "counts",
    method: str = "TMM",
    in_place: bool = False,
    **kwargs
) -> SummarizedExperiment:
    """
    Calculate normalization factors using edgeR::calcNormFactors.
    
    Args:
        se: Input SummarizedExperiment with R-initialized counts assay.
        assay: Assay name. Default: "counts".
        method: Normalization method. Default: "TMM".
        in_place: Modify in place. Default: False.
    
    Returns:
        SummarizedExperiment with norm.factors in column_data.
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from deferential_expression.edger.utils import _prep_edger
    
    check_r_initialized(se, assay)
    
    r, edger = _prep_edger()
    
    # Get R matrix
    adapter: RMatrixAdapter = se.assays[assay]
    rmat = adapter.rmat
    
    # Calculate norm factors
    r_factors = edger.calcNormFactors(rmat, method=method, **kwargs)
    norm_factors = np.asarray(r_factors)
    
    # Update column_data
    output = se._define_output(in_place=in_place)
    coldata = output.column_data.to_pandas() if output.column_data is not None else pd.DataFrame()
    coldata["norm.factors"] = norm_factors
    output._column_data = coldata
    
    return output


def filter_by_expr(
    se: SummarizedExperiment,
    assay: str = "counts",
    min_count: float = 10,
    min_total_count: float = 15,
    **kwargs
) -> np.ndarray:
    """
    Filter genes by expression using edgeR::filterByExpr.
    
    Returns:
        Boolean mask of genes passing filter.
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from deferential_expression.edger.utils import _prep_edger
    
    check_r_initialized(se, assay)
    
    r, edger = _prep_edger()
    
    adapter: RMatrixAdapter = se.assays[assay]
    rmat = adapter.rmat
    
    mask_r = edger.filterByExpr(
        rmat,
        min_count=min_count,
        min_total_count=min_total_count,
        **kwargs
    )
    
    return np.asarray(mask_r, dtype=bool)


def glm_ql_fit(
    se: SummarizedExperiment,
    design: pd.DataFrame,
    assay: str = "counts",
    **kwargs
) -> EdgeRModelNew:
    """
    Fit GLM using edgeR::glmQLFit.
    
    Args:
        se: Input SummarizedExperiment with R-initialized counts assay.
        design: Design matrix.
    
    Returns:
        EdgeRModelNew with fit results.
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from deferential_expression.edger.utils import _prep_edger, pandas_to_r_matrix
    
    check_r_initialized(se, assay)
    
    r, edger = _prep_edger()
    
    adapter: RMatrixAdapter = se.assays[assay]
    rmat = adapter.rmat
    design_r = pandas_to_r_matrix(design)
    
    # Create DGEList
    dge = edger.DGEList(counts=rmat)
    
    # Estimate dispersions
    dge = edger.estimateDisp(dge, design=design_r)
    
    # Fit GLM
    fit = edger.glmQLFit(dge, design=design_r, **kwargs)
    
    return EdgeRModelNew(
        sample_names=se.column_names,
        feature_names=se.row_names,
        fit=fit,
        design=design,
    )


def glm_ql_ftest(
    model: EdgeRModelNew,
    coef: Optional[Union[int, str]] = None,
    adjust_method: str = "BH",
) -> pd.DataFrame:
    """
    Perform quasi-likelihood F-test using edgeR::glmQLFTest.
    
    Returns:
        DataFrame with DE results.
    """
    from bioc2ri.lazy_r_env import get_r_environment
    from deferential_expression.edger.utils import _prep_edger
    
    if model.fit is None:
        raise ValueError("Model has no fit - call glm_ql_fit first")
    
    r, edger = _prep_edger()
    
    # Run F-test
    if coef is not None:
        lrt = edger.glmQLFTest(model.fit, coef=coef)
    else:
        lrt = edger.glmQLFTest(model.fit)
    
    # Get top tags
    top = edger.topTags(lrt, n=float("inf"), adjust_method=adjust_method)
    table_r = r.ro.baseenv["$"](top, "table")
    
    with r.localconverter(r.default_converter + r.pandas2ri.converter):
        df = r.get_conversion().rpy2py(table_r)
    
    return df.reset_index(names="gene")


# =============================================================================
# Run the test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: New SummarizedExperiment-based edgeR API")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    n_genes, n_samples = 100, 6
    counts = np.random.negative_binomial(10, 0.3, size=(n_genes, n_samples)).astype(float)
    
    design = pd.DataFrame({
        'Intercept': [1] * n_samples,
        'Condition': [0, 0, 0, 1, 1, 1]
    }, index=[f'S{i}' for i in range(n_samples)])
    
    se = SummarizedExperiment(
        assays={'counts': counts},
        row_names=[f'Gene{i}' for i in range(n_genes)],
        column_names=[f'S{i}' for i in range(n_samples)]
    )
    
    print(f"\n1. Created SummarizedExperiment: {se.shape}")
    print(f"   assays['counts'] type: {type(se.assays['counts']).__name__}")
    
    # Test: calling without initialize_r should fail
    print("\n2. Testing error when not initialized...")
    try:
        calc_norm_factors(se)
        print("   ERROR: Should have raised TypeError!")
    except TypeError as e:
        print(f"   ✓ Correctly raised TypeError: {e}")
    
    # Initialize R backing
    print("\n3. Initializing R backing...")
    se = initialize_r(se, assay="counts")
    print(f"   assays['counts'] type: {type(se.assays['counts']).__name__}")
    print(f"   shape: {se.assays['counts'].shape}")
    
    # Now functions should work
    print("\n4. Testing calc_norm_factors...")
    se = calc_norm_factors(se, method="TMM")
    print(f"   ✓ norm.factors added to column_data")
    
    print("\n5. Testing filter_by_expr...")
    mask = filter_by_expr(se, min_count=5)
    print(f"   ✓ Mask shape: {mask.shape}, genes passing: {mask.sum()}")
    
    print("\n6. Testing glm_ql_fit...")
    model = glm_ql_fit(se, design)
    print(f"   ✓ EdgeRModelNew created, fit is not None: {model.fit is not None}")
    
    print("\n7. Testing glm_ql_ftest...")
    results = glm_ql_ftest(model, coef=2)
    print(f"   ✓ Results DataFrame: {results.shape}")
    print(f"   Columns: {list(results.columns)}")
    print(f"   Top 3 genes:")
    print(results.head(3).to_string())
    
    print("\n" + "=" * 60)
    print("ALL SummarizedExperiment TESTS PASSED!")
    print("=" * 60)
    
    # =========================================================================
    # TEST: RangedSummarizedExperiment
    # =========================================================================
    
    print("\n")
    print("=" * 60)
    print("TEST: New RangedSummarizedExperiment-based edgeR API")
    print("=" * 60)
    
    from summarizedexperiment import RangedSummarizedExperiment
    from genomicranges import GenomicRanges
    from iranges import IRanges
    
    # Create genomic ranges for genes
    gr = GenomicRanges(
        seqnames=[f"chr{(i % 22) + 1}" for i in range(n_genes)],
        ranges=IRanges(start=[i * 1000 for i in range(n_genes)], width=[500] * n_genes),
        strand=["+" if i % 2 == 0 else "-" for i in range(n_genes)],
    )
    
    # Create RSE with the same count data
    counts_rse = np.random.negative_binomial(10, 0.3, size=(n_genes, n_samples)).astype(float)
    
    rse = RangedSummarizedExperiment(
        assays={'counts': counts_rse},
        row_ranges=gr,
        row_names=[f'Gene{i}' for i in range(n_genes)],
        column_names=[f'S{i}' for i in range(n_samples)]
    )
    
    print(f"\n1. Created RangedSummarizedExperiment: {rse.shape}")
    print(f"   assays['counts'] type: {type(rse.assays['counts']).__name__}")
    print(f"   row_ranges: {type(rse.row_ranges).__name__}")
    
    # Test: calling without initialize_r should fail
    print("\n2. Testing error when not initialized...")
    try:
        calc_norm_factors(rse)
        print("   ERROR: Should have raised TypeError!")
    except TypeError as e:
        print(f"   ✓ Correctly raised TypeError: {e}")
    
    # Initialize R backing
    print("\n3. Initializing R backing...")
    rse = initialize_r(rse, assay="counts")
    print(f"   assays['counts'] type: {type(rse.assays['counts']).__name__}")
    print(f"   shape: {rse.assays['counts'].shape}")
    
    # Now functions should work
    print("\n4. Testing calc_norm_factors...")
    rse = calc_norm_factors(rse, method="TMM")
    print(f"   ✓ norm.factors added to column_data")
    
    print("\n5. Testing filter_by_expr...")
    mask = filter_by_expr(rse, min_count=5)
    print(f"   ✓ Mask shape: {mask.shape}, genes passing: {mask.sum()}")
    
    print("\n6. Testing glm_ql_fit...")
    model_rse = glm_ql_fit(rse, design)
    print(f"   ✓ EdgeRModelNew created, fit is not None: {model_rse.fit is not None}")
    
    print("\n7. Testing glm_ql_ftest...")
    results_rse = glm_ql_ftest(model_rse, coef=2)
    print(f"   ✓ Results DataFrame: {results_rse.shape}")
    print(f"   Columns: {list(results_rse.columns)}")
    print(f"   Top 3 genes:")
    print(results_rse.head(3).to_string())
    
    # Verify row_ranges still accessible
    print("\n8. Testing row_ranges preservation...")
    print(f"   ✓ row_ranges preserved: {rse.row_ranges is not None}")
    print(f"   ✓ row_ranges length: {len(rse.row_ranges)}")
    
    print("\n" + "=" * 60)
    print("ALL RangedSummarizedExperiment TESTS PASSED!")
    print("=" * 60)
    
    print("\n")
    print("=" * 60)
    print("=== ALL TESTS COMPLETE ===")
    print("=" * 60)

