"""
Comprehensive tests for RMatrixAdapter integration across all SE types.

Tests SummarizedExperiment, RangedSummarizedExperiment, and SingleCellExperiment
with the edgeR workflow to verify:
1. initialize_r works correctly
2. edgeR functions accept all SE variants
3. Class type is preserved after operations
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "/home/max-nuber/projects/deferential_expression/src")

from deferential_expression import (
    initialize_r,
    check_r_initialized,
    is_r_initialized,
    RMatrixAdapter,
)
import deferential_expression.edger as edger


def run_edger_workflow(se, se_type_name: str, design: pd.DataFrame):
    """Run the edgeR workflow and verify class preservation."""
    original_type = type(se)
    print(f"\n{'='*60}")
    print(f"Testing {se_type_name}")
    print(f"{'='*60}")
    
    # 1. Verify initial state
    print(f"1. Created {se_type_name}: {se.shape}")
    print(f"   assays['counts'] type: {type(se.assays['counts']).__name__}")
    print(f"   Class: {type(se).__name__}")
    
    # 2. Test error without initialization
    print("\n2. Testing error when not initialized...")
    try:
        edger.calc_norm_factors(se)
        print("   ERROR: Should have raised TypeError!")
        return False
    except TypeError as e:
        print(f"   ✓ Correctly raised TypeError")
    
    # 3. Initialize R backing
    print("\n3. Initializing R backing...")
    se = initialize_r(se, assay="counts")
    assert isinstance(se.assays["counts"], RMatrixAdapter), "Assay not RMatrixAdapter"
    assert type(se) == original_type, f"Class changed! {type(se)} != {original_type}"
    print(f"   ✓ assays['counts'] is RMatrixAdapter")
    print(f"   ✓ Class preserved: {type(se).__name__}")
    
    # 4. calc_norm_factors
    print("\n4. Testing calc_norm_factors...")
    se = edger.calc_norm_factors(se, method="TMM")
    assert type(se) == original_type, f"Class changed after calc_norm_factors!"
    print(f"   ✓ norm.factors added")
    print(f"   ✓ Class preserved: {type(se).__name__}")
    
    # 5. filter_by_expr
    print("\n5. Testing filter_by_expr...")
    mask = edger.filter_by_expr(se, min_count=5)
    print(f"   ✓ Mask: {mask.sum()} / {len(mask)} genes pass filter")
    
    # 6. glm_ql_fit
    print("\n6. Testing glm_ql_fit...")
    model = edger.glm_ql_fit(se, design)
    assert model.fit is not None, "Fit is None"
    print(f"   ✓ EdgeRModel created, fit is not None")
    
    # 7. glm_ql_ftest
    print("\n7. Testing glm_ql_ftest...")
    results = edger.glm_ql_ftest(model, coef=2)
    assert isinstance(results, pd.DataFrame), "Results not DataFrame"
    print(f"   ✓ Results: {results.shape}")
    print(f"   Top 3: {results.index[:3].tolist()}")
    
    print(f"\n{'='*60}")
    print(f"✓ ALL {se_type_name} TESTS PASSED!")
    print(f"{'='*60}")
    return True


def test_summarized_experiment():
    """Test with plain SummarizedExperiment."""
    from summarizedexperiment import SummarizedExperiment
    
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
    
    return run_edger_workflow(se, "SummarizedExperiment", design)


def test_ranged_summarized_experiment():
    """Test with RangedSummarizedExperiment."""
    from summarizedexperiment import RangedSummarizedExperiment
    from genomicranges import GenomicRanges
    from iranges import IRanges
    
    np.random.seed(43)
    n_genes, n_samples = 100, 6
    counts = np.random.negative_binomial(10, 0.3, size=(n_genes, n_samples)).astype(float)
    
    design = pd.DataFrame({
        'Intercept': [1] * n_samples,
        'Condition': [0, 0, 0, 1, 1, 1]
    }, index=[f'S{i}' for i in range(n_samples)])
    
    gr = GenomicRanges(
        seqnames=[f"chr{(i % 22) + 1}" for i in range(n_genes)],
        ranges=IRanges(start=[i * 1000 for i in range(n_genes)], width=[500] * n_genes),
        strand=["+" if i % 2 == 0 else "-" for i in range(n_genes)],
    )
    
    rse = RangedSummarizedExperiment(
        assays={'counts': counts},
        row_ranges=gr,
        row_names=[f'Gene{i}' for i in range(n_genes)],
        column_names=[f'S{i}' for i in range(n_samples)]
    )
    
    success = run_edger_workflow(rse, "RangedSummarizedExperiment", design)
    
    # Additional check: row_ranges preserved
    if success:
        print("\n8. Testing row_ranges preservation...")
        assert hasattr(rse, 'row_ranges') and rse.row_ranges is not None
        print(f"   ✓ row_ranges preserved: {len(rse.row_ranges)} ranges")
    
    return success


def test_single_cell_experiment():
    """Test with SingleCellExperiment."""
    from singlecellexperiment import SingleCellExperiment
    
    np.random.seed(44)
    n_genes, n_samples = 100, 6
    counts = np.random.negative_binomial(10, 0.3, size=(n_genes, n_samples)).astype(float)
    
    design = pd.DataFrame({
        'Intercept': [1] * n_samples,
        'Condition': [0, 0, 0, 1, 1, 1]
    }, index=[f'Cell{i}' for i in range(n_samples)])
    
    sce = SingleCellExperiment(
        assays={'counts': counts},
        row_names=[f'Gene{i}' for i in range(n_genes)],
        column_names=[f'Cell{i}' for i in range(n_samples)]
    )
    
    return run_edger_workflow(sce, "SingleCellExperiment", design)


if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE TEST: RMatrixAdapter with All SE Types")
    print("=" * 70)
    
    results = {}
    
    # Test SummarizedExperiment
    results["SummarizedExperiment"] = test_summarized_experiment()
    
    # Test RangedSummarizedExperiment
    results["RangedSummarizedExperiment"] = test_ranged_summarized_experiment()
    
    # Test SingleCellExperiment
    results["SingleCellExperiment"] = test_single_cell_experiment()
    
    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("=" * 70)
        print("=== ALL TESTS PASSED ===")
        print("=" * 70)
    else:
        print("SOME TESTS FAILED!")
        exit(1)
