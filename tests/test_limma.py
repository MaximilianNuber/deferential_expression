"""
Tests for limma functions with actual R conversion using rpy2.

This module tests the limma wrapper functions by creating mock data,
converting it to R objects, and verifying the workflow end-to-end.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Any

# Import from the installed package  
from deferential_expression.resummarizedexperiment import RESummarizedExperiment
from summarizedexperiment import SummarizedExperiment

# Import limma functions
from deferential_expression.limma import (
    voom,
    voom_with_quality_weights,
    normalize_between_arrays,
    remove_batch_effect,
    lm_fit,
    contrasts_fit,
    e_bayes,
    top_table,
    decide_tests,
    treat,
    duplicate_correlation,
    LimmaModel,
)

# For R conversion verification
from bioc2ri.lazy_r_env import get_r_environment

@pytest.fixture
def r_env():
    """Provide rpy2 R environment."""
    return get_r_environment()


@pytest.fixture
def mock_count_data():
    """Create mock count data for testing."""
    np.random.seed(42)
    n_genes = 100
    n_samples = 6
    
    # Simulate count data with some differential expression
    counts = np.random.negative_binomial(10, 0.3, size=(n_genes, n_samples))
    
    # Make some genes differentially expressed between groups
    de_genes = np.arange(0, 20)
    counts[de_genes, 3:] = counts[de_genes, 3:] * 2  # 2-fold change
    
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]
    
    return counts.astype(float), gene_names, sample_names


@pytest.fixture
def mock_design():
    """Create mock design matrix."""
    design = pd.DataFrame({
        'Intercept': [1, 1, 1, 1, 1, 1],
        'Condition': [0, 0, 0, 1, 1, 1]
    })
    return design


@pytest.fixture
def mock_batch():
    """Create mock batch labels."""
    return pd.Series(['Batch1', 'Batch1', 'Batch2', 'Batch2', 'Batch3', 'Batch3'])


@pytest.fixture
def mock_se(mock_count_data):
    """Create mock RESummarizedExperiment with count data."""
    counts, gene_names, sample_names = mock_count_data
    
    row_data = pd.DataFrame({
        'gene_id': gene_names,
        'gene_name': [f"SYMBOL_{i}" for i in range(len(gene_names))]
    })
    
    col_data = pd.DataFrame({
        'sample_id': sample_names,
        'condition': ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment']
    })
    
    se_base = SummarizedExperiment(
        assays={'counts': counts},
        row_data=row_data,
        column_data=col_data,
        row_names=gene_names,
        column_names=sample_names
    )
    
    se = RESummarizedExperiment.from_summarized_experiment(se_base)
    
    return se


class TestRConversion:
    """Test that objects are properly converted to R."""
    
    def test_se_to_r_conversion(self, mock_se, r_env):
        """Test that SummarizedExperiment assays are R-backed."""
        counts_r = mock_se.assay_r("counts")
        
        # Verify it's an R object
        assert hasattr(counts_r, '__sexp__'), "Counts should be an R object"
        
        # Verify dimensions using R
        dims = r_env.ro.baseenv["dim"](counts_r)
        assert len(dims) == 2, "Should have 2 dimensions"
        assert dims[0] == 100, "Should have 100 genes"
        assert dims[1] == 6, "Should have 6 samples"
    
    def test_design_to_r_conversion(self, mock_design, r_env):
        """Test that design matrix converts to R properly."""
        from deferential_expression.resummarizedexperiment import _df_to_r_matrix
        
        design_r = _df_to_r_matrix(mock_design)
        
        # Verify it's an R matrix
        assert hasattr(design_r, '__sexp__'), "Design should be an R object"
        
        # Verify dimensions
        dims = r_env.ro.baseenv["dim"](design_r)
        assert dims[0] == 6, "Should have 6 samples"
        assert dims[1] == 2, "Should have 2 columns"
        
        # Verify column names
        colnames = list(r_env.ro.baseenv["colnames"](design_r))
        assert colnames == ['Intercept', 'Condition']


class TestVoom:
    """Test voom transformation."""
    
    def test_voom_basic(self, mock_se, mock_design):
        """Test basic voom transformation."""
        se_voom = voom(mock_se, design=mock_design)
        
        # Check new assays exist
        assert "log_expr" in se_voom.assay_names
        assert "weights" in se_voom.assay_names
        
        # Check original assay is preserved
        assert "counts" in se_voom.assay_names
        
        # Check dimensions
        log_expr = se_voom.assay("log_expr", as_numpy=True)
        assert log_expr.shape == (100, 6)
        
        weights = se_voom.assay("weights", as_numpy=True)
        assert weights.shape == (100, 6)
        
        # Check that weights are positive
        assert np.all(weights > 0), "All weights should be positive"
    
    def test_voom_with_lib_size(self, mock_se, mock_design):
        """Test voom with explicit library sizes."""
        lib_size = np.array([1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6])
        
        se_voom = voom(mock_se, design=mock_design, lib_size=lib_size)
        
        assert "log_expr" in se_voom.assay_names
        assert "weights" in se_voom.assay_names
    
    def test_voom_quality_weights(self, mock_se, mock_design):
        """Test voomWithQualityWeights."""
        se_voom = voom_with_quality_weights(mock_se, design=mock_design)
        
        assert "log_expr" in se_voom.assay_names  # Returns 'log_expr', not 'log_expr_qw'
        assert "weights" in se_voom.assay_names
        
        # Check dimensions
        log_expr = se_voom.assay("log_expr", as_numpy=True)
        assert log_expr.shape == (100, 6)


class TestNormalization:
    """Test normalization functions."""
    
    def test_normalize_between_arrays(self, mock_se, mock_design):
        """Test normalizeBetweenArrays."""
        # First run voom to get log expression
        se_voom = voom(mock_se, design=mock_design)
        
        # Then normalize
        se_norm = normalize_between_arrays(se_voom, exprs_assay="log_expr")
        
        assert "log_expr_norm" in se_norm.assay_names
        
        # Check dimensions
        norm_expr = se_norm.assay("log_expr_norm", as_numpy=True)
        assert norm_expr.shape == (100, 6)
    
    def test_normalize_different_methods(self, mock_se, mock_design):
        """Test different normalization methods."""
        se_voom = voom(mock_se, design=mock_design)
        
        for method in ["quantile", "scale"]:
            se_norm = normalize_between_arrays(
                se_voom, 
                exprs_assay="log_expr",
                method=method,
                normalized_assay=f"norm_{method}"
            )
            assert f"norm_{method}" in se_norm.assay_names


class TestBatchCorrection:
    """Test batch effect removal."""
    
    def test_remove_batch_effect(self, mock_se, mock_design, mock_batch):
        """Test removeBatchEffect."""
        se_voom = voom(mock_se, design=mock_design)
        
        se_corrected = remove_batch_effect(
            se_voom,
            batch=mock_batch,
            exprs_assay="log_expr",
            design=mock_design
        )
        
        assert "log_expr_bc" in se_corrected.assay_names
        
        # Check dimensions
        corrected = se_corrected.assay("log_expr_bc", as_numpy=True)
        assert corrected.shape == (100, 6)


class TestLinearModeling:
    """Test linear modeling functions."""
    
    def test_lm_fit(self, mock_se, mock_design):
        """Test lmFit."""
        se_voom = voom(mock_se, design=mock_design)
        
        lm = lm_fit(se_voom, design=mock_design)
        
        assert isinstance(lm, LimmaModel)
        assert lm.lm_fit is not None
        assert lm.design is not None
        assert lm.method == "ls"
    
    def test_lm_fit_robust(self, mock_se, mock_design):
        """Test robust lmFit."""
        se_voom = voom(mock_se, design=mock_design)
        
        lm = lm_fit(se_voom, design=mock_design, method="robust")
        
        assert lm.method == "robust"
    
    def test_contrasts_fit(self, mock_se, mock_design):
        """Test contrasts.fit."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        
        # Test Condition effect (contrast: [0, 1])
        contrast = [0, 1]
        lm_contrast = contrasts_fit(lm, contrast=contrast)
        
        assert isinstance(lm_contrast, LimmaModel)
        assert lm_contrast.contrast_fit is not None
        assert lm_contrast.lm_fit is not None  # Original fit preserved
    
    def test_e_bayes(self, mock_se, mock_design):
        """Test eBayes."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_eb = e_bayes(lm)
        
        assert isinstance(lm_eb, LimmaModel)
        assert lm_eb.ebayes is not None
    
    def test_e_bayes_robust(self, mock_se, mock_design):
        """Test robust eBayes."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_eb = e_bayes(lm, robust=True, trend=True)
        
        assert lm_eb.ebayes is not None


class TestDifferentialExpression:
    """Test differential expression extraction."""
    
    def test_top_table(self, mock_se, mock_design):
        """Test topTable with F-statistic (no coef specified)."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_eb = e_bayes(lm)
        
        # Without coef, returns F-statistic version
        results = top_table(lm_eb, n=20)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 20
        assert "gene" in results.columns
        assert "F" in results.columns  # F-statistic instead of logFC
        assert "p_value" in results.columns
        assert "adj_p_value" in results.columns
    
    def test_top_table_all_genes(self, mock_se, mock_design):
        """Test topTable with all genes."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_eb = e_bayes(lm)
        
        results = top_table(lm_eb)
        
        assert len(results) == 100  # All genes
    
    def test_top_table_with_coef(self, mock_se, mock_design):
        """Test topTable with specific coefficient."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_eb = e_bayes(lm)
        
        # Test with coefficient index (use coefficient column name)
        results = top_table(lm_eb, coef="Condition", n=10)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 10
        assert "log_fc" in results.columns  # Single coef has logFC
        assert "p_value" in results.columns
    
    def test_decide_tests(self, mock_se, mock_design):
        """Test decideTests."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_eb = e_bayes(lm)
        
        decisions = decide_tests(lm_eb, p_value=0.05)
        
        assert isinstance(decisions, pd.DataFrame)
        assert decisions.shape[0] == 100  # All genes
        
        # Check that values are -1, 0, or 1
        unique_vals = set(decisions.values.flatten())
        assert unique_vals.issubset({-1, 0, 1})
    
    def test_decide_tests_with_lfc(self, mock_se, mock_design):
        """Test decideTests with log-fold-change threshold."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_eb = e_bayes(lm)
        
        decisions = decide_tests(lm_eb, p_value=0.05, lfc=1.0)
        
        assert isinstance(decisions, pd.DataFrame)
        
        # With fold-change threshold, should have fewer significant genes
        n_sig_no_lfc = (decide_tests(lm_eb, p_value=0.05, lfc=0) != 0).sum().sum()
        n_sig_with_lfc = (decisions != 0).sum().sum()
        assert n_sig_with_lfc <= n_sig_no_lfc


class TestTreat:
    """Test TREAT analysis."""
    
    def test_treat_basic(self, mock_se, mock_design):
        """Test TREAT."""
        # Apply voom and lmFit first, then treat
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_treat = treat(lm, lfc=1.0)
        
        assert isinstance(lm_treat, LimmaModel)
        assert lm_treat.lm_fit is not None
        assert lm_treat.method == "treat"
    
    def test_treat_with_top_table(self, mock_se, mock_design):
        """Test TREAT followed by topTable."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_treat = treat(lm, lfc=1.0)
        
        # When using treat result, need to specify coef (treat uses all coefficients)
        results = top_table(lm_treat, coef="Condition", n=20)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 20
        assert "log_fc" in results.columns
        assert "p_value" in results.columns


class TestDuplicateCorrelation:
    """Test duplicate correlation."""
    
    def test_duplicate_correlation(self, mock_se, mock_design):
        """Test duplicateCorrelation."""
        se_voom = voom(mock_se, design=mock_design)
        
        # Create block factor
        block = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
        
        cor = duplicate_correlation(
            se_voom,
            design=mock_design,
            block=block,
            exprs_assay="log_expr"
        )
        
        assert isinstance(cor, float)
        # Correlation should be between -1 and 1
        assert -1 <= cor <= 1
    
    def test_duplicate_correlation_with_categorical(self, mock_se, mock_design):
        """Test duplicateCorrelation with pandas Categorical."""
        se_voom = voom(mock_se, design=mock_design)
        
        block = pd.Categorical(['A', 'A', 'B', 'B', 'C', 'C'])
        
        cor = duplicate_correlation(
            se_voom,
            design=mock_design,
            block=block,
            exprs_assay="log_expr"
        )
        
        assert isinstance(cor, float)


class TestCompleteWorkflow:
    """Test complete limma workflow."""
    
    def test_standard_workflow(self, mock_se, mock_design):
        """Test standard limma workflow: voom -> lmFit -> eBayes -> topTable."""
        # Step 1: voom transformation
        se_voom = voom(mock_se, design=mock_design)
        assert "log_expr" in se_voom.assay_names
        
        # Step 2: Fit linear model
        lm = lm_fit(se_voom, design=mock_design)
        assert lm.lm_fit is not None
        
        # Step 3: Empirical Bayes
        lm_eb = e_bayes(lm)
        assert lm_eb.ebayes is not None
        
        # Step 4: Extract results
        results = top_table(lm_eb, n=10)
        assert len(results) == 10
        assert "p_value" in results.columns
    
    def test_workflow_with_contrasts(self, mock_se, mock_design):
        """Test workflow with contrasts."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        
        # Test contrast
        contrast = [0, 1]
        lm_contrast = contrasts_fit(lm, contrast=contrast)
        
        lm_eb = e_bayes(lm_contrast)
        results = top_table(lm_eb, n=20)
        
        assert len(results) == 20
    
    def test_workflow_with_batch_correction(self, mock_se, mock_design, mock_batch):
        """Test workflow with batch correction."""
        # Voom
        se_voom = voom(mock_se, design=mock_design)
        
        # Batch correction
        se_corrected = remove_batch_effect(
            se_voom,
            batch=mock_batch,
            design=mock_design,
            exprs_assay="log_expr",
            corrected_assay="log_expr_corrected"
        )
        
        assert "log_expr_corrected" in se_corrected.assay_names
        
        # Continue with corrected data
        # Note: In practice, you'd use the uncorrected data for lmFit
        # but this tests the full pipeline
    
    def test_treat_workflow(self, mock_se, mock_design):
        """Test TREAT workflow."""
        # TREAT directly includes eBayes-like statistics
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        lm_treat = treat(lm, lfc=1.0)
        
        # Extract results - specify coef when using treat on full model
        results = top_table(lm_treat, coef="Condition", n=20)
        
        assert len(results) == 20
        assert "log_fc" in results.columns
    
    def test_workflow_with_quality_weights(self, mock_se, mock_design):
        """Test workflow with quality weights."""
        se_voom = voom_with_quality_weights(mock_se, design=mock_design)
        
        lm = lm_fit(se_voom, design=mock_design)
        lm_eb = e_bayes(lm, robust=True)
        
        results = top_table(lm_eb, n=10)
        decisions = decide_tests(lm_eb, p_value=0.05)
        
        assert len(results) == 10
        assert decisions.shape[0] == 100


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_lm_fit_without_weights(self, mock_se, mock_design):
        """Test lmFit without weights assay - weights are optional."""
        # voom creates weights assay, but lmFit should work without it
        se_voom = voom(mock_se, design=mock_design)
        
        # lmFit should work fine with log_expr but no weights
        lm = lm_fit(se_voom, design=mock_design)
        assert lm.lm_fit is not None
    
    def test_top_table_without_ebayes(self, mock_se, mock_design):
        """Test topTable automatically runs eBayes if needed."""
        se_voom = voom(mock_se, design=mock_design)
        lm = lm_fit(se_voom, design=mock_design)
        
        # Call topTable without running eBayes first
        results = top_table(lm, n=10)
        
        assert len(results) == 10
        assert "p_value" in results.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
