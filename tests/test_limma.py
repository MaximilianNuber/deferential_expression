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
from deferential_expression import initialize_r
from summarizedexperiment import SummarizedExperiment

# Import limma functions
from deferential_expression.limma import (
    voom,
    normalize_between_arrays,
    remove_batch_effect,
    lm_fit,
    contrasts_fit,
    e_bayes,
    top_table,
    decide_tests,
    treat,
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
    }, index=[f"Sample_{i}" for i in range(6)])
    return design


@pytest.fixture
def mock_se(mock_count_data):
    """Create mock SummarizedExperiment with count data."""
    counts, gene_names, sample_names = mock_count_data
    
    se = SummarizedExperiment(
        assays={'counts': counts},
        row_names=gene_names,
        column_names=sample_names
    )
    
    # Initialize R backing
    se = initialize_r(se, assay='counts')
    return se


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
        log_expr = np.asarray(se_voom.assays["log_expr"])
        assert log_expr.shape == (100, 6)


class TestNormalization:
    """Test normalization functions."""
    
    def test_normalize_between_arrays(self, mock_se, mock_design):
        """Test normalizeBetweenArrays."""
        # First run voom to get log expression
        se_voom = voom(mock_se, design=mock_design)
        
        # Initialize the log_expr assay
        se_voom = initialize_r(se_voom, assay="log_expr")
        
        # Then normalize
        se_norm = normalize_between_arrays(se_voom, assay="log_expr")
        
        assert "log_expr_norm" in se_norm.assay_names


class TestLinearModeling:
    """Test linear modeling functions."""
    
    def test_lm_fit(self, mock_se, mock_design):
        """Test lmFit."""
        se_voom = voom(mock_se, design=mock_design)
        se_voom = initialize_r(se_voom, assay="log_expr")
        
        lm = lm_fit(se_voom, design=mock_design, assay="log_expr")
        
        assert isinstance(lm, LimmaModel)
        assert lm.lm_fit is not None
    
    def test_e_bayes(self, mock_se, mock_design):
        """Test eBayes."""
        se_voom = voom(mock_se, design=mock_design)
        se_voom = initialize_r(se_voom, assay="log_expr")
        
        lm = lm_fit(se_voom, design=mock_design, assay="log_expr")
        lm_eb = e_bayes(lm)
        
        assert isinstance(lm_eb, LimmaModel)
        assert lm_eb.ebayes is not None


class TestDifferentialExpression:
    """Test differential expression extraction."""
    
    def test_top_table(self, mock_se, mock_design):
        """Test topTable."""
        se_voom = voom(mock_se, design=mock_design)
        se_voom = initialize_r(se_voom, assay="log_expr")
        
        lm = lm_fit(se_voom, design=mock_design, assay="log_expr")
        lm_eb = e_bayes(lm)
        
        results = top_table(lm_eb, coef=2, n=20)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 20
    
    def test_decide_tests(self, mock_se, mock_design):
        """Test decideTests."""
        se_voom = voom(mock_se, design=mock_design)
        se_voom = initialize_r(se_voom, assay="log_expr")
        
        lm = lm_fit(se_voom, design=mock_design, assay="log_expr")
        lm_eb = e_bayes(lm)
        
        decisions = decide_tests(lm_eb, p_value=0.05)
        
        assert isinstance(decisions, pd.DataFrame)


class TestCompleteWorkflow:
    """Test complete limma workflow."""
    
    def test_standard_workflow(self, mock_se, mock_design):
        """Test standard limma workflow: voom -> lmFit -> eBayes -> topTable."""
        # Step 1: voom transformation
        se_voom = voom(mock_se, design=mock_design)
        assert "log_expr" in se_voom.assay_names
        
        # Step 2: Initialize R for log_expr
        se_voom = initialize_r(se_voom, assay="log_expr")
        
        # Step 3: Fit linear model
        lm = lm_fit(se_voom, design=mock_design, assay="log_expr")
        assert lm.lm_fit is not None
        
        # Step 4: Empirical Bayes
        lm_eb = e_bayes(lm)
        assert lm_eb.ebayes is not None
        
        # Step 5: Extract results
        results = top_table(lm_eb, coef=2, n=10)
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
