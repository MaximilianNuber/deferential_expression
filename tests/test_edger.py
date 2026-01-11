"""
Tests for edgeR functions with actual R conversion using rpy2.

This module tests the edgeR wrapper functions by creating mock data,
converting it to R objects, and verifying the workflow end-to-end.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Any

# Import from the installed package  
from deferential_expression import initialize_r
from deferential_expression.edger import (
    calc_norm_factors,
    cpm,
    filter_by_expr,
    glm_ql_fit,
    glm_ql_ftest,
    top_tags,
    EdgeRModel,
)
from summarizedexperiment import SummarizedExperiment

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
    counts[de_genes, 3:] = counts[de_genes, 3:] * 3  # 3-fold change
    
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


class TestNormalization:
    """Test normalization functions."""
    
    def test_calc_norm_factors(self, mock_se):
        """Test calcNormFactors."""
        se_norm = calc_norm_factors(mock_se, method="TMM")
        
        # Check norm.factors were added
        coldata = se_norm.get_column_data()
        assert "norm.factors" in coldata.column_names
        
        # Check factors are reasonable
        nf = np.asarray(coldata["norm.factors"])
        assert len(nf) == 6
        assert np.all(nf > 0)


class TestCPM:
    """Test CPM functions."""
    
    def test_cpm_basic(self, mock_se):
        """Test basic CPM calculation."""
        se_cpm = cpm(mock_se, log=False)
        
        assert "cpm" in se_cpm.assay_names
        cpm_vals = np.asarray(se_cpm.assays["cpm"])
        assert cpm_vals.shape == (100, 6)
        assert np.all(cpm_vals >= 0)
    
    def test_log_cpm(self, mock_se):
        """Test log-CPM calculation."""
        se_cpm = cpm(mock_se, log=True)
        
        assert "logcpm" in se_cpm.assay_names


class TestFiltering:
    """Test filtering functions."""
    
    def test_filter_by_expr(self, mock_se):
        """Test filterByExpr."""
        mask = filter_by_expr(mock_se, min_count=10)
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == 100


class TestGLMFitting:
    """Test GLM fitting functions."""
    
    def test_glm_ql_fit(self, mock_se, mock_design):
        """Test glmQLFit."""
        se_norm = calc_norm_factors(mock_se)
        model = glm_ql_fit(se_norm, mock_design)
        
        assert isinstance(model, EdgeRModel)
        assert model.fit is not None
        assert model.design is not None
    
    def test_glm_ql_ftest(self, mock_se, mock_design):
        """Test glmQLFTest."""
        se_norm = calc_norm_factors(mock_se)
        model = glm_ql_fit(se_norm, mock_design)
        results = glm_ql_ftest(model, coef=2)
        
        assert isinstance(results, pd.DataFrame)
        assert "logFC" in results.columns
        assert len(results) == 100


class TestCompleteWorkflow:
    """Test complete edgeR workflow."""
    
    def test_standard_workflow(self, mock_se, mock_design):
        """Test standard edgeR workflow."""
        # Normalize
        se = calc_norm_factors(mock_se, method="TMM")
        
        # Filter
        mask = filter_by_expr(se, min_count=5)
        se_filtered = se[mask, :]
        
        # Reinitialize R after subsetting
        se_filtered = initialize_r(se_filtered, assay='counts')
        
        # Fit model
        design = mock_design.loc[se_filtered.column_names]
        model = glm_ql_fit(se_filtered, design)
        
        # Test
        results = model.glm_ql_ftest(coef=2)
        
        assert isinstance(results, pd.DataFrame)
        assert "logFC" in results.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
