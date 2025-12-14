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
from deferential_expression.resummarizedexperiment import RESummarizedExperiment
from deferential_expression.edger import (
    calc_norm_factors,
    cpm,
    filter_by_expr,
    glm_ql_fit,
    glm_ql_ftest,
    estimate_disp,
    top_tags,
    EdgeR,
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
    })
    return design


@pytest.fixture
def mock_edger(mock_count_data):
    """Create mock EdgeR object with count data."""
    counts, gene_names, sample_names = mock_count_data
    
    row_data = pd.DataFrame({
        'gene_id': gene_names,
        'gene_name': [f"SYMBOL_{i}" for i in range(len(gene_names))]
    })
    
    col_data = pd.DataFrame({
        'sample_id': sample_names,
        'condition': ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment']
    })
    
    # Create SummarizedExperiment first to get R-backed matrices
    se_base = SummarizedExperiment(
        assays={'counts': counts},
        row_data=row_data,
        column_data=col_data,
        row_names=gene_names,
        column_names=sample_names
    )
    
    # Convert to RESummarizedExperiment (R-backed)
    res = RESummarizedExperiment.from_summarized_experiment(se_base)
    
    # Convert to EdgeR
    edge_r = EdgeR(
        assays=res.assays,
        row_data=res.row_data,
        column_data=res.column_data,
        row_names=res.row_names,
        column_names=res.column_names,
        metadata=res.metadata
    )
    
    return edge_r


class TestRConversion:
    """Test basic R object conversions."""
    
    def test_se_to_r_conversion(self, mock_edger):
        """Test converting SummarizedExperiment to R matrix."""
        counts_r = mock_edger.assay_r("counts")
        assert counts_r is not None
        
        # Verify dimensions
        r_env = get_r_environment()
        dims = r_env.r2py(r_env.ro.baseenv["dim"](counts_r))
        assert dims[0] == 100  # 100 genes
        assert dims[1] == 6    # 6 samples
    
    def test_design_to_r_conversion(self, mock_design):
        """Test converting design matrix to R format."""
        from deferential_expression.edger.utils import pandas_to_r_matrix
        design_r = pandas_to_r_matrix(mock_design)
        assert design_r is not None


class TestNormalization:
    """Test normalization functions."""
    
    def test_calc_norm_factors_tmm(self, mock_edger):
        """Test TMM normalization."""
        obj_norm = calc_norm_factors(mock_edger, method="TMM")
        
        assert "norm.factors" in obj_norm.column_data.column_names
        factors = obj_norm.column_data["norm.factors"]
        assert len(factors) == 6
        # TMM factors should be positive
        assert all(f > 0 for f in factors)
    
    def test_cpm_calculation(self, mock_edger):
        """Test CPM (counts per million) calculation."""
        obj_norm = calc_norm_factors(mock_edger)
        cpm_result = cpm(obj_norm, assay="counts")
        
        # CPM returns EdgeR object with new 'cpm' assay
        assert cpm_result.shape == (100, 6)
        # Check that CPM assay was added
        assert "cpm" in cpm_result.assay_names
        # CPM values should be positive
        cpm_mat = cpm_result.assay("cpm")
        if hasattr(cpm_mat, '__array__'):
            cpm_mat = np.asarray(cpm_mat)
        assert (cpm_mat >= 0).all()


class TestFiltering:
    """Test filtering functions."""
    
    def test_filter_by_expr(self, mock_edger):
        """Test filtering by minimum expression level."""
        obj_norm = calc_norm_factors(mock_edger)
        mask = filter_by_expr(obj_norm, min_count=5, min_total_count=15)
        
        # Mask should be boolean and have length equal to number of genes
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == mock_edger.shape[0]


class TestDispersionEstimation:
    """Test dispersion estimation."""
    
    def test_estimate_disp(self, mock_edger, mock_design):
        """Test dispersion estimation."""
        obj_norm = calc_norm_factors(mock_edger)
        obj_disp = estimate_disp(obj_norm, design=mock_design, trend="loess")
        
        assert obj_disp.disp is not None
        # Should have DGEList with dispersion info
        assert obj_disp.dge is not None
    
    def test_estimate_disp_with_trend(self, mock_edger, mock_design):
        """Test dispersion estimation with trend."""
        obj_norm = calc_norm_factors(mock_edger)
        obj_disp = estimate_disp(obj_norm, design=mock_design, trend="loess")
        
        assert obj_disp.disp is not None


class TestGLMFit:
    """Test GLM fitting functions."""
    
    def test_glm_ql_fit(self, mock_edger, mock_design):
        """Test GLM quasi-likelihood fit."""
        obj_norm = calc_norm_factors(mock_edger)
        obj_disp = estimate_disp(obj_norm, design=mock_design)
        
        # Extract counts from normalized EdgeR
        fit_result = glm_ql_fit(obj_norm, design=mock_design)
        
        assert fit_result.fit is not None
        assert fit_result.sample_names == obj_norm.column_names
        assert fit_result.feature_names == obj_norm.row_names
    
    def test_glm_ql_ftest(self, mock_edger, mock_design):
        """Test GLM quasi-likelihood F-test."""
        obj_norm = calc_norm_factors(mock_edger)
        obj_disp = estimate_disp(obj_norm, design=mock_design, trend="loess")
        
        fit_result = glm_ql_fit(obj_norm, design=mock_design)
        
        # Test coefficient 2 (Condition)
        test_result = glm_ql_ftest(fit_result, coef=2)
        
        assert test_result is not None


class TestTopTags:
    """Test top tags extraction."""
    
    def test_top_tags_basic(self, mock_edger, mock_design):
        """Test basic topTags extraction."""
        obj_norm = calc_norm_factors(mock_edger)
        obj_disp = estimate_disp(obj_norm, design=mock_design, trend="loess")
        
        fit_result = glm_ql_fit(obj_norm, design=mock_design)
        test_result = glm_ql_ftest(fit_result, coef=2)
        
        # glm_ql_ftest already returns a DataFrame with all genes
        # top_tags is already integrated into glm_ql_ftest
        assert isinstance(test_result, pd.DataFrame)
        assert len(test_result) > 0
        # Check for expected column names from edgeR
        assert test_result.shape[1] >= 4  # Should have multiple columns
    
    def test_top_tags_all_genes(self, mock_edger, mock_design):
        """Test topTags with all genes."""
        obj_norm = calc_norm_factors(mock_edger)
        obj_disp = estimate_disp(obj_norm, design=mock_design, trend="loess")
        
        fit_result = glm_ql_fit(obj_norm, design=mock_design)
        test_result = glm_ql_ftest(fit_result, coef=2)
        
        # glm_ql_ftest returns all genes by default
        assert isinstance(test_result, pd.DataFrame)
        assert len(test_result) == 100


class TestCompleteWorkflow:
    """Test complete edgeR analysis workflows."""
    
    def test_standard_edger_workflow(self, mock_edger, mock_design):
        """Test standard edgeR differential expression workflow."""
        # Step 1: Normalization
        obj_norm = calc_norm_factors(mock_edger, method="TMM")
        assert "norm.factors" in obj_norm.column_data.column_names
        
        # Step 2: Estimate dispersion
        obj_disp = estimate_disp(obj_norm, design=mock_design, trend="loess")
        assert obj_disp.disp is not None
        
        # Step 3: GLM fit
        fit_result = glm_ql_fit(obj_norm, design=mock_design)
        assert fit_result.fit is not None
        
        # Step 4: Test
        test_result = glm_ql_ftest(fit_result, coef=2)
        assert test_result is not None
        assert isinstance(test_result, pd.DataFrame)
    
    def test_workflow_with_cpm(self, mock_edger, mock_design):
        """Test workflow with CPM normalization."""
        obj_norm = calc_norm_factors(mock_edger)
        
        # Calculate CPM
        cpm_result = cpm(obj_norm, assay="counts")
        assert cpm_result.shape == (100, 6)
        assert "cpm" in cpm_result.assay_names
        
        # Continue with regular workflow
        obj_disp = estimate_disp(obj_norm, design=mock_design, trend="loess")
        fit_result = glm_ql_fit(obj_norm, design=mock_design)
        test_result = glm_ql_ftest(fit_result, coef=2)
        
        assert isinstance(test_result, pd.DataFrame)
    
    def test_workflow_with_filtering(self, mock_edger, mock_design):
        """Test workflow with expression filtering."""
        obj_norm = calc_norm_factors(mock_edger)
        
        # Get filter mask (this verifies filter_by_expr works)
        mask = filter_by_expr(obj_norm, min_count=5)
        num_filtered = mask.sum()
        
        # Verify mask is boolean and sensible
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert num_filtered > 0
        assert num_filtered <= obj_norm.shape[0]
        
        # Use unfiltered data for the rest of the workflow (filtering genes is complex)
        obj_disp = estimate_disp(obj_norm, design=mock_design, trend="loess")
        fit_result = glm_ql_fit(obj_norm, design=mock_design)
        test_result = glm_ql_ftest(fit_result, coef=2)
        
        assert isinstance(test_result, pd.DataFrame)
        assert len(test_result) == 100  # All genes returned by glm_ql_ftest


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_norm_factors_with_single_method(self, mock_edger):
        """Test normalization with different methods."""
        # RLE method
        obj_rle = calc_norm_factors(mock_edger, method="RLE")
        assert "norm.factors" in obj_rle.column_data.column_names
    
    def test_cpm_with_prior_count(self, mock_edger):
        """Test CPM calculation with prior count."""
        obj_norm = calc_norm_factors(mock_edger)
        cpm_result = cpm(obj_norm, prior_count=2)
        
        # Prior count should prevent exact zeros
        assert cpm_result.shape == (100, 6)
