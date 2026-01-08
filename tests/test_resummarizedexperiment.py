# tests/test_resummarizedexperiment.py

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rpy2")  # skip cleanly if R/rpy2 is not installed

from biocframe import BiocFrame
from summarizedexperiment import SummarizedExperiment

# ⬇️ adjust this import to your actual module path
from deferential_expression_acc.resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter
from numpy.typing import NDArray

import warnings
warnings.filterwarnings(
    "ignore",
    message='Environment variable "R_SESSION_TMPDIR" redefined by R',
    category=UserWarning,
)

@pytest.fixture
def simple_re_se() -> RESummarizedExperiment:
    """Minimal RESummarizedExperiment with 2x2 assay and dimnames."""
    counts = np.array([[1, 2], [3, 4]], dtype=float)

    row_names = ["g1", "g2"]
    col_names = ["s1", "s2"]

    # BiocFrame expects a dict-like `data`, not a pandas DataFrame here.
    # We only need row_names, so we pass an empty dict + row_names.
    row_bf = BiocFrame({}, row_names=row_names)
    col_bf = BiocFrame({}, row_names=col_names)

    re_se = RESummarizedExperiment(
        assays={"counts": counts},
        row_data=row_bf,
        column_data=col_bf,
        row_names=row_names,
        column_names=col_names,
        metadata={"source": "test"},
    )
    return re_se


def test_set_assay_returns_new_instance_and_is_immutable(simple_re_se):
    orig = simple_re_se
    new = orig.set_assay("logcounts", np.array([[10.0, 20.0], [30.0, 40.0]]))

    # different objects
    assert new is not orig
    assert new.assays is not orig.assays

    # original untouched
    assert "logcounts" not in orig.assay_names
    assert "counts" in orig.assay_names

    # new has both
    assert "logcounts" in new.assay_names
    assert "counts" in new.assay_names


def test_set_assay_from_numpy_creates_rmatrixadapter(simple_re_se):
    new = simple_re_se.set_assay("logcounts", np.array([[10.0, 20.0], [30.0, 40.0]]))

    obj = new.assays["logcounts"]
    assert isinstance(obj, RMatrixAdapter)
    assert obj.shape == (2, 2)

    # roundtrip to numpy works and has correct values
    np.testing.assert_array_equal(
        obj.to_numpy(),
        np.array([[10.0, 20.0], [30.0, 40.0]]),
    )


def test_set_assay_preserves_dimnames_via_self_names(simple_re_se):
    # We created simple_re_se with row_names=["g1","g2"], column_names=["s1","s2"]
    new = simple_re_se.set_assay("logcounts", np.array([[5, 6], [7, 8]]))

    # Ask for pandas view (your .assay(as_pandas=True) helper)
    df = new.assay("logcounts", as_pandas=True)

    assert list(df.index) == ["g1", "g2"]
    assert list(df.columns) == ["s1", "s2"]


def test_set_assay_with_explicit_dimnames(simple_re_se):
    rn = ["geneA", "geneB"]
    cn = ["sampleX", "sampleY"]

    new = simple_re_se.set_assay(
        "logcounts",
        np.array([[1, 2], [3, 4]]),
        rownames=rn,
        colnames=cn,
    )

    df = new.assay("logcounts", as_pandas=True)
    assert list(df.index) == rn
    assert list(df.columns) == cn


def test_set_assay_raises_on_rownames_length_mismatch(simple_re_se):
    arr = np.array([[1, 2], [3, 4]], dtype=float)
    bad_rn = ["only_one"]

    with pytest.raises(ValueError):
        simple_re_se.set_assay("logcounts", arr, rownames=bad_rn)


def test_set_assay_accepts_existing_rmatrixadapter(simple_re_se):
    # Build an adapter by using set_assay once
    temp = simple_re_se.set_assay("logcounts", np.array([[1, 2], [3, 4]]))
    adapter = temp.assays["logcounts"]
    assert isinstance(adapter, RMatrixAdapter)

    # Now re-use this adapter under a new name
    new = simple_re_se.set_assay("vst", adapter)

    assert "vst" in new.assay_names
    assert isinstance(new.assays["vst"], RMatrixAdapter)
    # shape preserved
    assert new.assays["vst"].shape == adapter.shape


def test_slicing_with_ranges():
    """Test slicing with slice objects."""
    counts = np.array([[1, 2, 3, 4], 
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]], dtype=float)
    
    row_names = ["gene1", "gene2", "gene3", "gene4"]
    col_names = ["sample1", "sample2", "sample3", "sample4"]
    
    re_se = RESummarizedExperiment(
        assays={"counts": counts},
        row_data=BiocFrame({}, row_names=row_names),
        column_data=BiocFrame({}, row_names=col_names),
        row_names=row_names,
        column_names=col_names,
    )
    
    # Slice rows
    sliced = re_se[0:2, :]
    assert sliced.shape == (2, 4)
    assert list(sliced.row_names) == ["gene1", "gene2"]
    np.testing.assert_array_equal(
        sliced.assay("counts", as_numpy=True),
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    )
    
    # Slice columns
    sliced = re_se[:, 1:3]
    assert sliced.shape == (4, 2)
    assert list(sliced.column_names) == ["sample2", "sample3"]
    np.testing.assert_array_equal(
        sliced.assay("counts", as_numpy=True),
        np.array([[2, 3], [6, 7], [10, 11], [14, 15]])
    )


def test_slicing_with_single_integer():
    """Test slicing with single integer index (should preserve 2D structure)."""
    counts = np.array([[1, 2, 3, 4], 
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]], dtype=float)
    
    row_names = ["gene1", "gene2", "gene3", "gene4"]
    col_names = ["sample1", "sample2", "sample3", "sample4"]
    
    re_se = RESummarizedExperiment(
        assays={"counts": counts},
        row_data=BiocFrame({}, row_names=row_names),
        column_data=BiocFrame({}, row_names=col_names),
        row_names=row_names,
        column_names=col_names,
    )
    
    # Single row
    sliced = re_se[1, :]
    assert sliced.shape == (1, 4)
    assert list(sliced.row_names) == ["gene2"]
    np.testing.assert_array_equal(
        sliced.assay("counts", as_numpy=True),
        np.array([[5, 6, 7, 8]])
    )
    
    # Single column
    sliced = re_se[:, 2]
    assert sliced.shape == (4, 1)
    assert list(sliced.column_names) == ["sample3"]
    np.testing.assert_array_equal(
        sliced.assay("counts", as_numpy=True),
        np.array([[3], [7], [11], [15]])
    )


def test_slicing_with_boolean_mask():
    """Test slicing with boolean mask."""
    counts = np.array([[1, 2, 3, 4], 
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]], dtype=float)
    
    row_names = ["gene1", "gene2", "gene3", "gene4"]
    col_names = ["sample1", "sample2", "sample3", "sample4"]
    
    re_se = RESummarizedExperiment(
        assays={"counts": counts},
        row_data=BiocFrame({}, row_names=row_names),
        column_data=BiocFrame({}, row_names=col_names),
        row_names=row_names,
        column_names=col_names,
    )
    
    # Boolean mask for rows
    mask = np.array([True, False, True, False])
    sliced = re_se[mask, :]
    assert sliced.shape == (2, 4)
    assert list(sliced.row_names) == ["gene1", "gene3"]
    np.testing.assert_array_equal(
        sliced.assay("counts", as_numpy=True),
        np.array([[1, 2, 3, 4], [9, 10, 11, 12]])
    )


def test_slicing_with_integer_array():
    """Test slicing with integer array."""
    counts = np.array([[1, 2, 3, 4], 
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]], dtype=float)
    
    row_names = ["gene1", "gene2", "gene3", "gene4"]
    col_names = ["sample1", "sample2", "sample3", "sample4"]
    
    re_se = RESummarizedExperiment(
        assays={"counts": counts},
        row_data=BiocFrame({}, row_names=row_names),
        column_data=BiocFrame({}, row_names=col_names),
        row_names=row_names,
        column_names=col_names,
    )
    
    # Integer array for rows
    idx = np.array([0, 2, 3])
    sliced = re_se[idx, :]
    assert sliced.shape == (3, 4)
    assert list(sliced.row_names) == ["gene1", "gene3", "gene4"]
    np.testing.assert_array_equal(
        sliced.assay("counts", as_numpy=True),
        np.array([[1, 2, 3, 4], [9, 10, 11, 12], [13, 14, 15, 16]])
    )


__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"


