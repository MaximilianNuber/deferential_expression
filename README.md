# deferential_expression

Python-first access to battle-tested **R/Bioconductor differential expression** tooling — without giving up modern Python data workflows.

This package is meant for the very common situation where:

- you work in Python
- your data already lives in a Bioconductor-like container
- you want **edgeR / limma / DESeq2**-level robustness
- you do *not* want to reimplement decades of statistical machinery

The key idea is simple:

> **Users construct a Python `SummarizedExperiment` (from BiocPy).  
> This package converts it into an R-backed representation only when needed.**

> Status: early/active development. APIs may still shift.

---

## Installation

There are two main setups:

### 1) Conda environment (recommended)
Use the provided `environment.yaml` to get Python, R, `rpy2`, and required R packages in one reproducible environment.

```bash
conda env create -f environment.yaml
conda activate def_exp_test_env
pip install -e .
# optional
pip install -e ".[docs]"
```

### 2) System R

If you prefer a system-installed R (instead of conda R), make sure your R setup is compatible with rpy2 and required Bioconductor packages are installed.

Then install the package with pip:
```
pip install "deferential_expression @ git+https://github.com/MaximilianNuber/deferential_expression_acc.git@main"
```

When the R-functions are used, `deferential_expression`automatically checks if required R-packages are installed, and installs if necessary.

## Data model and object conversion

### 1) Start from a Python `SummarizedExperiment` (BiocPy)

Users are expected to **first construct a standard Python `SummarizedExperiment`**
using **BiocPy**. This keeps data preparation, I/O, and metadata handling entirely
on the Python side.

A Python `SummarizedExperiment` contains:

- `assays`: typically NumPy arrays or SciPy sparse matrices
- `row_data`: feature metadata (genes)
- `column_data`: sample metadata

Example:

```python
from summarizedexperiment import SummarizedExperiment
import numpy as np
import pandas as pd

assays = {
    "counts": counts_matrix  # numpy array or scipy sparse
}

row_data = pd.DataFrame(
    {"gene_name": gene_names},
    index=gene_ids
)

col_data = pd.DataFrame(
    {"condition": conditions, "batch": batch_labels},
    index=sample_ids
)

se = SummarizedExperiment(
    assays=assays,
    row_data=row_data,
    column_data=col_data,
)
```

At this stage:

- everything is pure Python
- no R session is required
- no conversion has happened yet


### 2) Convert to RESummarizedExperiment

To run differential expression using R/Bioconductor methods, the Python
SummarizedExperiment is converted into an RESummarizedExperiment:

```python
from deferential_expression_acc import RESummarizedExperiment

rse = RESummarizedExperiment.from_summarized_experiment(se)
```

What this does:

- each assay is converted to an R matrix 
- assays are stored using an internal RMatrixAdapter
- row_data and col_data are preserved
- the object now mirrors an R SummarizedExperiment, but remains usable from Python

Importantly:

The conversion is explicit and one-way by default.
This avoids accidental implicit R dependencies in upstream Python code.

---

## Accessor API

The package provides a modern **accessor pattern** (inspired by pandas/xarray) for accessing R/Bioconductor methods. Accessors are registered dynamically when their module is imported.

### Available Accessors

| Accessor | Import | Description |
|----------|--------|-------------|
| `se.edger` | `import deferential_expression_acc.edger` | edgeR differential expression |
| `se.limma` | `import deferential_expression_acc.limma` | limma linear models |
| `se.sva` | `import deferential_expression_acc.sva` | Batch correction & surrogate variable analysis |

### Design Principles

- **Lazy registration**: Accessors are only available after importing the corresponding module
- **Functional style**: Methods return new objects, original data is unchanged
- **Results stored in SE**: 
  - Matrix results → `assays` (R-backed via `RMatrixAdapter`)
  - Vector results → `column_data` or `row_data`
  - Scalar/complex results → `metadata`

---

## EdgeR Accessor

```python
import deferential_expression_acc.edger  # Register se.edger
```

### Normalization & CPM

```python
# Calculate TMM normalization factors → stored in column_data["norm.factors"]
rse = rse.edger.calc_norm_factors(method="TMM")

# Calculate CPM → stored in assays["cpm"] or assays["logcpm"]
rse = rse.edger.cpm(log=True)
```

### Filtering

```python
# Get boolean mask of genes passing expression filter
mask = rse.edger.filter_by_expr(min_count=10)
rse_filtered = rse[mask, :]
```

### GLM Fitting & Testing

```python
import pandas as pd

# Create design matrix
design = pd.DataFrame({
    'Intercept': [1] * n_samples,
    'Condition': [0, 0, 0, 1, 1, 1]  # Control vs Treatment
}, index=sample_names)

# Fit quasi-likelihood GLM
model = rse.edger.glm_ql_fit(design)

# Run F-test on coefficient (method chaining on model)
results = model.glm_ql_ftest(coef=2)

# Or via accessor
results = rse.edger.glm_ql_ftest(model, coef=2)
```

### Complete EdgeR Workflow

```python
import deferential_expression_acc.edger
import pandas as pd

# Filter, normalize, fit, test
mask = rse.edger.filter_by_expr(min_count=10)
rse = rse[mask, :]
rse = rse.edger.calc_norm_factors(method="TMM")

design = pd.DataFrame({
    'Intercept': [1] * 6,
    'Condition': [0, 0, 0, 1, 1, 1]
}, index=rse.column_names)

model = rse.edger.glm_ql_fit(design)
results = model.glm_ql_ftest(coef=2)  # DataFrame with logFC, PValue, FDR
```

---

## Limma Accessor

```python
import deferential_expression_acc.limma  # Register se.limma
```

### Voom Transformation

```python
# Apply voom transformation → assays["log_expr"] and assays["weights"]
rse_voom = rse.limma.voom(design)
```

### Linear Model Fitting

```python
# Fit linear model
model = rse_voom.limma.lm_fit(design)

# Apply empirical Bayes moderation (method chaining)
model = model.e_bayes()

# Extract top genes
results = model.top_table(n=100)
```

### Contrasts

```python
# Fit with contrast
model = rse_voom.limma.lm_fit(design)
model = model.contrasts_fit([0, 1, -1])  # Contrast vector
model = model.e_bayes()
results = model.top_table()
```

### Batch Correction

```python
# Remove batch effects (preserving biological variation)
rse_bc = rse.limma.remove_batch_effect(
    batch="batch_column",
    design=design  # Biological design to preserve
)

# Quantile normalization
rse_norm = rse.limma.normalize_between_arrays(method="quantile")
```

### TREAT (Threshold-based Testing)

```python
# Test for fold-change > threshold
model = rse_voom.limma.lm_fit(design)
model_treat = model.treat(lfc=1.0)  # Test |logFC| > 1
results = model_treat.top_table()
```

### Complete Limma Workflow

```python
import deferential_expression_acc.limma
import pandas as pd

design = pd.DataFrame({
    'Intercept': [1] * 6,
    'Condition': [0, 0, 0, 1, 1, 1]
}, index=rse.column_names)

# Voom → lmFit → eBayes → topTable
rse_voom = rse.limma.voom(design)
results = (
    rse_voom.limma.lm_fit(design)
    .e_bayes(robust=True)
    .top_table(n=100, adjust_method="BH")
)
```

---

## SVA Accessor

```python
import deferential_expression_acc.sva  # Register se.sva
```

### ComBat Batch Correction (Continuous Data)

```python
# Batch correction for log-expression/CPM data
rse_combat = rse.sva.combat(
    batch="batch_column",
    assay="log_expr",
    output_assay="log_expr_combat"
)
```

### ComBat-seq Batch Correction (Count Data)

```python
# Batch correction for count data (preserves discrete nature)
rse_combat = rse.sva.combat_seq(
    batch="batch_column",
    assay="counts",
    group="condition"  # Biological group to preserve
)
```

### Surrogate Variable Analysis

```python
# Identify hidden batch effects
design = pd.DataFrame({
    'Intercept': [1] * n_samples,
    'Condition': condition_vector
}, index=sample_names)

rse_sva = rse.sva.sva(mod=design, assay="log_expr")

# Results stored in:
# - metadata["sva$sv"]: surrogate variable matrix
# - metadata["sva$n.sv"]: number of SVs found
# - row_data["sva$pprob.gam"]: posterior probabilities

# Extract SVs as DataFrame
sv_df = rse_sva.sva.get_sv()  # columns: SV1, SV2, ...; index: sample names
```

### Complete SVA Workflow

```python
import deferential_expression_acc.sva
import deferential_expression_acc.limma
import pandas as pd

# 1. Batch correction
rse = rse.sva.combat_seq(batch="batch", group="condition")

# 2. Find surrogate variables
design = pd.DataFrame({
    'Intercept': [1] * n_samples,
    'Condition': condition_vector
}, index=sample_names)

rse_sva = rse.sva.sva(mod=design, assay="log_expr")
sv_df = rse_sva.sva.get_sv()

# 3. Add SVs to design matrix for limma
design_with_sv = pd.concat([design, sv_df], axis=1)

# 4. Run limma with SV-adjusted design
rse_voom = rse.limma.voom(design_with_sv)
results = rse_voom.limma.lm_fit(design_with_sv).e_bayes().top_table()
```

---

## Working with feature names

To use gene symbols (or other identifiers) as row names:
```python
rse = rse.set_row_names(rse.row_data["gene_name"])
rse = rse.propagate_dimnames_to_assays()
```

`rse.propagate_dimnames_to_assays()` takes the current row and column names of the `RESummarizedExperiment` and sets the same for each R-matrix in assays.


## Philosophy

This package is deliberately opinionated:

Python is responsible for:

- data loading
- container construction
- metadata wrangling
- design matrices

R is responsible for:

- statistical modeling
- normalization
- hypothesis testing

The boundary is the RESummarizedExperiment. Conversion of large arrays happens once, when `RESummarizedExperiment` is constructed.

No hidden magic.


## What this package is (and isn't)

It is:

- a Python interface to proven R/Bioconductor DE pipelines
- built around BiocPy's SummarizedExperiment
- explicit about when data crosses the Python ↔ R boundary

It is not:

- a reimplementation of DE methods
- a framework for implicit R execution


## Supported backends

| Backend | Status | Functions |
|---------|--------|-----------|
| **edgeR** | ✅ Supported | `filter_by_expr`, `calc_norm_factors`, `cpm`, `glm_ql_fit`, `glm_ql_ftest`, `top_tags` |
| **limma** | ✅ Supported | `voom`, `lm_fit`, `contrasts_fit`, `e_bayes`, `top_table`, `decide_tests`, `treat`, `normalize_between_arrays`, `remove_batch_effect` |
| **sva** | ✅ Supported | `ComBat`, `ComBat_seq`, `sva` |
| **RUVSeq** | 🔜 Planned | Remove Unwanted Variation |
| **DESeq2** | 🔜 Planned | Alternative DE testing |

---

### Design principle

Adding a new backend means:

- operating on `RESummarizedExperiment`
- exposing **minimal, method-faithful Python wrappers**
- avoiding reinvention of statistical logic
- keeping the Python ↔ R boundary explicit

This ensures that extensions remain predictable, testable, and aligned with
their Bioconductor counterparts.
