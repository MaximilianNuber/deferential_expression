# deferential_expression

Python-first access to battle-tested **R/Bioconductor differential expression** tooling — without giving up modern Python data workflows.

This package is meant for the very common situation where:

- you work in Python
- your data already lives in a Bioconductor-like container
- you want **edgeR / limma / DESeq2**-level robustness
- you do *not* want to reimplement decades of statistical machinery

The key idea is simple:

> **Users work with BiocPy's `SummarizedExperiment` (or subclasses like `RangedSummarizedExperiment`, `SingleCellExperiment`).  
> This package converts assays to R-backed `RMatrixAdapter` objects, enabling seamless R function calls while keeping Python in control.**

> Status: early/active development. APIs may still shift.

---

## Installation

### Conda environment (recommended)

```bash
conda env create -f environment.yaml
conda activate deferential_expression
pip install -e .
```

### Pip with System R

```bash
pip install "deferential_expression @ git+https://github.com/MaximilianNuber/deferential_expression.git@main"
```

R packages (edgeR, limma, sva) are automatically checked and installed when needed.

---

## Quick Start

```python
from summarizedexperiment import SummarizedExperiment
from deferential_expression import initialize_r
import deferential_expression.edger as edger
import pandas as pd
import numpy as np

# Create data using BiocPy's SummarizedExperiment
counts = np.random.negative_binomial(10, 0.3, size=(1000, 6))
se = SummarizedExperiment(
    assays={'counts': counts},
    row_names=[f'Gene{i}' for i in range(1000)],
    column_names=['S1','S2','S3','S4','S5','S6']
)

# Initialize R backing - converts counts assay to RMatrixAdapter
se = initialize_r(se, assay='counts')

# Now use edgeR functions directly
design = pd.DataFrame({
    'Intercept': [1]*6,
    'Condition': [0,0,0,1,1,1]
}, index=se.column_names)

se = edger.calc_norm_factors(se, method="TMM")
model = edger.glm_ql_fit(se, design)
results = edger.glm_ql_ftest(model, coef=2)
```

### Works with RangedSummarizedExperiment too!

```python
from summarizedexperiment import RangedSummarizedExperiment
from genomicranges import GenomicRanges
from iranges import IRanges

# Create genomic ranges for features
gr = GenomicRanges(
    seqnames=['chr1'] * 1000,
    ranges=IRanges(start=list(range(0, 1000000, 1000)), width=[500] * 1000)
)

# RangedSummarizedExperiment - class type is preserved!
rse = RangedSummarizedExperiment(
    assays={'counts': counts},
    row_ranges=gr,
    row_names=[f'Gene{i}' for i in range(1000)],
    column_names=['S1','S2','S3','S4','S5','S6']
)

rse = initialize_r(rse, assay='counts')
rse = edger.calc_norm_factors(rse)  # Returns RangedSummarizedExperiment!
```

---

## Core Concept: `initialize_r()` and `RMatrixAdapter`

The `RMatrixAdapter` is the bridge between Python and R:

```python
from deferential_expression import initialize_r, RMatrixAdapter

# Before: assay is a numpy array
se = SummarizedExperiment(assays={'counts': counts_array}, ...)
type(se.assays['counts'])  # numpy.ndarray

# After: assay is an RMatrixAdapter wrapping an R matrix
se = initialize_r(se, assay='counts')
type(se.assays['counts'])  # RMatrixAdapter

# RMatrixAdapter behaves like a numpy array
np.asarray(se.assays['counts'])  # Works!
se.assays['counts'].sum(axis=0)   # Computed in R!
se.assays['counts'].shape         # (1000, 6)
```

---

## EdgeR Module

```python
import deferential_expression.edger as edger
from deferential_expression import initialize_r

se = initialize_r(se, assay='counts')
```

### Normalization

```python
se = edger.calc_norm_factors(se, method="TMM")
```

### CPM

```python
se = edger.cpm(se, log=True)
# Access: np.asarray(se.assays['logcpm'])
```

### Filtering

```python
mask = edger.filter_by_expr(se, min_count=10)
se_filtered = se[mask, :]
```

### GLM Fitting & Testing

```python
model = edger.glm_ql_fit(se, design)
results = edger.glm_ql_ftest(model, coef=2)

# Or use method chaining on EdgeRModel
results = model.glm_ql_ftest(coef=2)
```

### Complete EdgeR Workflow

```python
import deferential_expression.edger as edger
from deferential_expression import initialize_r
import pandas as pd

# Initialize R backing
se = initialize_r(se, assay='counts')

# Filter, normalize, fit, test
mask = edger.filter_by_expr(se, min_count=10)
se = se[mask, :]
se = edger.calc_norm_factors(se, method="TMM")

design = pd.DataFrame({
    'Intercept': [1] * 6,
    'Condition': [0, 0, 0, 1, 1, 1]
}, index=se.column_names)

model = edger.glm_ql_fit(se, design)
results = model.glm_ql_ftest(coef=2)  # DataFrame with logFC, PValue, FDR
sig_genes = results[results["FDR"] < 0.05]
```

---

## Limma Module

```python
import deferential_expression.limma as limma
from deferential_expression import initialize_r
```

### Voom Transformation

```python
se = initialize_r(se, assay='counts')
se_voom = limma.voom(se, design)
# Adds log_expr and weights assays
```

### Linear Model Fitting

```python
se = initialize_r(se, assay='log_expr')
model = limma.lm_fit(se, design)
model = limma.e_bayes(model)
results = limma.top_table(model, n=100)

# Method chaining on LimmaModel
model = limma.lm_fit(se, design)
results = model.e_bayes().top_table(n=100)
```

### Contrasts

```python
model = limma.lm_fit(se, design)
model = model.contrasts_fit([0, 1, -1])
results = model.e_bayes().top_table()
```

### Batch Correction

```python
# Quantile normalization
se = initialize_r(se, assay='log_expr')
se = limma.normalize_between_arrays(se, method="quantile")
```

### TREAT (Threshold-based Testing)

```python
model = limma.lm_fit(se, design)
results = model.treat(lfc=1.0).top_table()  # Test |logFC| > 1
```

### Complete Limma Workflow

```python
import deferential_expression.limma as limma
from deferential_expression import initialize_r
import pandas as pd

se = initialize_r(se, assay='counts')

design = pd.DataFrame({
    'Intercept': [1] * 6,
    'Condition': [0, 0, 0, 1, 1, 1]
}, index=se.column_names)

# Voom → lmFit → eBayes → topTable
se_voom = limma.voom(se, design)
se_voom = initialize_r(se_voom, assay='log_expr')

results = (
    limma.lm_fit(se_voom, design, assay='log_expr')
    .e_bayes(robust=True)
    .top_table(n=100, adjust_method="BH")
)
```

---

## SVA Module

```python
import deferential_expression.sva as sva
from deferential_expression import initialize_r
```

### ComBat Batch Correction (Continuous Data)

```python
se = initialize_r(se, assay='log_expr')
se = sva.combat(se, batch="batch_column", assay="log_expr")
```

### ComBat-seq Batch Correction (Count Data)

```python
se = initialize_r(se, assay='counts')
se = sva.combat_seq(se, batch="batch_column", group="condition")
```

---

## Lazy Loading & R Dependencies

Submodules are **lazily loaded** - R dependencies are only checked when you actually import them:

```python
import deferential_expression  # No R check yet

import deferential_expression.edger  # NOW checks/installs edgeR
import deferential_expression.limma  # NOW checks/installs limma
import deferential_expression.sva    # NOW checks/installs sva
```

This means you can:
- Work with SummarizedExperiment in environments without R
- Only trigger R dependency checks for the modules you actually use

---

## Philosophy

**Python is responsible for:**
- Data loading & container construction
- Metadata wrangling
- Design matrices

**R is responsible for:**
- Statistical modeling
- Normalization
- Hypothesis testing

The boundary is `initialize_r()` + `RMatrixAdapter`. Conversion of large arrays happens once.

---

## Supported SE Types

| Type | Supported |
|------|-----------|
| `SummarizedExperiment` | ✅ |
| `RangedSummarizedExperiment` | ✅ |
| `SingleCellExperiment` | ✅ |

Class type is **preserved** through all operations - if you pass an RSE, you get an RSE back!

---

## Supported Backends

| Backend | Status | Functions |
|---------|--------|-----------|
| **edgeR** | ✅ | `calc_norm_factors`, `cpm`, `filter_by_expr`, `glm_ql_fit`, `glm_ql_ftest`, `top_tags` |
| **limma** | ✅ | `voom`, `lm_fit`, `contrasts_fit`, `e_bayes`, `top_table`, `decide_tests`, `treat`, `normalize_between_arrays`, `remove_batch_effect` |
| **sva** | ✅ | `combat`, `combat_seq`, `sva` |
| **RUVSeq** | 🔜 | Planned |
| **DESeq2** | 🔜 | Planned |
