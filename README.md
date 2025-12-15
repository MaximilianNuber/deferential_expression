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
pip install "deferential_expression @ git+https://github.com/MaximilianNuber/deferential_expression.git@main"
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
from biocpy import SummarizedExperiment
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
    {"condition": conditions},
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
from deferential_expression import RESummarizedExperiment

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

## Differential expression with edgeR

Once you have an RESummarizedExperiment, you can run standard edgeR workflows.

Filtering lowly expressed features
```python
import deferential_expression as de

keep = de.edger.filter_by_expr(rse, group="condition")
rse = rse[keep, :]
```

Normalization factors
```python
rse = de.edger.calc_norm_factors(rse)

rse.col_data.to_pandas()["norm.factors"].describe()
```

Design matrices in Python

Design matrices are constructed in Python, typically using
formulaic_contrasts.

```python
from formulaic_contrasts import FormulaicContrasts

fc = FormulaicContrasts(
    rse.col_data.to_pandas(),
    "~ condition"
)

design = fc.design_matrix
```

This design matrix is then passed into edgeR:
```python
fit = de.edger.glm_ql_fit(rse, design=design)
res = de.edger.glm_ql_ftest(fit, coef=2)
res
```


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


## What this package is (and isn’t)

It is:

- a Python interface to proven R/Bioconductor DE pipelines

- built around BiocPy’s SummarizedExperiment

- explicit about when data crosses the Python ↔ R boundary

It is not:

- a reimplementation of DE methods

- a framework for implicit R execution


## Supported and planned differential expression backends

This package provides **thin, explicit Python interfaces** to established
R/Bioconductor methods. The goal is not to wrap everything at once, but to
expose **well-defined, composable building blocks** that can be used in
reproducible Python workflows.

### Currently supported

- **edgeR**
  - `filterByExpr`
  - TMM normalization (`calcNormFactors`)
  - GLM QL fit and tests (`glmQLFit`, `glmQLFTest`)
  - Designed for count-based bulk RNA-seq and pseudobulk analyses

- **limma**
  - Linear modeling for (log-)expression data
  - Empirical Bayes variance moderation
  - Particularly suitable for voom-transformed RNA-seq and microarray-style data

Both backends operate on an `RESummarizedExperiment`, ensuring consistent
data handling and metadata semantics across methods.

---

### Planned extensions

The package is intentionally structured to grow beyond classical DE testing.
Planned and actively considered additions include:

- **sva**
  - Surrogate variable analysis for latent confounder correction
  - Explicit Python-side control over model matrices and covariates

- **RUVSeq**
  - Remove Unwanted Variation (RUVg / RUVs / RUVr)
  - Integration with control genes or replicate information

These methods naturally complement edgeR and limma workflows and fit the same
container-based design philosophy.

---

### Design principle

Adding a new backend means:

- operating on `RESummarizedExperiment`
- exposing **minimal, method-faithful Python wrappers**
- avoiding reinvention of statistical logic
- keeping the Python ↔ R boundary explicit

This ensures that extensions remain predictable, testable, and aligned with
their Bioconductor counterparts.



