# deferential_expression

Python-first access to battle-tested **R/Bioconductor differential expression** tooling — without giving up modern Python data workflows.

This package is meant for the very common situation where:

- you have expression data in Python (`numpy` / `scipy.sparse`, etc.)
- you want **edgeR / limma**-level DE robustness
- you don’t want to rebuild decades of statistical engineering from scratch

Internally, this package uses an `RMatrixAdapter` to bridge Python matrices into R efficiently, and wraps results in a Python-friendly `RESummarizedExperiment` (a Python-side representation aligned with Bioconductor’s `SummarizedExperiment`).

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

## Quickstart: recount3 ➜ edgeR (all from Python)

Below is the basic flow demonstrated in the notebook: create/load data, store counts in an RESummarizedExperiment, run edgeR filtering + normalization + QL pipeline.

Note: In some setups you may need to point R_HOME explicitly, i.e. when using `Positron`.


```python
import os
os.environ["R_HOME"] = "/Users/maximiliannuber/miniconda3/envs/def_exp_test_env/lib/R"  # only if needed

import pandas as pd
from bioc2ri.lazy_r_env import r  # lazy rpy2 env (from bioc2ri)
import deferential_expression as de
```


