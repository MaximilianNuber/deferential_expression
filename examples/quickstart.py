# examples/quickstart.py
import pandas as pd
import numpy as np
import deferential_expression as de

# --- make a tiny toy counts matrix (genes x samples) ---
genes = [f"gene{i+1}" for i in range(200)]
samples = [f"S{i+1:02d}" for i in range(12)]
rng = np.random.default_rng(1)
counts = rng.negative_binomial(n=10, p=0.5, size=(len(genes), len(samples)))

# sample meta with a batch and condition
batch = np.array(["b1"]*6 + ["b2"]*6)
cond  = np.array(["A","B"] * 6)

# wrap as R-backed SE (counts can be numpy; edgeR functions expect R matrix, but your
# Limma/EdgeR helpers convert where needed or accept adapters—as in your codebase).
from biocframe import BiocFrame
from summarizedexperiment import SummarizedExperiment

se = SummarizedExperiment(
    assays={"counts": counts},
    row_names=genes,
    column_names=samples,
    column_data=BiocFrame({"batch": batch, "condition": cond})
)

# Use edgeR helpers
edg = de.edger.EdgeR(
    assays=se.assays,
    row_names=se.row_names,
    column_names=se.column_names,
    column_data=se.column_data,
)

# Filter, CPM, design, fit
mask = de.edger.filter_by_expr(edg, design=pd.get_dummies(pd.Series(cond, index=samples), drop_first=True))
edg = edg[mask, :]
edg = edg.cpm(out_name="cpm")

design = pd.get_dummies(
    pd.DataFrame({"condition": cond, "batch": batch}, index=samples),
    drop_first=False
).astype(float)

edg = edg.estimate_disp(design=design)
edg = edg.glm_ql_fit()
edg = edg.glm_qlf_test(coef=1)  # first coefficient (depends on your design)
edg2, tt = edg.top_tags(n=20)
print(tt.head())