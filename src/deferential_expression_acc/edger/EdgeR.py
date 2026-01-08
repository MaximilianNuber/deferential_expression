from typing import Optional, Sequence, Union, Tuple
import numpy as np
import pandas as pd

from biocframe import BiocFrame
# from deferential_expression_acc.rpy2_manager import Rpy2Manager
from functools import lru_cache

from ..resummarizedexperiment import RESummarizedExperiment, RMatrixAdapter

# ------------------------------------------------------------------
# EdgeR class: stores R fit objects + fluent ops returning new EdgeR
# ------------------------------------------------------------------
class EdgeR(RESummarizedExperiment):
    """``RESummarizedExperiment`` subclass adding slots for edgeR model state.

    Attributes:
        dge: R ``DGEList`` object (after construction/normalization).
        disp: R object holding dispersion estimates (e.g., from ``estimateDisp``).
        glm: R GLM fit object from ``edgeR::glmQLFit``.
        lrt: R test object from ``edgeR::glmQLFTest``.
    """
    __slots__ = ("dge", "disp", "glm", "lrt")  # add others as you need

    def __init__(self, *, assays = None, row_data=None, column_data=None,
                 row_names=None, column_names=None, metadata=None, **kwargs):
        """Initialize an ``EdgeR`` container with standard SE fields.

        Args:
            assays: Mapping of assay names to arrays or R-backed matrices.
            row_data: Optional feature annotations.
            column_data: Optional sample annotations.
            row_names: Optional feature names.
            column_names: Optional sample names.
            metadata: Optional free-form metadata dictionary.
            **kwargs: Forwarded to ``RESummarizedExperiment``.
        """
        # initialize R object slots
        self.dge = None
        self.disp = None
        self.glm = None
        self.lrt = None
        super().__init__(assays=assays,
                         row_data=row_data,
                         column_data=column_data,
                         row_names=row_names,
                         column_names=column_names,
                         metadata=metadata,
                         **kwargs)

    # ---------- internal helper to clone ----------
    def _clone(self,
               *,
               assays=None,
               row_data=None,
               column_data=None,
               metadata=None,
               dge=None,
               disp=None,
               glm=None,
               lrt=None):
        """Clone the object with optional replacements and updated edgeR slots.

        Args:
            assays: Optional replacement assays mapping.
            row_data: Optional replacement row annotations.
            column_data: Optional replacement column annotations.
            metadata: Optional replacement metadata dictionary.
            dge: Optional replacement DGE object.
            disp: Optional replacement dispersion object.
            glm: Optional replacement GLM fit.
            lrt: Optional replacement test object.

        Returns:
            EdgeR: A new ``EdgeR`` with fields replaced as requested.
        """
        return EdgeR(
            assays=assays if assays is not None else dict(self.assays),
            row_data=row_data if row_data is not None else self.row_data,
            column_data=column_data if column_data is not None else self.column_data,
            row_names=self.row_names,
            column_names=self.column_names,
            metadata=metadata if metadata is not None else dict(self.metadata)
        )._set_r_objs(dge if dge is not None else self.dge,
                      disp if disp is not None else self.disp,
                      glm if glm is not None else self.glm,
                      lrt if lrt is not None else self.lrt)

    def _set_r_objs(self, dge, disp, glm, lrt):
        """Set edgeR-related R objects on the instance (fluent).

        Args:
            dge: R ``DGEList`` object.
            disp: R dispersion fit object.
            glm: R GLM fit object from ``glmQLFit``.
            lrt: R test object from ``glmQLFTest``.

        Returns:
            EdgeR: The same instance for chaining.
        """
        self.dge, self.disp, self.glm, self.lrt = dge, disp, glm, lrt
        return self

    @property
    def samples(self) -> pd.DataFrame:
        """pandas.DataFrame: Column (sample) annotations as pandas."""
        return self.column_data.to_pandas()

    # ------------------- edgeR ops -------------------

    def filter_by_expr(self,
                       design: Optional[pd.DataFrame] = None,
                       assay: str = "counts",
                       **kwargs) -> Tuple["EdgeR", np.ndarray]:
        """Compute an expression filter mask using ``edgeR::filterByExpr``.

        Args:
            design: Optional design matrix (samples × covariates) as pandas.
            assay: Name of the counts assay to evaluate.
            **kwargs: Additional args forwarded to ``filterByExpr``.

        Returns:
            np.ndarray: Boolean mask of rows to keep.

        Notes:
            The code currently returns only the boolean mask. Subsetting and
            DGE updates below are unreachable due to the early ``return``.
        """
        edger = edgeR_pkg()
        counts_r = self.assay_r(assay)
        # samples_r = _df_to_r_df(self.column_data.to_pandas() or pd.DataFrame(index=np.arange(counts_r.ncol)))
        # dge = edger.DGEList(counts=counts_r, samples=samples_r)

        design_r = _r.ro.NULL if design is None else _df_to_r_matrix(design)
        mask_r = edger.filterByExpr(counts_r, design=design_r, **kwargs)
        mask = np.array(mask_r, dtype=bool)

        cd = self.column_data

        return mask

        # subset all assays row-wise via BaseSE slicing (mask length = rows)
        new_se = self[mask, :]
        # keep same R objects? We built a temp dge above; update dge to filtered
        bracket = _r.ro.baseenv["["]
        dge_filtered = edger.DGEList(counts=bracket(counts_r, mask_r, _r.ro.NULL),
                                     samples=samples_r)
        return new_se._clone(dge=dge_filtered), mask

    def calc_norm_factors(self,
                          assay: str = "counts",
                          **kwargs) -> "EdgeR":
        """Compute TMM normalization factors and store them in column data.

        Args:
            assay: Name of the counts assay.
            **kwargs: Additional args forwarded to ``edgeR::calcNormFactors``
                when called on a ``DGEList``.

        Returns:
            EdgeR: A cloned object with updated ``column_data`` containing
            ``edgeR_norm_factors`` and with ``dge`` set to the updated DGEList.
        """
        edger = edgeR_pkg()
        counts_r = self.assay_r(assay)
        samples_r = _df_to_r_df(self.column_data_df() or pd.DataFrame(index=np.arange(counts_r.ncol)))
        dge = edger.DGEList(counts=counts_r, samples=samples_r)
        dge = edger.calcNormFactors(dge, **kwargs)

        # pull norm factors
        factors = np.array(dge.rx2("samples").rx2("norm.factors"))
        col_df = self.column_data_df().copy() if self.column_data_df() is not None else pd.DataFrame(index=list(_r.ro.baseenv["colnames"](counts_r)))
        col_df["edgeR_norm_factors"] = factors
        return self._clone(column_data=col_df, dge=dge)

    def cpm(self,
            assay: str = "counts",
            log: bool = True,
            prior_count: float = 1.0,
            out_name: str = "cpm",
            **kwargs) -> "EdgeR":
        """Compute CPM (optionally log-CPM) using ``edgeR::cpm`` and add as assay.

        Args:
            assay: Source assay name (counts).
            log: Whether to compute log-CPM.
            prior_count: Prior count for log-CPM.
            out_name: Name for the output assay.
            **kwargs: Additional arguments forwarded to ``edgeR::cpm``.

        Returns:
            EdgeR: A cloned object with the new CPM assay (R-backed).
        """
        edger = edgeR_pkg()
        counts_r = self.assay_r(assay)
        cpm_r = edger.cpm(counts_r, log=log, prior_count=prior_count, **kwargs)
        # wrap as R matrix adapter and drop straight into assays
        assays = dict(self.assays)
        assays[out_name] = RMatrixAdapter(cpm_r, _r)
        return self._clone(assays=assays)

    def estimate_disp(self,
                      design: pd.DataFrame,
                      assay: str = "counts",
                      **kwargs) -> "EdgeR":
        """Estimate dispersion using ``edgeR::estimateDisp``.

        Args:
            design: Design matrix (samples × covariates) as pandas.
            assay: Counts assay name.
            **kwargs: Additional arguments forwarded to ``estimateDisp``.

        Returns:
            EdgeR: A cloned object with ``dge`` and ``disp`` set.
        """
        edger = edgeR_pkg()
        counts_r = self.assay_r(assay)
        samples_r = _df_to_r_df(self.col_data_df() or pd.DataFrame(index=np.arange(counts_r.ncol)))
        dge = edger.DGEList(counts=counts_r, samples=samples_r)
        dge = edger.calcNormFactors(dge)  # ensure norm factors
        design_r = _df_to_r_matrix(design)
        disp = edger.estimateDisp(dge, design=design_r, **kwargs)
        return self._clone(dge=dge, disp=disp)

    def glm_ql_fit(self,
                   design: Optional[pd.DataFrame] = None,
                   **kwargs) -> "EdgeR":
        """Fit the quasi-likelihood GLM via ``edgeR::glmQLFit``.

        Args:
            design: Optional design matrix. If ``disp`` is missing, this is
                required to compute it on-the-fly via ``estimate_disp``.
            **kwargs: Additional args forwarded to ``glmQLFit``.

        Returns:
            EdgeR: A cloned object with ``glm`` set.

        Raises:
            ValueError: If ``disp`` is not present and no ``design`` is provided.
        """
        if self.disp is None:
            if design is None:
                raise ValueError("No dispersion in object and no design provided.")
            # fallback: compute disp on the fly
            tmp = self.estimate_disp(design, **kwargs)
            return tmp.glm_ql_fit(design=None, **kwargs)

        edger = edgeR_pkg()
        glm = edger.glmQLFit(self.disp, design=self.disp.rx2("design"), **kwargs)
        return self._clone(glm=glm)

    def glm_qlf_test(self,
                     contrast: Sequence[float] | None = None,
                     coef: Optional[Union[int, str]] = None,
                     **kwargs) -> "EdgeR":
        """Run ``edgeR::glmQLFTest`` on the stored GLM fit.

        Args:
            contrast: Optional numeric contrast vector.
            coef: Optional coefficient index/name to test (alternative to ``contrast``).
            **kwargs: Additional args forwarded to ``glmQLFTest``.

        Returns:
            EdgeR: A cloned object with ``lrt`` set.

        Raises:
            ValueError: If neither ``contrast`` nor ``coef`` is provided.
            ValueError: If ``glm`` has not been fitted yet.
        """
        if self.glm is None:
            raise ValueError("Run glm_ql_fit first.")
        edger = edgeR_pkg()
        if contrast is not None:
            with _r.localconverter(_r.default_converter + _r.numpy2ri.converter):
                contrast_r = _r.get_conversion().py2rpy(np.asarray(contrast, dtype=float))
            lrt = edger.glmQLFTest(self.glm, contrast=contrast_r, **kwargs)
        else:
            # coef-based test
            if coef is None:
                raise ValueError("Provide either `contrast` or `coef`.")
            lrt = edger.glmQLFTest(self.glm, coef=coef, **kwargs)
        return self._clone(lrt=lrt)

    def top_tags(self,
                 n: Optional[int] = None,
                 adjust_method: str = "BH",
                 sort_by: str = "PValue",
                 **kwargs) -> Tuple["EdgeR", pd.DataFrame]:
        """Extract top results using ``edgeR::topTags`` and return a pandas table.

        Args:
            n: Number of rows to return; if ``None``, uses ``Inf``.
            adjust_method: Multiple-testing method (e.g., ``"BH"``).
            sort_by: Sorting key (e.g., ``"PValue"``).
            **kwargs: Additional args forwarded to ``topTags``.

        Returns:
            Tuple[EdgeR, pandas.DataFrame]: A cloned object (with metadata note)
            and a DataFrame with columns standardized to ``gene``, ``p_value``,
            ``log_fc``, and ``adj_p_value``.

        Raises:
            ValueError: If ``glm_qlf_test`` has not been run.
        """
        if self.lrt is None:
            raise ValueError("Run glm_qlf_test first.")
        edger = edgeR_pkg()
        n_val = _r.ro.r("Inf") if n is None else int(n)
        tt = edger.topTags(self.lrt, n=n_val, adjust_method=adjust_method, sort_by=sort_by, **kwargs)
        # extract table
        table_r = tt.rx2("table") if "table" in list(tt.names) else _r.ro.baseenv["as.data.frame"](tt)
        with _r.localconverter(_r.default_converter + _r.pandas2ri.converter):
            df = _r.get_conversion().rpy2py(table_r)
        df = df.reset_index().rename(columns={
            "index": "gene",
            "PValue": "p_value",
            "logFC": "log_fc",
            "FDR": "adj_p_value"
        })
        # store R table? you can put into metadata if desired
        meta = dict(self.metadata)
        meta["edgeR_topTags_last"] = {"n": n, "adjust_method": adjust_method, "sort_by": sort_by}
        return self._clone(metadata=meta), df