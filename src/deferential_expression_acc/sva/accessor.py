"""
SVA accessor for RESummarizedExperiment.

Provides surrogate variable analysis methods via the accessor pattern.

Usage:
    import deferential_expression_acc.sva  # Triggers accessor registration
    
    se = RESummarizedExperiment(...)
    
    # ComBat batch correction (for continuous/normalized data)
    se_combat = se.sva.combat(batch="batch_col", assay="cpm")
    
    # ComBat-seq batch correction (for count data)
    se_combat = se.sva.combat_seq(batch="batch_col", assay="counts")
    
    # Surrogate variable analysis
    se_sva = se.sva.sva(mod=design, assay="cpm")
    sv_matrix = se_sva.sva.get_sv()  # Returns surrogate variables as DataFrame
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union
import numpy as np
import pandas as pd

from ..extensions import register_rese_accessor
from ..resummarizedexperiment import RMatrixAdapter
from ..edger.utils import pandas_to_r_matrix, numpy_to_r_matrix

if TYPE_CHECKING:
    from ..resummarizedexperiment import RESummarizedExperiment


@lru_cache(maxsize=1)
def _prep_sva():
    """Lazily prepare the sva runtime.
    
    Returns:
        Tuple[Any, Any]: (r_env, sva_pkg)
    """
    from bioc2ri.lazy_r_env import get_r_environment
    r = get_r_environment()
    sva_pkg = r.lazy_import_r_packages("sva")
    return r, sva_pkg


@register_rese_accessor("sva")
class SVAAccessor:
    """
    Accessor providing SVA (Surrogate Variable Analysis) methods.
    
    This accessor encapsulates R FFI calls for the sva Bioconductor package.
    
    Methods:
        combat: Batch correction for continuous/normalized data
        combat_seq: Batch correction for count data
        sva: Surrogate variable analysis
        get_sv: Extract surrogate variables from metadata
    
    Attributes:
        _se: Reference to the parent RESummarizedExperiment.
    """
    
    def __init__(self, se: RESummarizedExperiment) -> None:
        self._se = se
        self._r: Any = None
        self._pkg: Any = None
    
    @property
    def _sva(self):
        """Lazy import of sva R package."""
        if self._pkg is None:
            self._r, self._pkg = _prep_sva()
        return self._pkg
    
    @property
    def _renv(self):
        """Lazy access to rpy2 environment."""
        if self._r is None:
            self._r, self._pkg = _prep_sva()
        return self._r
    
    # =========================================================================
    # ComBat - batch correction for continuous data
    # =========================================================================
    
    def combat(
        self,
        batch: Union[str, Sequence, np.ndarray, pd.Series],
        assay: str = "cpm",
        output_assay: Optional[str] = None,
        mod: Optional[pd.DataFrame] = None,
        par_prior: bool = True,
        prior_plots: bool = False,
        mean_only: bool = False,
        ref_batch: Optional[str] = None,
        **kwargs
    ) -> RESummarizedExperiment:
        """
        Apply ComBat batch correction to continuous/normalized expression data.
        
        Wraps `sva::ComBat`. Takes a matrix of continuous values (e.g., log-CPM)
        and returns a batch-corrected matrix of the same dimensions.
        
        Args:
            batch: Batch factor. Either column name in column_data (str) or
                array-like of batch labels.
            assay: Input assay name (should be continuous, e.g., "cpm", "log_expr").
            output_assay: Output assay name. Defaults to "{assay}_combat".
            mod: Optional model matrix for biological covariates to preserve.
            par_prior: Use parametric adjustments. Default: True.
            prior_plots: Show prior distribution plots. Default: False.
            mean_only: Only adjust mean (not variance). Default: False.
            ref_batch: Reference batch level. Default: None.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            New RESummarizedExperiment with batch-corrected assay.
        
        Example:
            >>> se_combat = se.sva.combat(batch="batch_id", assay="log_expr")
        """
        from rpy2.robjects.vectors import StrVector
        
        # Get input matrix
        dat_r = self._se.assay_r(assay)
        
        # Resolve batch to R vector
        batch_r = self._resolve_batch(batch)
        
        # Handle mod (model matrix)
        if mod is not None:
            mod_r = pandas_to_r_matrix(mod)
        else:
            mod_r = self._renv.ro.NULL
        
        # Handle ref.batch
        ref_batch_r = self._renv.ro.NULL if ref_batch is None else ref_batch
        
        # Call ComBat
        result_r = self._sva.ComBat(
            dat_r,
            batch=batch_r,
            mod=mod_r,
            par_prior=par_prior,
            prior_plots=prior_plots,
            mean_only=mean_only,
            ref_batch=ref_batch_r,
            **kwargs
        )
        
        # Store result
        out_name = output_assay if output_assay else f"{assay}_combat"
        return self._se.set_assay(name=out_name, value=RMatrixAdapter(result_r, self._renv))
    
    # =========================================================================
    # ComBat-seq - batch correction for count data
    # =========================================================================
    
    def combat_seq(
        self,
        batch: Union[str, Sequence, np.ndarray, pd.Series],
        assay: str = "counts",
        output_assay: Optional[str] = None,
        group: Optional[Union[str, Sequence, np.ndarray, pd.Series]] = None,
        covar_mod: Optional[pd.DataFrame] = None,
        full_mod: bool = True,
        shrink: bool = False,
        shrink_disp: bool = False,
        gene_subset_n: Optional[int] = None,
        **kwargs
    ) -> RESummarizedExperiment:
        """
        Apply ComBat-seq batch correction to count data.
        
        Wraps `sva::ComBat_seq`. Takes a count matrix and returns a
        batch-corrected count matrix of the same dimensions.
        
        Args:
            batch: Batch factor. Either column name in column_data (str) or
                array-like of batch labels.
            assay: Input count assay name. Default: "counts".
            output_assay: Output assay name. Defaults to "{assay}_combat_seq".
            group: Optional biological group factor to preserve.
            covar_mod: Optional covariate model matrix.
            full_mod: Include group in model matrix. Default: True.
            shrink: Apply shrinkage. Default: False.
            shrink_disp: Shrink dispersion estimates. Default: False.
            gene_subset_n: Number of genes for subset. Default: None.
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            New RESummarizedExperiment with batch-corrected count assay.
        
        Example:
            >>> se_combat = se.sva.combat_seq(batch="batch_id", group="condition")
        """
        # Get input matrix
        counts_r = self._se.assay_r(assay)
        
        # Resolve batch
        batch_r = self._resolve_batch(batch)
        
        # Resolve group if provided
        if group is not None:
            group_r = self._resolve_batch(group)
        else:
            group_r = self._renv.ro.NULL
        
        # Handle covar_mod
        if covar_mod is not None:
            covar_mod_r = pandas_to_r_matrix(covar_mod)
        else:
            covar_mod_r = self._renv.ro.NULL
        
        # Handle gene_subset_n
        gene_subset_n_r = self._renv.ro.NULL if gene_subset_n is None else gene_subset_n
        
        # Call ComBat_seq
        result_r = self._sva.ComBat_seq(
            counts_r,
            batch=batch_r,
            group=group_r,
            covar_mod=covar_mod_r,
            full_mod=full_mod,
            shrink=shrink,
            shrink_disp=shrink_disp,
            gene_subset_n=gene_subset_n_r,
            **kwargs
        )
        
        # Store result
        out_name = output_assay if output_assay else f"{assay}_combat_seq"
        return self._se.set_assay(name=out_name, value=RMatrixAdapter(result_r, self._renv))
    
    # =========================================================================
    # SVA - Surrogate Variable Analysis
    # =========================================================================
    
    def sva(
        self,
        mod: pd.DataFrame,
        assay: str = "cpm",
        mod0: Optional[pd.DataFrame] = None,
        n_sv: Optional[int] = None,
        method: str = "irw",
        vfilter: Optional[int] = None,
        B: int = 5,
        numSVmethod: str = "be",
        **kwargs
    ) -> RESummarizedExperiment:
        """
        Perform Surrogate Variable Analysis.
        
        Wraps `sva::sva`. Identifies and estimates surrogate variables representing
        unknown batch effects or other unwanted variation.
        
        Results are stored in the returned RESummarizedExperiment:
        - Surrogate variable matrix → metadata["sva$sv"] (numpy array)
        - pprob.gam (posterior prob of association) → row_data["sva$pprob.gam"]
        - pprob.b (posterior prob for each sv) → row_data["sva$pprob.b"]
        - n.sv (number of surrogate variables) → metadata["sva$n.sv"]
        
        Use `get_sv()` to extract the surrogate variables as a DataFrame.
        
        Args:
            mod: Full model matrix (samples × covariates) including biological variables.
            assay: Expression assay name (should be continuous). Default: "cpm".
            mod0: Null model matrix (intercept only by default).
            n_sv: Number of surrogate variables. If None, estimated automatically.
            method: SVA method: "irw" (default) or "two-step".
            vfilter: Number of most variable genes to use. Default: None (all genes).
            B: Number of bootstrap iterations for "irw". Default: 5.
            numSVmethod: Method for estimating n.sv: "be" or "leek". Default: "be".
            **kwargs: Additional args forwarded to R function.
        
        Returns:
            New RESummarizedExperiment with SVA results in metadata and row_data.
        
        Example:
            >>> design = pd.DataFrame({'Intercept': [1]*6, 'Cond': [0,0,0,1,1,1]})
            >>> se_sva = se.sva.sva(mod=design, assay="log_expr")
            >>> sv_df = se_sva.sva.get_sv()  # Get surrogate variables as DataFrame
        """
        # Get input matrix
        dat_r = self._se.assay_r(assay)
        
        # Convert mod to R matrix
        mod_r = pandas_to_r_matrix(mod)
        
        # Handle mod0 - default to intercept only
        if mod0 is not None:
            mod0_r = pandas_to_r_matrix(mod0)
        else:
            # Create intercept-only matrix from first column of mod
            mod0_df = mod.iloc[:, [0]]
            mod0_r = pandas_to_r_matrix(mod0_df)
        
        # Handle optional parameters
        n_sv_r = self._renv.ro.NULL if n_sv is None else n_sv
        vfilter_r = self._renv.ro.NULL if vfilter is None else vfilter
        
        # Call sva
        sva_result = self._sva.sva(
            dat_r,
            mod=mod_r,
            mod0=mod0_r,
            n_sv=n_sv_r,
            method=method,
            vfilter=vfilter_r,
            B=B,
            numSVmethod=numSVmethod,
            **kwargs
        )
        
        # Extract results from R list
        # n.sv: number of surrogate variables (extract first to handle edge case)
        n_sv_result = int(self._renv.ro.baseenv["$"](sva_result, "n.sv")[0])
        
        # sv: matrix of surrogate variables (n_samples x n_sv)
        sv_r = self._renv.ro.baseenv["$"](sva_result, "sv")
        sv_np = np.asarray(sv_r)
        # Ensure 2D even if empty
        if sv_np.ndim == 0 or (sv_np.ndim == 1 and n_sv_result == 0):
            sv_np = np.empty((len(self._se.column_names or []), 0))
        elif sv_np.ndim == 1:
            sv_np = sv_np.reshape(-1, 1)
        
        # pprob.gam: posterior probabilities for each gene
        pprob_gam_r = self._renv.ro.baseenv["$"](sva_result, "pprob.gam")
        pprob_gam = np.asarray(pprob_gam_r) if pprob_gam_r is not self._renv.ro.NULL else np.zeros(len(self._se.row_names or []))
        
        # pprob.b: posterior probabilities for each gene for each SV
        pprob_b_r = self._renv.ro.baseenv["$"](sva_result, "pprob.b")
        pprob_b = np.asarray(pprob_b_r) if pprob_b_r is not self._renv.ro.NULL else np.array([])
        
        # Update metadata
        new_metadata = dict(self._se.metadata)
        new_metadata["sva$sv"] = sv_np
        new_metadata["sva$n.sv"] = n_sv_result
        
        # Update row_data with pprob vectors
        row_data = self._se.row_data_df
        if row_data is None:
            row_data = pd.DataFrame(index=self._se.row_names)
        else:
            row_data = row_data.copy()
        
        # Add pprob.gam if available
        if pprob_gam.size > 0:
            row_data["sva$pprob.gam"] = pprob_gam
        
        # pprob.b might be a matrix if multiple SVs, or empty if n.sv=0
        if pprob_b.size > 0:
            if pprob_b.ndim == 1:
                row_data["sva$pprob.b"] = pprob_b
            elif pprob_b.ndim == 2 and pprob_b.shape[1] > 0:
                # Store each column separately for multiple SVs
                for i in range(pprob_b.shape[1]):
                    row_data[f"sva$pprob.b.{i+1}"] = pprob_b[:, i]
        
        # Return new SE with updated metadata and row_data
        from ..resummarizedexperiment import RESummarizedExperiment
        return RESummarizedExperiment(
            assays=dict(self._se.assays),
            row_data=row_data,
            column_data=self._se.column_data_df,
            row_names=self._se.row_names,
            column_names=self._se.column_names,
            metadata=new_metadata,
        )
    
    def get_sv(
        self,
        key: str = "sva$sv",
        as_pandas: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Extract surrogate variables from metadata.
        
        Args:
            key: Metadata key where SV matrix is stored. Default: "sva$sv".
            as_pandas: If True, return as DataFrame with column_names as index
                and SV1, SV2, ... as columns. Default: True.
        
        Returns:
            Surrogate variable matrix as numpy array or pandas DataFrame.
            If no SVs were found (n.sv=0), returns empty array/DataFrame.
        
        Raises:
            KeyError: If the key is not found in metadata.
        
        Example:
            >>> se_sva = se.sva.sva(mod=design)
            >>> sv_df = se_sva.sva.get_sv()  # DataFrame with SVs
            >>> sv_np = se_sva.sva.get_sv(as_pandas=False)  # numpy array
        """
        if key not in self._se.metadata:
            raise KeyError(f"Key '{key}' not found in metadata. Run .sva() first.")
        
        sv_np = self._se.metadata[key]
        
        if not as_pandas:
            return sv_np
        
        # Handle empty case (n.sv=0)
        if sv_np.size == 0 or (sv_np.ndim == 2 and sv_np.shape[1] == 0):
            index = self._se.column_names if self._se.column_names else list(range(sv_np.shape[0] if sv_np.ndim >= 1 else 0))
            return pd.DataFrame(index=index)
        
        # Create DataFrame with proper index and column names
        n_sv = sv_np.shape[1] if sv_np.ndim > 1 else 1
        columns = [f"SV{i+1}" for i in range(n_sv)]
        
        index = self._se.column_names if self._se.column_names else list(range(sv_np.shape[0]))
        
        return pd.DataFrame(sv_np, index=index, columns=columns)
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def _resolve_batch(
        self,
        batch: Union[str, Sequence, np.ndarray, pd.Series]
    ):
        """Resolve batch specification to R character vector."""
        from rpy2.robjects.vectors import StrVector
        
        if isinstance(batch, str):
            # Treat as column name in column_data
            cd = self._se.column_data_df
            if cd is None or batch not in cd.columns:
                raise KeyError(f"Batch column '{batch}' not found in column_data.")
            arr = cd[batch].to_numpy()
        else:
            arr = np.asarray(batch)
        
        # Convert to strings for R factor
        vals = [str(v) for v in arr.tolist()]
        return StrVector(vals)


def activate():
    """
    Called on module import to register the SVA accessor.
    """
    pass  # Registration happens via decorator at class definition
