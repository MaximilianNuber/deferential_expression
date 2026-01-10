"""Create a beautiful volcano plot from differential expression results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def volcano_plot(
    results: pd.DataFrame,
    logfc_col: str = "logFC",
    fdr_col: str = "FDR",
    fdr_threshold: float = 0.05,
    logfc_threshold: float = 1.0,
    figsize: tuple = (10, 8),
    title: str = "Volcano Plot",
    xlabel: str = "log₂(Fold Change)",
    ylabel: str = "-log₁₀(Adjusted p-value)",
    save_path: str = None,
    **kwargs
) -> plt.Figure:
    """
    Create a beautiful, publication-quality volcano plot.
    
    Args:
        results: DataFrame with differential expression results
        logfc_col: Column name for log fold change (default: "logFC")
        fdr_col: Column name for adjusted p-value/FDR (default: "FDR")
        fdr_threshold: FDR significance threshold (default: 0.05)
        logfc_threshold: Log fold change threshold for highlighting (default: 1.0)
        figsize: Figure size tuple (default: (10, 8))
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        **kwargs: Additional arguments for customization
            - point_size: Size of points (default: 50)
            - sig_color: Color for significant points (default: '#e74c3c' - red)
            - nonsig_color: Color for non-significant points (default: '#95a5a6' - gray)
            - text_color: Color for axis text (default: '#2c3e50' - dark)
            - edge_colors: Edge color for points (default: 'none')
            - alpha: Transparency (default: 0.7)
            - dpi: DPI for saved figure (default: 300)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    
    Examples:
        >>> fig = volcano_plot(results, title="Treatment vs Control")
        >>> plt.show()
        
        >>> fig = volcano_plot(
        ...     results, 
        ...     fdr_threshold=0.01, 
        ...     logfc_threshold=1.5,
        ...     save_path="volcano.png"
        ... )
    """
    # Extract parameters with defaults
    point_size = kwargs.get('point_size', 50)
    sig_color = kwargs.get('sig_color', '#e74c3c')  # Red
    nonsig_color = kwargs.get('nonsig_color', '#95a5a6')  # Gray
    text_color = kwargs.get('text_color', '#2c3e50')  # Dark blue-gray
    edge_colors = kwargs.get('edge_colors', 'none')
    alpha = kwargs.get('alpha', 0.7)
    dpi = kwargs.get('dpi', 300)
    
    # Create a copy to avoid modifying original
    df = results.copy()
    
    # Calculate -log10(FDR)
    df['-log10(FDR)'] = -np.log10(df[fdr_col])
    
    # Classify points as significant or not
    sig_mask = (df[fdr_col] < fdr_threshold) & (np.abs(df[logfc_col]) > logfc_threshold)
    
    # Create figure with high quality
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Plot non-significant points first (so they appear behind)
    non_sig = df[~sig_mask]
    ax.scatter(
        non_sig[logfc_col],
        non_sig['-log10(FDR)'],
        s=point_size,
        color=nonsig_color,
        alpha=alpha * 0.5,
        edgecolors=edge_colors,
        linewidth=0.5,
        label='Not significant',
        zorder=1
    )
    
    # Plot significant points on top
    sig = df[sig_mask]
    ax.scatter(
        sig[logfc_col],
        sig['-log10(FDR)'],
        s=point_size * 1.3,
        color=sig_color,
        alpha=alpha,
        edgecolors='white',
        linewidth=1,
        label=f'FDR < {fdr_threshold}, |logFC| > {logfc_threshold}',
        zorder=2
    )
    
    # Add threshold lines
    ax.axvline(-logfc_threshold, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)
    ax.axvline(logfc_threshold, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)
    ax.axhline(-np.log10(fdr_threshold), color='#34495e', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold', color=text_color)
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', color=text_color)
    ax.set_title(title, fontsize=15, fontweight='bold', color=text_color, pad=20)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.5)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(colors=text_color, labelsize=11, width=1.5, length=6)
    
    # Legend
    ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        framealpha=0.95
    )
    
    # Add statistics box
    n_sig = sig_mask.sum()
    n_total = len(df)
    stats_text = f'Significant: {n_sig}/{n_total}\n'
    stats_text += f'Up-regulated: {(sig_mask & (df[logfc_col] > 0)).sum()}\n'
    stats_text += f'Down-regulated: {(sig_mask & (df[logfc_col] < 0)).sum()}'
    
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=text_color, linewidth=1.5),
        family='monospace',
        color=text_color
    )
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage with synthetic data
    from src.deferential_expression.edger import EdgeR, calc_norm_factors, estimate_disp, glm_ql_fit, glm_ql_ftest
    from summarizedexperiment import SummarizedExperiment
    from src.deferential_expression.resummarizedexperiment import RESummarizedExperiment
    
    # Create synthetic data
    np.random.seed(42)
    counts = np.random.negative_binomial(5, 0.3, (200, 6))
    design = pd.DataFrame({
        'Intercept': [1] * 6,
        'Condition': [0, 0, 0, 1, 1, 1]
    })
    
    # Run edgeR pipeline
    se = SummarizedExperiment(assays={'counts': counts})
    res = RESummarizedExperiment.from_summarized_experiment(se)
    edger_obj = EdgeR(
        assays=res.assays,
        row_data=res.row_data,
        column_data=res.column_data,
        row_names=res.row_names,
        column_names=res.column_names
    )
    
    obj_norm = calc_norm_factors(edger_obj)
    obj_disp = estimate_disp(obj_norm, design=design, trend="loess")
    fit_result = glm_ql_fit(obj_norm, design=design)
    results = glm_ql_ftest(fit_result, coef=2)
    
    # Create volcano plot
    fig = volcano_plot(
        results,
        title="Differential Expression: Control vs Treatment",
        fdr_threshold=0.05,
        logfc_threshold=0.5,
        save_path="volcano_plot.png"
    )
    
    plt.show()
