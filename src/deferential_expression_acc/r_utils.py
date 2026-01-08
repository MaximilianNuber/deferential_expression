import warnings
import sys

_deps_checked = False

def ensure_r_dependencies():
    """
    Checks if required R packages (edgeR, limma) are installed.
    If not, attempts to install them using BiocManager via rpy2.
    """
    global _deps_checked
    if _deps_checked:
        return

    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        from rpy2.robjects import r
    except ImportError:
        raise ImportError(
            "rpy2 is not installed. Please install it via 'pip install rpy2' "
            "or use the provided conda environment."
        )

    # List of required Bioconductor packages
    required_pkgs = ["edgeR", "limma", "sva"]
    
    # Check which packages are missing
    missing_pkgs = [pkg for pkg in required_pkgs if not rpackages.isinstalled(pkg)]

    if missing_pkgs:
        print(f"Missing R packages detected: {', '.join(missing_pkgs)}")
        print("Attempting to install via BiocManager...")

        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)  # Select first mirror automatically

        # Ensure BiocManager is installed
        if not rpackages.isinstalled("BiocManager"):
            utils.install_packages(StrVector(["BiocManager"]))

        # Install missing packages
        bioc_manager = rpackages.importr("BiocManager")
        bioc_manager.install(StrVector(missing_pkgs), ask=False)
        
        print("R packages installed successfully.")
    else:
        # Optional: Debug logging
        pass

    _deps_checked = True