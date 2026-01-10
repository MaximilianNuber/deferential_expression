"""R dependency and renv management utilities."""

from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

# Track which packages have been checked
_checked_packages: set = set()


# =============================================================================
# renv management
# =============================================================================

def check_renv(install: bool = False) -> bool:
    """
    Check if renv is installed in R, optionally install it.
    
    Args:
        install: If True, install renv if not found. Default: False.
    
    Returns:
        True if renv is installed (or was successfully installed), False otherwise.
    
    Raises:
        ImportError: If rpy2 is not installed.
    
    Example:
        >>> check_renv()  # Just check
        True
        >>> check_renv(install=True)  # Install if missing
        True
    """
    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
    except ImportError:
        raise ImportError(
            "rpy2 is not installed. Please install it via 'pip install rpy2' "
            "or use the provided conda environment."
        )
    
    is_installed = rpackages.isinstalled("renv")
    
    if not is_installed and install:
        print("renv not found. Installing...")
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(StrVector(["renv"]))
        is_installed = rpackages.isinstalled("renv")
        if is_installed:
            print("renv installed successfully.")
        else:
            print("Failed to install renv.")
    
    return is_installed


def is_renv_installed() -> bool:
    """
    Check if renv is installed in R.
    
    Returns:
        True if renv is installed, False otherwise.
    """
    return check_renv(install=False)


def activate_renv(path: Optional[Union[str, Path]] = None) -> bool:
    """
    Activate an renv environment if one exists.
    
    Checks if renv is installed, then looks for an renv.lock file in the
    specified directory (or current directory). If found, activates the renv.
    
    Args:
        path: Directory containing renv.lock. Default: current working directory.
    
    Returns:
        True if renv was activated, False if no renv found or activation failed.
    
    Raises:
        ImportError: If rpy2 is not installed.
        RuntimeError: If renv is not installed in R.
    
    Example:
        >>> activate_renv()  # Activate renv in current directory
        True
        >>> activate_renv("/path/to/project")  # Activate renv in specific directory
        True
    """
    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import r
    except ImportError:
        raise ImportError(
            "rpy2 is not installed. Please install it via 'pip install rpy2' "
            "or use the provided conda environment."
        )
    
    # Check if renv is installed
    if not rpackages.isinstalled("renv"):
        raise RuntimeError(
            "renv is not installed in R. Install it with: "
            "check_renv(install=True)"
        )
    
    # Determine path
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
    
    # Check for renv directory (more reliable than renv.lock which may not exist with bare=TRUE)
    renv_dir = path / "renv"
    renv_activate = path / "renv" / "activate.R"
    if not renv_dir.exists() or not renv_activate.exists():
        print(f"No renv found in {path} (missing renv/activate.R)")
        return False
    
    # Activate renv using R source
    path_str = str(path.absolute()).replace("\\", "/")
    r(f'setwd("{path_str}")')
    r('source("renv/activate.R")')
    
    # Verify activation
    lib_paths = list(r('.libPaths()'))
    print(f"Activated renv in {path}")
    print(f"Library paths: {lib_paths[:2]}...")  # Show first 2
    return True


def has_renv(path: Optional[Union[str, Path]] = None) -> bool:
    """
    Check if an renv environment exists in the specified directory.
    
    Checks for the existence of renv/activate.R which is more reliable
    than renv.lock (which may not exist with bare initialization).
    
    Args:
        path: Directory to check. Default: current working directory.
    
    Returns:
        True if renv exists, False otherwise.
    """
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
    
    # Check for renv/activate.R
    return (path / "renv" / "activate.R").exists()


def create_renv(path: Union[str, Path]) -> None:
    """
    Create a new renv environment in the specified directory.
    
    Args:
        path: Directory where renv should be initialized.
    
    Raises:
        ImportError: If rpy2 is not installed.
        RuntimeError: If renv is not installed in R.
    
    Example:
        >>> create_renv("/path/to/project")
    """
    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import r
    except ImportError:
        raise ImportError(
            "rpy2 is not installed. Please install it via 'pip install rpy2' "
            "or use the provided conda environment."
        )
    
    # Check if renv is installed
    if not rpackages.isinstalled("renv"):
        raise RuntimeError(
            "renv is not installed in R. Install it with: "
            "check_renv(install=True)"
        )
    
    path = Path(path).absolute()
    path.mkdir(parents=True, exist_ok=True)
    
    path_str = str(path).replace("\\", "/")
    
    # Initialize renv in the directory
    r(f'setwd("{path_str}")')
    r('renv::init(bare = TRUE)')  # bare = TRUE for minimal init
    
    # Activate it
    r('source("renv/activate.R")')
    
    # Show library paths
    lib_paths = list(r('.libPaths()'))
    print(f"Created new renv in {path}")
    print(f"renv library: {lib_paths[0]}")


def get_lib_paths() -> list:
    """
    Get the current R library paths.
    
    Returns:
        List of library paths currently in use by R.
    
    Example:
        >>> get_lib_paths()
        ['/path/to/renv/library/...', '/usr/lib/R/library']
    """
    try:
        from rpy2.robjects import r
    except ImportError:
        raise ImportError("rpy2 is not installed.")
    
    return list(r('.libPaths()'))



# =============================================================================
# Base R dependencies (required for RESummarizedExperiment without modules)
# =============================================================================

# Base R packages needed for the package to work (without edgeR/limma/sva)
BASE_R_PACKAGES = [
    "BiocManager",
    "Matrix",
    "SummarizedExperiment",
    "BiocGenerics",
    "S4Vectors",
    "methods",
    "utils",
]


def install_base_dependencies(
    use_renv: bool = False,
    renv_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Install base R dependencies required for RESummarizedExperiment.
    
    These are the R packages needed before importing any analysis modules
    (edgeR, limma, sva). They include BiocManager, Matrix, SummarizedExperiment,
    BiocGenerics, S4Vectors, methods, and utils.
    
    Args:
        use_renv: If True, install packages into an renv environment.
            Default: False (install to default R library).
        renv_path: Path to renv directory. Only used if use_renv=True.
            Default: current working directory.
    
    Raises:
        ImportError: If rpy2 is not installed.
        RuntimeError: If use_renv=True but no renv found at the specified path.
    
    Example:
        >>> install_base_dependencies()  # Install to default R library
        >>> install_base_dependencies(use_renv=True)  # Install to renv in cwd
        >>> install_base_dependencies(use_renv=True, renv_path="/path/to/project")
    """
    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        from rpy2.robjects import r
    except ImportError:
        raise ImportError(
            "rpy2 is not installed. Please install it via 'pip install rpy2' "
            "or use the provided conda environment."
        )
    
    # Handle renv activation
    if use_renv:
        if renv_path is None:
            renv_path = Path.cwd()
        else:
            renv_path = Path(renv_path)
        
        # Check if renv exists
        if not has_renv(renv_path):
            raise RuntimeError(
                f"No renv.lock found in {renv_path}. "
                f"Create an renv first with: create_renv('{renv_path}')"
            )
        
        # Check if renv package is installed
        if not is_renv_installed():
            raise RuntimeError(
                "renv is not installed in R. Install it with: "
                "check_renv(install=True)"
            )
        
        # Activate renv
        activate_renv(renv_path)
    
    # Check which packages are missing
    # Separate CRAN packages from Bioconductor packages
    cran_packages = ["Matrix", "methods", "utils"]
    bioc_packages = ["SummarizedExperiment", "BiocGenerics", "S4Vectors"]
    
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    
    # Install BiocManager if needed
    if not rpackages.isinstalled("BiocManager"):
        print("Installing BiocManager...")
        utils.install_packages(StrVector(["BiocManager"]))
    
    bioc_manager = rpackages.importr("BiocManager")
    
    # Check for missing CRAN packages
    missing_cran = [pkg for pkg in cran_packages if not rpackages.isinstalled(pkg)]
    if missing_cran:
        print(f"Installing CRAN packages: {', '.join(missing_cran)}")
        utils.install_packages(StrVector(missing_cran))
    
    # Check for missing Bioconductor packages
    missing_bioc = [pkg for pkg in bioc_packages if not rpackages.isinstalled(pkg)]
    if missing_bioc:
        print(f"Installing Bioconductor packages: {', '.join(missing_bioc)}")
        bioc_manager.install(StrVector(missing_bioc), ask=False)
    
    print("Base R dependencies installed successfully.")


# =============================================================================
# Module-specific R dependencies
# =============================================================================

def ensure_r_dependencies(packages: Sequence[str]) -> None:
    """
    Checks if required R packages are installed.
    If not, attempts to install them using BiocManager via rpy2.
    
    Args:
        packages: Sequence of R package names to check/install.
            e.g., ["edgeR"], ["limma", "sva"]
    
    Example:
        >>> ensure_r_dependencies(["edgeR"])  # Check only edgeR
        >>> ensure_r_dependencies(["limma", "sva"])  # Check limma and sva
    """
    global _checked_packages
    
    # Filter to only packages we haven't checked yet
    packages_to_check = [pkg for pkg in packages if pkg not in _checked_packages]
    if not packages_to_check:
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

    # Check which packages are missing
    missing_pkgs = [pkg for pkg in packages_to_check if not rpackages.isinstalled(pkg)]

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

    # Mark all requested packages as checked
    _checked_packages.update(packages_to_check)