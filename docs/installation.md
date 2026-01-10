# Installation Guide

`deferential_expression` relies on R and specific Bioconductor packages (`edgeR`, `limma`, `sva`). You can install these dependencies automatically using Conda (recommended) or manually using a system-installed R.

## Method 1: Conda (Recommended)

The easiest way to install all dependencies, including R and the required R packages, is using Conda.

1.  Ensure you have Miniforge or Anaconda installed.
2.  Create the environment using the provided `environment.yaml` file:

    ```bash
    conda env create -f environment.yaml
    conda activate deferential_expression
    ```

3.  Install the package in editable mode:

    ```bash
    pip install -e .
    ```

This method automatically handles the installation of `r-base`, `rpy2`, `edgeR`, `limma`, and `sva`.

## Method 2: Pip with System R

If you prefer to use your system's R installation, follow these steps.

### Prerequisites

1.  **Install R**: Ensure R (version 4.0 or higher) is installed on your system.
    *   **macOS**: `brew install r` or download from CRAN.
    *   **Linux**: Use your package manager (e.g., `sudo apt install r-base`).
    *   **Windows**: Download from CRAN.

2.  **Set R_HOME (Optional but Recommended)**:
    If `rpy2` fails to find your R installation, you may need to set the `R_HOME` environment variable.
    
    To find your R home, run `R RHOME` in your terminal.
    
    ```bash
    export R_HOME=$(R RHOME)
    ```

### Installation

1.  Install the package via pip:

    ```bash
    pip install "deferential_expression @ git+https://github.com/MaximilianNuber/deferential_expression.git@main"
    ```

2.  **Automatic R Package Installation**:
    When you first run the package, it will attempt to detect if `edgeR`, `limma`, and `sva` are installed in your R library. If they are missing, it will attempt to install them using `BiocManager`.
    
    *Note: You may need write permissions to your R library, or R will ask to create a user library.*
