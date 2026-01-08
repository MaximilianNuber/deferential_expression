import pandas as pd
import numpy as np
from typing import Any, Optional, Sequence, Union

# Singleton manager for lazy rpy2 imports
import functools
from typing import Any, Callable, Dict, Optional, Type, Union, Protocol

import pandas as pd
import numpy as np

# --- 1. Rpy2Manager: The central, lazy-loaded rpy2 host ---
# This class incorporates all imports and setup from your original lazy_import_r_env.

class Rpy2Manager:
    _instance = None
    _initialized = False

    # Attributes to be populated on first initialization
    _ro: Any = None
    _numpy2ri_converter: Any = None
    _pandas2ri_converter: Any = None
    _conversion: Any = None
    _importr: Any = None
    _STAP: Any = None
    _RRuntimeError: Any = None
    _localconverter_func: Any = None
    _default_converter_instance: Any = None
    _get_conversion_func: Any = None
    _ListVector: Any = None
    _IntVector: Any = None
    _FloatVector: Any = None
    _StrVector: Any = None
    _BoolVector: Any = None
    _scipy2ri: Any = None
    _utils_pkg: Any = None
    _methods_pkg: Any = None
    _lazy_import_r_packages_func: Any = None
    _py2r_func: Any = None
    _r2py_func: Any = None


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            print("Rpy2Manager: Initializing rpy2 components for workflow...")
            
            # --- START: Content from your original lazy_import_r_env ---
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr, STAP
            from rpy2.rinterface_lib.embedded import RRuntimeError
            from rpy2.robjects import conversion
            from rpy2.robjects import pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects import default_converter
            from rpy2.robjects.conversion import get_conversion
            from rpy2.robjects.vectors import ListVector, IntVector, FloatVector, StrVector, BoolVector
            
            # Assuming anndata2ri is installed and desired
            try:
                from anndata2ri import scipy2ri 
            except ImportError:
                print("Warning: 'anndata2ri' not found. scipy2ri will not be available.")
                scipy2ri = None # Set to None if not found

            # Your own converters & lazy-loader (assuming these are in a sibling module)
            # Adjust this import path '.rpy_conversions' if your file structure is different.
            try:
                from deferential_expression_acc.rpy2_conversions import _py_to_r, _r_to_py, lazy_import_r_packages
            except ImportError:
                print("Warning: 'rpy_conversions.py' not found. Custom _py_to_r, _r_to_py, lazy_import_r_packages will be replaced by dummies.")
                # Provide dummy functions if they are strictly necessary for init to complete
                _py_to_r = lambda x: x # Identity function as a fallback
                _r_to_py = lambda x: x # Identity function as a fallback
                lazy_import_r_packages = lambda pkgs: (None, None) # Dummy for package import

            # Load the R "utils" and "methods" packages using your lazy_import_r_packages
            # Only if lazy_import_r_packages is not a dummy
            utils, methods_pkg = (None, None)
            if lazy_import_r_packages is not None and lazy_import_r_packages.__code__.co_code != (lambda: (None,None)).__code__.co_code:
                 utils, methods_pkg = lazy_import_r_packages(["utils", "methods"])

            # --- END: Content from your original lazy_import_r_env ---

            # Assign to instance attributes
            self._ro = ro
            self._importr = importr
            self._STAP = STAP
            self._RRuntimeError = RRuntimeError
            self._conversion = conversion
            self._pandas2ri_converter = pandas2ri.converter # Directly store converter instance
            self._numpy2ri_converter = numpy2ri.converter # Directly store converter instance
            self._localconverter_func = localconverter
            self._default_converter_instance = default_converter
            self._get_conversion_func = get_conversion
            self._ListVector = ListVector
            self._IntVector = IntVector
            self._FloatVector = FloatVector
            self._StrVector = StrVector
            self._BoolVector = BoolVector
            self._scipy2ri = scipy2ri
            self._utils_pkg = utils
            self._methods_pkg = methods_pkg
            self._lazy_import_r_packages_func = lazy_import_r_packages
            self._py2r_func = _py_to_r
            self._r2py_func = _r_to_py

            self._initialized = True
            print("Rpy2Manager: rpy2 components initialized.")

    # --- Properties to access the loaded components ---
    @property
    def ro(self) -> Any:
        self.__init__()
        return self._ro

    @property
    def r(self) -> Any: # Convenience for ro.r
        self.__init__()
        return self._ro.r

    @property
    def importr(self) -> Any:
        self.__init__()
        return self._importr

    @property
    def STAP(self) -> Any:
        self.__init__()
        return self._STAP

    @property
    def RRuntimeError(self) -> Any:
        self.__init__()
        return self._RRuntimeError

    @property
    def conversion(self) -> Any:
        self.__init__()
        return self._conversion

    @property
    def pandas2ri_converter(self) -> Any: # Exposes pandas2ri.converter directly
        self.__init__()
        return self._pandas2ri_converter

    @property
    def numpy2ri_converter(self) -> Any: # Exposes numpy2ri.converter directly
        self.__init__()
        return self._numpy2ri_converter

    @property
    def scipy2ri(self) -> Any:
        self.__init__()
        return self._scipy2ri

    @property
    def localconverter(self) -> Any:
        self.__init__()
        return self._localconverter_func

    @property
    def default_converter(self) -> Any:
        self.__init__()
        return self._default_converter_instance
    
    @property
    def get_conversion(self) -> Any:
        self.__init__()
        return self._get_conversion_func

    @property
    def ListVector(self) -> Any:
        self.__init__()
        return self._ListVector

    @property
    def IntVector(self) -> Any:
        self.__init__()
        return self._IntVector

    @property
    def FloatVector(self) -> Any:
        self.__init__()
        return self._FloatVector

    @property
    def StrVector(self) -> Any:
        self.__init__()
        return self._StrVector

    @property
    def BoolVector(self) -> Any:
        self.__init__()
        return self._BoolVector

    @property
    def utils_pkg(self) -> Any:
        self.__init__()
        return self._utils_pkg

    @property
    def methods_pkg(self) -> Any:
        self.__init__()
        return self._methods_pkg

    @property
    def lazy_import_r_packages(self) -> Any:
        self.__init__()
        return self._lazy_import_r_packages_func

    @property
    def py2r(self) -> Any: # Your custom py2r function
        self.__init__()
        return self._py2r_func

    @property
    def r2py(self) -> Any: # Your custom r2py function
        self.__init__()
        return self._r2py_func
    

from typing import Any, Callable, ContextManager, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Rpy2ManagerProto(Protocol):
    """
    Minimal protocol for the rpy2 manager/namespace that get_r_environment()
    returns. This is structural: anything with these attributes will satisfy it.
    """

    # Core rpy2 "ro" namespace (rpy2.robjects)
    ro: Any

    # Converters / conversion helpers
    numpy2ri: Any      # usually rpy2.robjects.numpy2ri
    pandas2ri: Any     # usually rpy2.robjects.pandas2ri
    conversion: Any    # rpy2.robjects.conversion
    default_converter: Any

    # Context manager factory: localconverter(...)
    def localconverter(self, conv) -> ContextManager[None]: ...

    # Function returning the active conversion object
    def get_conversion(self) -> Any: ...

    # R object constructors
    ListVector: Any
    IntVector: Any
    FloatVector: Any
    StrVector: Any
    BoolVector: Any

    # Error / utility bits
    RRuntimeError: type[Exception]
    utils_pkg: Any
    methods_pkg: Any

    # High-level helpers you expose
    def lazy_import_r_packages(self, packages: Sequence[str]) -> None: ...
    def py2r(self, obj: Any) -> Any: ...
    def r2py(self, sexp: Any) -> Any: ...
