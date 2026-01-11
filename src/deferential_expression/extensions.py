"""
Extension infrastructure for registering accessors on RESummarizedExperiment.

This module provides a descriptor-based accessor pattern similar to xarray/pandas,
allowing modular extension of RESummarizedExperiment with domain-specific methods.

Example:
    # In edger/accessor.py
    from ..extensions import register_rese_accessor
    
    @register_rese_accessor("edger")
    class EdgeRAccessor:
        def __init__(self, se):
            self._se = se
        
        def calc_norm_factors(self, ...):
            ...

    # Then users can do:
    import deferential_expression.edger  # Triggers registration
    se.edger.calc_norm_factors()
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .resummarizedexperiment import RESummarizedExperiment


class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""


class _CachedAccessor:
    """
    Custom property-like descriptor for caching accessors.
    
    Modeled after xarray's CachedAccessor. When accessed on an instance,
    creates the accessor object once and caches it for future access.
    
    Attributes:
        _name: Name of the accessor.
        _accessor: Accessor class to instantiate.
    """
    
    def __init__(self, name: str, accessor: Type) -> None:
        self._name = name
        self._accessor = accessor
    
    def __get__(self, obj, cls):
        if obj is None:
            # Accessing from the class, return the accessor class itself
            return self._accessor
        
        # Accessing from an instance - use cache
        try:
            cache = obj._cache
        except AttributeError:
            cache = obj._cache = {}
        
        try:
            return cache[self._name]
        except KeyError:
            pass
        
        try:
            accessor_obj = self._accessor(obj)
        except AttributeError as err:
            # Raise as RuntimeError to avoid being swallowed by __getattr__
            raise RuntimeError(f"Error initializing {self._name!r} accessor.") from err
        
        cache[self._name] = accessor_obj
        return accessor_obj


def register_rese_accessor(name: str):
    """
    Register a custom accessor on RESummarizedExperiment.
    
    The accessor is only registered when this decorator is executed,
    which happens when the module containing the accessor class is imported.
    
    Parameters
    ----------
    name : str
        Name under which the accessor should be registered (e.g., "edger", "limma").
        A warning is issued if this name conflicts with a preexisting attribute.
    
    Returns
    -------
    decorator
        A decorator that adds the accessor to RESummarizedExperiment.
    
    Example
    -------
    >>> from deferential_expression.extensions import register_rese_accessor
    >>> 
    >>> @register_rese_accessor("my_tool")
    ... class MyToolAccessor:
    ...     def __init__(self, se):
    ...         self._se = se
    ...     
    ...     def my_method(self):
    ...         return "Hello from my_tool!"
    ...
    >>> # After import: se.my_tool.my_method()
    """
    def decorator(accessor):
        from .resummarizedexperiment import RESummarizedExperiment
        
        if hasattr(RESummarizedExperiment, name):
            warnings.warn(
                f"Registration of accessor {accessor!r} under name {name!r} for type "
                f"RESummarizedExperiment is overriding a preexisting attribute with the same name.",
                AccessorRegistrationWarning,
                stacklevel=2,
            )
        
        setattr(RESummarizedExperiment, name, _CachedAccessor(name, accessor))
        return accessor
    
    return decorator
