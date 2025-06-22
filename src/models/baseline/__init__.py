# src/models/baseline/__init__.py
"""Modelos baseline."""

from .global_mean import GlobalMeanModel
from .popularity import PopularityModel

__all__ = ['GlobalMeanModel', 'PopularityModel']
