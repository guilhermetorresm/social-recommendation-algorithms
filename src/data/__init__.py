# src/data/__init__.py
"""Módulos de dados."""

from .data_loader import DatasetLoader, MovieLens100KLoader
from .preprocessor import DataPreprocessor
from .splitter import DataSplitter

__all__ = ['DatasetLoader', 'MovieLens100KLoader', 'DataPreprocessor', 'DataSplitter']