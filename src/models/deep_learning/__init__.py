"""Modelos de recomendação baseados em Deep Learning."""

from .two_tower import TwoTowerModel
from .ncf import NCFModel

__all__ = ['TwoTowerModel', 'NCFModel'] 