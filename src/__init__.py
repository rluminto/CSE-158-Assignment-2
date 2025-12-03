"""
Food.com Recipe Recommender System
Source code package for data processing, modeling, and evaluation.
"""

__version__ = "1.0.0"
__author__ = "CSE 158 Assignment 2"

from . import data_utils
from . import features
from . import models
from . import eval_utils

__all__ = ['data_utils', 'features', 'models', 'eval_utils']
