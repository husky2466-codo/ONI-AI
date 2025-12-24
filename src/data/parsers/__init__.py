"""
Data parsers for ONI save files.

This module provides parsers for extracting structured data from ONI save files
using the oni-save-parser JavaScript library through a Python wrapper.
"""

from .oni_save_parser import ONISaveParser, GameState, Duplicant, Building
from .interface import parse_save

__all__ = ['ONISaveParser', 'GameState', 'Duplicant', 'Building', 'parse_save']