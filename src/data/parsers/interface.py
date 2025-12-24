"""
Main interface for ONI save file parsing.

This module provides the primary interface function for parsing ONI save files
as specified in the task requirements.
"""

from typing import Union
from pathlib import Path

from .oni_save_parser import ONISaveParser, GameState


def parse_save(file_path: Union[str, Path]) -> GameState:
    """
    Parse an ONI save file and return structured game state data.
    
    This is the main interface function that implements the required
    `parse_save(file_path: str) -> GameState` interface from the task.
    
    Args:
        file_path: Path to the .sav file (string or Path object)
        
    Returns:
        GameState object containing all extracted game data including:
        - Grid tiles with material states, elements, temperatures
        - Duplicant information (positions, health, stress, skills)
        - Building locations and operational status
        - Resource counts and cycle information
        
    Raises:
        FileNotFoundError: If save file doesn't exist
        RuntimeError: If parsing fails or dependencies are missing
        
    Example:
        >>> game_state = parse_save("Colony001.sav")
        >>> print(f"Cycle: {game_state.cycle}")
        >>> print(f"Duplicants: {len(game_state.duplicants)}")
        >>> print(f"Grid shape: {game_state.grid.shape}")
    """
    # Convert to string if Path object
    if isinstance(file_path, Path):
        file_path = str(file_path)
    
    # Initialize parser and parse the save file
    parser = ONISaveParser()
    return parser.parse_save(file_path)