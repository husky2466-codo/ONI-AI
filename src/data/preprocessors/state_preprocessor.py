"""
State Preprocessor for ONI AI Agent.

This module implements the data preprocessing pipeline to convert parsed
GameState objects into ML-ready tensor formats with proper normalization
and multi-channel representation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings

from ..parsers.oni_save_parser import GameState, Duplicant, Building


@dataclass
class StateTensor:
    """
    ML-ready tensor representation of game state.
    
    This dataclass contains the preprocessed game state in tensor format
    suitable for neural network training and inference.
    """
    spatial: np.ndarray  # (height, width, channels) - spatial game state
    global_features: np.ndarray  # (64,) - global state vector
    metadata: Dict[str, Any]  # Additional information about preprocessing
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the spatial tensor."""
        return self.spatial.shape
    
    @property
    def channels(self) -> int:
        """Get the number of channels in the spatial tensor."""
        return self.spatial.shape[2] if len(self.spatial.shape) == 3 else 0


class StatePreprocessor:
    """
    Preprocessor for converting GameState to ML-ready tensors.
    
    This class handles the conversion of raw game state data into normalized,
    multi-channel tensor representations suitable for machine learning models.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (64, 64),
                 temperature_range: Tuple[float, float] = (-50.0, 200.0),
                 max_element_id: int = 255,
                 max_building_types: int = 20):
        """
        Initialize the state preprocessor.
        
        Args:
            target_size: Target spatial dimensions (height, width)
            temperature_range: Min and max temperatures for normalization
            max_element_id: Maximum element ID for normalization
            max_building_types: Maximum number of building types for encoding
        """
        self.target_size = target_size
        self.temperature_range = temperature_range
        self.max_element_id = max_element_id
        self.max_building_types = max_building_types
        
        # Channel configuration for spatial tensor
        self.channel_config = {
            'solid': 0,      # Material state: solid
            'liquid': 1,     # Material state: liquid  
            'gas': 2,        # Material state: gas
            'element_id': 3, # Element ID (normalized)
            'temperature': 4, # Temperature (normalized)
            'building': 5,   # Building presence/type
            'duplicant': 6   # Duplicant positions
        }
        self.num_channels = len(self.channel_config)
        
        # Building type mapping for categorical encoding
        self.building_type_map = {
            'power': 1,
            'plumbing': 2,
            'ventilation': 3,
            'agriculture': 4,
            'living': 5,
            'research': 6,
            'storage': 7,
            'infrastructure': 8,
            'other': 9
        }
    
    def preprocess_state(self, game_state: GameState) -> StateTensor:
        """
        Convert GameState to ML-ready StateTensor.
        
        Args:
            game_state: Parsed game state from ONI save file
            
        Returns:
            StateTensor with spatial and global features
            
        Raises:
            ValueError: If game state is invalid or preprocessing fails
        """
        # Validate input
        self._validate_game_state(game_state)
        
        # Create spatial tensor
        spatial_tensor = self._create_spatial_tensor(game_state)
        
        # Create global features vector
        global_features = self._create_global_features(game_state)
        
        # Create metadata
        metadata = self._create_metadata(game_state)
        
        return StateTensor(
            spatial=spatial_tensor,
            global_features=global_features,
            metadata=metadata
        )
    
    def _validate_game_state(self, game_state: GameState) -> None:
        """Validate that the game state is suitable for preprocessing."""
        if game_state is None:
            raise ValueError("GameState cannot be None")
        
        if game_state.grid is None:
            raise ValueError("GameState must have a grid")
        
        if len(game_state.grid.shape) != 3:
            raise ValueError(f"Grid must be 3D (height, width, channels), got shape {game_state.grid.shape}")
        
        if game_state.world_size is None or len(game_state.world_size) != 2:
            raise ValueError("GameState must have valid world_size (height, width)")
        
        # Warn about potential issues but don't fail
        if len(game_state.duplicants) == 0:
            warnings.warn("No duplicants found in game state")
        
        if len(game_state.buildings) == 0:
            warnings.warn("No buildings found in game state")
    
    def _create_spatial_tensor(self, game_state: GameState) -> np.ndarray:
        """
        Create the spatial tensor from game state grid and objects.
        
        Creates a multi-channel tensor with:
        - Channel 0: Solid material state
        - Channel 1: Liquid material state  
        - Channel 2: Gas material state
        - Channel 3: Element ID (normalized)
        - Channel 4: Temperature (normalized)
        - Channel 5: Building type (categorical)
        - Channel 6: Duplicant positions
        """
        # Get original grid dimensions
        original_height, original_width = game_state.world_size
        target_height, target_width = self.target_size
        
        # Initialize output tensor
        spatial_tensor = np.zeros((target_height, target_width, self.num_channels), dtype=np.float32)
        
        # Resize/sample the original grid to target size
        if game_state.grid.shape[:2] == self.target_size:
            # Direct copy if sizes match
            base_grid = game_state.grid.copy()
        else:
            # Downsample or upsample the grid
            base_grid = self._resize_grid(game_state.grid, self.target_size)
        
        # Extract material state channels (solid, liquid, gas)
        if base_grid.shape[2] >= 3:
            spatial_tensor[:, :, 0] = base_grid[:, :, 0]  # Solid
            spatial_tensor[:, :, 1] = base_grid[:, :, 1]  # Liquid
            spatial_tensor[:, :, 2] = base_grid[:, :, 2]  # Gas
        else:
            # Default to gas if material states not available
            spatial_tensor[:, :, 2] = 1.0
        
        # Extract and normalize element ID channel
        if base_grid.shape[2] >= 4:
            element_channel = base_grid[:, :, 3]
            # Normalize element IDs to [0, 1] range
            spatial_tensor[:, :, 3] = np.clip(element_channel / self.max_element_id, 0.0, 1.0)
        
        # Extract and normalize temperature channel
        if base_grid.shape[2] >= 5:
            temp_channel = base_grid[:, :, 4]
            # Normalize temperature to [0, 1] range
            temp_min, temp_max = self.temperature_range
            normalized_temp = (temp_channel - temp_min) / (temp_max - temp_min)
            spatial_tensor[:, :, 4] = np.clip(normalized_temp, 0.0, 1.0)
        else:
            # Default to room temperature (normalized)
            room_temp = 20.0
            temp_min, temp_max = self.temperature_range
            normalized_room_temp = (room_temp - temp_min) / (temp_max - temp_min)
            spatial_tensor[:, :, 4] = normalized_room_temp
        
        # Add building information
        self._add_building_channel(spatial_tensor, game_state.buildings, 
                                 original_width, original_height)
        
        # Add duplicant information
        self._add_duplicant_channel(spatial_tensor, game_state.duplicants,
                                  original_width, original_height)
        
        return spatial_tensor
    
    def _resize_grid(self, grid: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize grid to target dimensions using appropriate sampling.
        
        For downsampling, uses area averaging to preserve information.
        For upsampling, uses nearest neighbor to avoid artifacts.
        """
        original_height, original_width, channels = grid.shape
        target_height, target_width = target_size
        
        # Create output grid
        resized_grid = np.zeros((target_height, target_width, channels), dtype=np.float32)
        
        # Calculate scaling factors
        height_scale = original_height / target_height
        width_scale = original_width / target_width
        
        for y in range(target_height):
            for x in range(target_width):
                # Map target coordinates to original coordinates
                orig_y_start = int(y * height_scale)
                orig_y_end = int((y + 1) * height_scale)
                orig_x_start = int(x * width_scale)
                orig_x_end = int((x + 1) * width_scale)
                
                # Ensure bounds
                orig_y_start = max(0, min(orig_y_start, original_height - 1))
                orig_y_end = max(orig_y_start + 1, min(orig_y_end, original_height))
                orig_x_start = max(0, min(orig_x_start, original_width - 1))
                orig_x_end = max(orig_x_start + 1, min(orig_x_end, original_width))
                
                # Sample the region and take mean (area averaging)
                region = grid[orig_y_start:orig_y_end, orig_x_start:orig_x_end, :]
                if region.size > 0:
                    resized_grid[y, x, :] = np.mean(region, axis=(0, 1))
        
        return resized_grid
    
    def _add_building_channel(self, spatial_tensor: np.ndarray, buildings: List[Building],
                            original_width: int, original_height: int) -> None:
        """Add building information to the spatial tensor."""
        target_height, target_width = self.target_size
        
        for building in buildings:
            # Convert world coordinates to grid coordinates
            x, y, z = building.position
            
            # Scale coordinates to target size
            grid_x = int((x / original_width) * target_width)
            grid_y = int((y / original_height) * target_height)
            
            # Ensure coordinates are within bounds
            if 0 <= grid_x < target_width and 0 <= grid_y < target_height:
                # Encode building type as normalized value
                building_type_id = self.building_type_map.get(building.building_type, 0)
                normalized_type = building_type_id / self.max_building_types
                
                # Set building presence (operational buildings get higher values)
                if building.operational:
                    spatial_tensor[grid_y, grid_x, 5] = normalized_type
                else:
                    spatial_tensor[grid_y, grid_x, 5] = normalized_type * 0.5  # Reduced for non-operational
    
    def _add_duplicant_channel(self, spatial_tensor: np.ndarray, duplicants: List[Duplicant],
                             original_width: int, original_height: int) -> None:
        """Add duplicant position information to the spatial tensor."""
        target_height, target_width = self.target_size
        
        for duplicant in duplicants:
            # Convert world coordinates to grid coordinates
            x, y, z = duplicant.position
            
            # Scale coordinates to target size
            grid_x = int((x / original_width) * target_width)
            grid_y = int((y / original_height) * target_height)
            
            # Ensure coordinates are within bounds
            if 0 <= grid_x < target_width and 0 <= grid_y < target_height:
                # Mark duplicant presence (value based on health/stress)
                health_factor = duplicant.health / 100.0  # Normalize health
                stress_factor = 1.0 - (duplicant.stress_level / 100.0)  # Invert stress
                duplicant_value = (health_factor + stress_factor) / 2.0
                
                # Accumulate if multiple duplicants in same cell
                spatial_tensor[grid_y, grid_x, 6] = min(1.0, 
                    spatial_tensor[grid_y, grid_x, 6] + duplicant_value)
    
    def _create_global_features(self, game_state: GameState) -> np.ndarray:
        """
        Create global features vector (64 dimensions).
        
        Features include:
        - Resource counts (normalized)
        - Duplicant statistics
        - Cycle information
        - World statistics
        """
        features = np.zeros(64, dtype=np.float32)
        
        # Resource features (indices 0-15)
        resource_names = [
            'oxygen', 'water', 'food', 'power', 'polluted_water', 'carbon_dioxide',
            'algae', 'dirt', 'sandstone', 'copper_ore'
        ]
        
        for i, resource_name in enumerate(resource_names):
            if i < 10:  # Ensure we don't exceed bounds
                resource_value = game_state.resources.get(resource_name, 0.0)
                # Log-normalize large resource values
                normalized_value = np.log1p(max(0.0, resource_value)) / 10.0  # Scale down
                features[i] = min(1.0, normalized_value)
        
        # Duplicant statistics (indices 10-25)
        if game_state.duplicants:
            # Count and health statistics
            features[10] = min(1.0, len(game_state.duplicants) / 20.0)  # Normalize count
            
            health_values = [d.health for d in game_state.duplicants]
            stress_values = [d.stress_level for d in game_state.duplicants]
            
            features[11] = np.mean(health_values) / 100.0  # Average health
            features[12] = np.mean(stress_values) / 100.0  # Average stress
            features[13] = np.min(health_values) / 100.0   # Min health
            features[14] = np.max(stress_values) / 100.0   # Max stress
            
            # Skill statistics
            all_skills = {}
            for duplicant in game_state.duplicants:
                for skill, level in duplicant.skills.items():
                    all_skills[skill] = all_skills.get(skill, 0) + level
            
            # Top 5 skill totals (normalized)
            skill_totals = sorted(all_skills.values(), reverse=True)[:5]
            for i, skill_total in enumerate(skill_totals):
                if i < 5:
                    features[15 + i] = min(1.0, skill_total / 50.0)  # Normalize skill totals
        
        # Building statistics (indices 20-35)
        if game_state.buildings:
            features[20] = min(1.0, len(game_state.buildings) / 100.0)  # Building count
            
            # Count by building type
            type_counts = {}
            operational_counts = {}
            
            for building in game_state.buildings:
                building_type = building.building_type
                type_counts[building_type] = type_counts.get(building_type, 0) + 1
                if building.operational:
                    operational_counts[building_type] = operational_counts.get(building_type, 0) + 1
            
            # Encode top building types
            building_types = ['power', 'plumbing', 'ventilation', 'agriculture', 'living']
            for i, building_type in enumerate(building_types):
                if i < 5:
                    count = type_counts.get(building_type, 0)
                    features[21 + i] = min(1.0, count / 20.0)  # Normalize counts
                    
                    operational_count = operational_counts.get(building_type, 0)
                    operational_ratio = operational_count / max(1, count)
                    features[26 + i] = operational_ratio
        
        # Cycle and time features (indices 31-40)
        features[31] = min(1.0, game_state.cycle / 1000.0)  # Normalize cycle count
        features[32] = (game_state.cycle % 100) / 100.0     # Cycle within 100-cycle periods
        features[33] = game_state.timestamp / 100000.0      # Normalized timestamp
        
        # World size features (indices 34-35)
        world_height, world_width = game_state.world_size
        features[34] = world_height / 500.0  # Normalize world dimensions
        features[35] = world_width / 500.0
        
        # Metadata features (indices 36-45)
        metadata = game_state.metadata
        if isinstance(metadata, dict):
            features[36] = 1.0 if metadata.get('mock', False) else 0.0
            features[37] = 1.0 if metadata.get('real_parse', False) else 0.0
            features[38] = min(1.0, metadata.get('object_count', 0) / 1000.0)
            features[39] = min(1.0, metadata.get('extracted_duplicants', 0) / 20.0)
            features[40] = min(1.0, metadata.get('extracted_buildings', 0) / 100.0)
        
        # Reserve indices 41-63 for future features
        
        return features
    
    def _create_metadata(self, game_state: GameState) -> Dict[str, Any]:
        """Create metadata about the preprocessing operation."""
        return {
            'original_world_size': game_state.world_size,
            'target_size': self.target_size,
            'num_channels': self.num_channels,
            'channel_config': self.channel_config.copy(),
            'temperature_range': self.temperature_range,
            'max_element_id': self.max_element_id,
            'building_type_map': self.building_type_map.copy(),
            'num_duplicants': len(game_state.duplicants),
            'num_buildings': len(game_state.buildings),
            'cycle': game_state.cycle,
            'preprocessing_version': '1.0'
        }


# Main interface function
def preprocess_state(game_state: GameState, 
                    target_size: Tuple[int, int] = (64, 64),
                    temperature_range: Tuple[float, float] = (-50.0, 200.0)) -> StateTensor:
    """
    Convert GameState to ML-ready StateTensor.
    
    This is the main interface function that implements the required
    `preprocess_state(game_state: GameState) -> StateTensor` interface.
    
    Args:
        game_state: Parsed game state from ONI save file
        target_size: Target spatial dimensions (height, width)
        temperature_range: Min and max temperatures for normalization
        
    Returns:
        StateTensor with spatial and global features
        
    Example:
        >>> game_state = parse_save("Colony001.sav")
        >>> state_tensor = preprocess_state(game_state)
        >>> print(f"Spatial shape: {state_tensor.spatial.shape}")
        >>> print(f"Global features: {state_tensor.global_features.shape}")
    """
    preprocessor = StatePreprocessor(
        target_size=target_size,
        temperature_range=temperature_range
    )
    return preprocessor.preprocess_state(game_state)