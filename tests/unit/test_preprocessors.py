"""
Unit tests for data preprocessors.

Tests the state preprocessing functionality including tensor creation,
normalization, and data validation.
"""

import numpy as np
import pytest
from unittest.mock import Mock

from src.data.parsers.oni_save_parser import GameState, Duplicant, Building
from src.data.preprocessors.state_preprocessor import (
    StatePreprocessor, StateTensor, preprocess_state
)


class TestStatePreprocessor:
    """Test cases for StatePreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = StatePreprocessor(
            target_size=(32, 32),
            temperature_range=(-50.0, 200.0),
            max_element_id=255,
            max_building_types=20
        )
    
    def create_mock_game_state(self) -> GameState:
        """Create a mock GameState for testing."""
        # Create mock grid (height=64, width=64, channels=7)
        grid = np.zeros((64, 64, 7), dtype=np.float32)
        
        # Set up realistic grid data
        grid[:, :, 2] = 1.0  # Gas (oxygen)
        grid[:, :, 4] = 20.0  # Temperature
        
        # Add some solid ground at bottom
        grid[50:64, :, 0] = 1.0  # Solid
        grid[50:64, :, 2] = 0.0  # No gas in solid areas
        grid[50:64, :, 4] = 15.0  # Cooler underground
        
        # Create mock duplicants
        duplicants = [
            Duplicant(
                name="Test Duplicant 1",
                position=(32.0, 30.0, 0.0),
                stress_level=25.0,
                health=95.0,
                skills={"Mining": 2, "Construction": 1},
                traits=["Quick Learner"]
            ),
            Duplicant(
                name="Test Duplicant 2", 
                position=(35.0, 30.0, 0.0),
                stress_level=15.0,
                health=100.0,
                skills={"Research": 3, "Cooking": 2},
                traits=["Fast Worker", "Gourmet"]
            )
        ]
        
        # Create mock buildings
        buildings = [
            Building(
                name="Manual Generator",
                position=(30.0, 25.0, 0.0),
                building_type="power",
                operational=True,
                temperature=28.0
            ),
            Building(
                name="Oxygen Diffuser",
                position=(35.0, 25.0, 0.0),
                building_type="ventilation",
                operational=True,
                temperature=22.0
            ),
            Building(
                name="Research Station",
                position=(40.0, 25.0, 0.0),
                building_type="research",
                operational=False,
                temperature=20.0
            )
        ]
        
        # Create mock resources
        resources = {
            'oxygen': 1000.0,
            'water': 500.0,
            'food': 200.0,
            'power': 800.0,
            'polluted_water': 50.0,
            'carbon_dioxide': 100.0,
            'algae': 300.0,
            'dirt': 2000.0,
            'sandstone': 5000.0,
            'copper_ore': 800.0
        }
        
        return GameState(
            grid=grid,
            duplicants=duplicants,
            buildings=buildings,
            resources=resources,
            cycle=50,
            timestamp=30000.0,
            world_size=(64, 64),
            metadata={'mock': True, 'real_parse': False}
        )
    
    def test_preprocess_state_basic(self):
        """Test basic state preprocessing functionality."""
        game_state = self.create_mock_game_state()
        
        result = self.preprocessor.preprocess_state(game_state)
        
        # Check result type and structure
        assert isinstance(result, StateTensor)
        assert isinstance(result.spatial, np.ndarray)
        assert isinstance(result.global_features, np.ndarray)
        assert isinstance(result.metadata, dict)
        
        # Check tensor dimensions
        assert result.spatial.shape == (32, 32, 7)  # Target size with 7 channels
        assert result.global_features.shape == (64,)  # 64 global features
        
        # Check data types
        assert result.spatial.dtype == np.float32
        assert result.global_features.dtype == np.float32
    
    def test_spatial_tensor_channels(self):
        """Test that spatial tensor channels are correctly populated."""
        game_state = self.create_mock_game_state()
        result = self.preprocessor.preprocess_state(game_state)
        
        spatial = result.spatial
        
        # Check material state channels (0, 1, 2)
        assert np.any(spatial[:, :, 0] > 0)  # Some solid areas
        assert np.any(spatial[:, :, 2] > 0)  # Some gas areas
        
        # Check temperature channel (4) is normalized
        temp_channel = spatial[:, :, 4]
        assert np.all(temp_channel >= 0.0)
        assert np.all(temp_channel <= 1.0)
        
        # Check building channel (5) has some buildings
        building_channel = spatial[:, :, 5]
        assert np.any(building_channel > 0)  # Some buildings present
        
        # Check duplicant channel (6) has duplicants
        duplicant_channel = spatial[:, :, 6]
        assert np.any(duplicant_channel > 0)  # Some duplicants present
    
    def test_global_features_content(self):
        """Test that global features contain expected information."""
        game_state = self.create_mock_game_state()
        result = self.preprocessor.preprocess_state(game_state)
        
        global_features = result.global_features
        
        # Check resource features (indices 0-9)
        assert global_features[0] > 0  # Oxygen should be present
        assert global_features[1] > 0  # Water should be present
        assert global_features[2] > 0  # Food should be present
        
        # Check duplicant statistics (indices 10-19)
        assert global_features[10] > 0  # Duplicant count
        assert 0 <= global_features[11] <= 1  # Average health (normalized)
        assert 0 <= global_features[12] <= 1  # Average stress (normalized)
        
        # Check building statistics (indices 20-30)
        assert global_features[20] > 0  # Building count
        
        # Check cycle information (indices 31-33)
        assert global_features[31] > 0  # Cycle count
        assert 0 <= global_features[32] <= 1  # Cycle within period
    
    def test_temperature_normalization(self):
        """Test temperature normalization is working correctly."""
        game_state = self.create_mock_game_state()
        
        # Set specific temperatures in the grid
        game_state.grid[:10, :10, 4] = -30.0  # Cold area
        game_state.grid[10:20, :10, 4] = 20.0  # Room temperature
        game_state.grid[20:30, :10, 4] = 100.0  # Hot area
        
        result = self.preprocessor.preprocess_state(game_state)
        temp_channel = result.spatial[:, :, 4]
        
        # Check normalization bounds
        assert np.all(temp_channel >= 0.0)
        assert np.all(temp_channel <= 1.0)
        
        # Check that different temperatures map to different normalized values
        unique_temps = np.unique(temp_channel)
        assert len(unique_temps) > 1  # Should have variation
    
    def test_building_encoding(self):
        """Test building type encoding in spatial tensor."""
        game_state = self.create_mock_game_state()
        result = self.preprocessor.preprocess_state(game_state)
        
        building_channel = result.spatial[:, :, 5]
        
        # Should have non-zero values where buildings are placed
        assert np.any(building_channel > 0)
        
        # Operational buildings should have higher values than non-operational
        # (This is hard to test precisely due to coordinate mapping, but we can
        # check that we have a range of values)
        unique_values = np.unique(building_channel[building_channel > 0])
        assert len(unique_values) >= 1  # At least some building values
    
    def test_duplicant_encoding(self):
        """Test duplicant position encoding in spatial tensor."""
        game_state = self.create_mock_game_state()
        result = self.preprocessor.preprocess_state(game_state)
        
        duplicant_channel = result.spatial[:, :, 6]
        
        # Should have non-zero values where duplicants are placed
        assert np.any(duplicant_channel > 0)
        
        # Values should be based on health/stress
        duplicant_values = duplicant_channel[duplicant_channel > 0]
        assert np.all(duplicant_values > 0)
        assert np.all(duplicant_values <= 1.0)
    
    def test_grid_resizing(self):
        """Test grid resizing functionality."""
        # Create game state with different world size
        game_state = self.create_mock_game_state()
        game_state.world_size = (128, 128)  # Different from target size
        
        # Create larger grid
        large_grid = np.zeros((128, 128, 7), dtype=np.float32)
        large_grid[:, :, 2] = 1.0  # Fill with gas
        large_grid[:, :, 4] = 25.0  # Set temperature
        game_state.grid = large_grid
        
        result = self.preprocessor.preprocess_state(game_state)
        
        # Should still produce target size output
        assert result.spatial.shape == (32, 32, 7)
    
    def test_validation_errors(self):
        """Test input validation and error handling."""
        # Test None input
        with pytest.raises(ValueError, match="GameState cannot be None"):
            self.preprocessor.preprocess_state(None)
        
        # Test invalid grid
        invalid_state = self.create_mock_game_state()
        invalid_state.grid = None
        
        with pytest.raises(ValueError, match="GameState must have a grid"):
            self.preprocessor.preprocess_state(invalid_state)
        
        # Test invalid grid shape
        invalid_state = self.create_mock_game_state()
        invalid_state.grid = np.zeros((64, 64))  # 2D instead of 3D
        
        with pytest.raises(ValueError, match="Grid must be 3D"):
            self.preprocessor.preprocess_state(invalid_state)
    
    def test_metadata_creation(self):
        """Test metadata creation and content."""
        game_state = self.create_mock_game_state()
        result = self.preprocessor.preprocess_state(game_state)
        
        metadata = result.metadata
        
        # Check required metadata fields
        assert 'original_world_size' in metadata
        assert 'target_size' in metadata
        assert 'num_channels' in metadata
        assert 'channel_config' in metadata
        assert 'num_duplicants' in metadata
        assert 'num_buildings' in metadata
        assert 'cycle' in metadata
        assert 'preprocessing_version' in metadata
        
        # Check values
        assert metadata['original_world_size'] == (64, 64)
        assert metadata['target_size'] == (32, 32)
        assert metadata['num_channels'] == 7
        assert metadata['num_duplicants'] == 2
        assert metadata['num_buildings'] == 3
        assert metadata['cycle'] == 50


class TestPreprocessStateFunction:
    """Test the main preprocess_state function interface."""
    
    def create_simple_game_state(self) -> GameState:
        """Create a simple GameState for testing."""
        grid = np.zeros((32, 32, 7), dtype=np.float32)
        grid[:, :, 2] = 1.0  # Gas
        grid[:, :, 4] = 20.0  # Temperature
        
        return GameState(
            grid=grid,
            duplicants=[],
            buildings=[],
            resources={'oxygen': 100.0},
            cycle=1,
            timestamp=600.0,
            world_size=(32, 32),
            metadata={}
        )
    
    def test_preprocess_state_function(self):
        """Test the main preprocess_state function."""
        game_state = self.create_simple_game_state()
        
        result = preprocess_state(game_state)
        
        assert isinstance(result, StateTensor)
        assert result.spatial.shape == (64, 64, 7)  # Default target size
        assert result.global_features.shape == (64,)
    
    def test_preprocess_state_custom_params(self):
        """Test preprocess_state with custom parameters."""
        game_state = self.create_simple_game_state()
        
        result = preprocess_state(
            game_state,
            target_size=(16, 16),
            temperature_range=(0.0, 100.0)
        )
        
        assert result.spatial.shape == (16, 16, 7)  # Custom target size
        assert result.global_features.shape == (64,)
    
    def test_empty_duplicants_and_buildings(self):
        """Test preprocessing with no duplicants or buildings."""
        game_state = self.create_simple_game_state()
        
        # Should work without warnings in test environment
        result = preprocess_state(game_state)
        
        # Check that channels are still properly initialized
        assert result.spatial.shape == (64, 64, 7)
        assert np.all(result.spatial[:, :, 5] == 0)  # No buildings
        assert np.all(result.spatial[:, :, 6] == 0)  # No duplicants


if __name__ == '__main__':
    pytest.main([__file__])