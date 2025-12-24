"""
Unit tests for Mini-ONI Environment.

Tests the core functionality of the simplified ONI environment.
"""

import pytest
import numpy as np
from src.environments.mini_oni.environment import MiniONIEnvironment
from src.environments.mini_oni.actions import ActionType, PlaceBuildingAction, DigAction, Region
from src.environments.mini_oni.building_types import BuildingType


class TestMiniONIEnvironment:
    """Test cases for Mini-ONI Environment."""
    
    def test_environment_initialization(self):
        """Test environment initializes correctly."""
        env = MiniONIEnvironment(
            map_width=32,
            map_height=32,
            max_cycles=50,
            num_duplicants=2
        )
        
        assert env.map_width == 32
        assert env.map_height == 32
        assert env.max_cycles == 50
        assert env.num_duplicants == 2
        assert len(env.action_space) > 0
        assert env.num_actions == len(env.action_space)
    
    def test_environment_constraints(self):
        """Test environment enforces size and cycle constraints."""
        # Test maximum constraints
        env = MiniONIEnvironment(
            map_width=100,  # Should be clamped to 64
            map_height=100,  # Should be clamped to 64
            max_cycles=200  # Should be clamped to 100
        )
        
        assert env.map_width == 64
        assert env.map_height == 64
        assert env.max_cycles == 100
    
    def test_reset_functionality(self):
        """Test environment reset works correctly."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        
        # Reset environment
        obs = env.reset()
        
        # Check observation shape
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        
        # Check game state is initialized
        assert env.game_state is not None
        assert env.game_state.cycle == 0
        assert len(env.game_state.duplicants) == env.num_duplicants
        assert len(env.game_state.buildings) > 0  # Should have starting buildings
        
        # Check all duplicants are alive and positioned
        for dup in env.game_state.duplicants:
            assert dup.is_alive
            assert 0 <= dup.x < env.map_width
            assert 0 <= dup.y < env.map_height
    
    def test_step_functionality(self):
        """Test environment step function."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        # Take a no-op action (should be action 0)
        obs, reward, done, info = env.step(0)
        
        # Check return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check info contains expected keys
        expected_keys = ['action_success', 'cycle', 'episode_step', 'breathable_tiles', 
                        'happy_duplicants', 'living_duplicants', 'success_score']
        for key in expected_keys:
            assert key in info
    
    def test_action_space_generation(self):
        """Test action space is generated correctly."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        
        # Check action space contains different action types
        action_types = set()
        for action in env.action_space:
            action_types.add(action.action_type)
        
        # Should have at least no-op and building placement
        assert ActionType.NO_OP in action_types
        assert ActionType.PLACE_BUILDING in action_types
        
        # Action space should be limited
        assert len(env.action_space) <= 200
    
    def test_building_placement(self):
        """Test building placement functionality."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        initial_buildings = len(env.game_state.buildings)
        
        # Find a place building action
        place_action = None
        for action in env.action_space:
            if isinstance(action, PlaceBuildingAction):
                if action.is_valid(env.game_state):
                    place_action = action
                    break
        
        if place_action:
            action_idx = env.action_space.index(place_action)
            obs, reward, done, info = env.step(action_idx)
            
            # Should have placed a building if action was successful
            if info['action_success']:
                assert len(env.game_state.buildings) > initial_buildings
    
    def test_digging_functionality(self):
        """Test digging functionality."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        # Find a dig action
        dig_action = None
        for action in env.action_space:
            if isinstance(action, DigAction):
                if action.is_valid(env.game_state):
                    dig_action = action
                    break
        
        if dig_action:
            action_idx = env.action_space.index(dig_action)
            
            # Count solid tiles before digging
            solid_tiles_before = sum(
                1 for y in range(env.map_height) for x in range(env.map_width)
                if env.game_state.get_tile(x, y).material_state == "solid"
            )
            
            obs, reward, done, info = env.step(action_idx)
            
            # Should have fewer solid tiles if digging was successful
            if info['action_success']:
                solid_tiles_after = sum(
                    1 for y in range(env.map_height) for x in range(env.map_width)
                    if env.game_state.get_tile(x, y).material_state == "solid"
                )
                assert solid_tiles_after < solid_tiles_before
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        env = MiniONIEnvironment(map_width=32, map_height=32, max_cycles=5)
        env.reset()
        
        # Run until episode should terminate
        done = False
        steps = 0
        max_steps = 100  # Safety limit
        
        while not done and steps < max_steps:
            obs, reward, done, info = env.step(0)  # No-op action
            steps += 1
        
        # Should terminate due to cycle limit or other conditions
        assert done
        assert steps < max_steps  # Should not hit safety limit
    
    def test_observation_shape(self):
        """Test observation has correct shape and values."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        obs = env.reset()
        
        # Observation should be flattened combination of spatial + global
        spatial_size = 32 * 32 * 8  # 32x32 grid with 8 channels
        global_size = 64  # Global features
        expected_size = spatial_size + global_size
        
        assert obs.shape == (expected_size,)
        assert obs.dtype == np.float32
        
        # Values should be normalized (mostly between 0 and 1)
        assert np.all(obs >= -1.0)  # Allow some negative values for temperature
        assert np.all(obs <= 1.0)
    
    def test_action_masking(self):
        """Test action masking functionality."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        mask = env.get_action_mask()
        
        # Mask should have same length as action space
        assert len(mask) == len(env.action_space)
        assert mask.dtype == bool
        
        # At least no-op should be valid
        assert mask[0]  # Assuming no-op is first action
    
    def test_render_functionality(self):
        """Test rendering functionality."""
        env = MiniONIEnvironment(map_width=16, map_height=16)
        env.reset()
        
        # Test text rendering (should not raise exception)
        env.render(mode='human')
        
        # Test RGB array rendering
        rgb_array = env.render(mode='rgb_array')
        assert isinstance(rgb_array, np.ndarray)
        assert rgb_array.shape == (16, 16, 3)
        assert rgb_array.dtype == np.uint8
    
    def test_success_score_calculation(self):
        """Test success score calculation."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        score = env.game_state.get_success_score()
        
        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0
        
        # Score should be positive for initial state (duplicants alive)
        assert score > 0.0
    
    def test_resource_tracking(self):
        """Test resource tracking functionality."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        resources = env.game_state.resources
        
        # Resources should be initialized
        assert hasattr(resources, 'oxygen')
        assert hasattr(resources, 'food')
        assert hasattr(resources, 'water')
        assert hasattr(resources, 'power')
        
        # Update resources
        env.game_state.update_resources()
        
        # Should have some oxygen from breathable tiles
        assert resources.oxygen >= 0
    
    def test_objective_system_integration(self):
        """Test objective system integration."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        # Should have objective system
        assert hasattr(env, 'objective_system')
        assert env.objective_system is not None
        
        # Should have objective status methods
        assert hasattr(env, 'is_primary_objective_met')
        assert hasattr(env, 'is_secondary_objective_met')
        assert hasattr(env, 'is_tertiary_objective_met')
        assert hasattr(env, 'are_all_objectives_met')
        
        # Methods should return boolean values
        assert isinstance(env.is_primary_objective_met(), bool)
        assert isinstance(env.is_secondary_objective_met(), bool)
        assert isinstance(env.is_tertiary_objective_met(), bool)
        assert isinstance(env.are_all_objectives_met(), bool)
    
    def test_objective_info_in_step(self):
        """Test objective information is included in step info."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        obs, reward, done, info = env.step(0)
        
        # Should have objective information
        assert 'objectives' in info
        assert 'objective_summary' in info
        
        objectives = info['objectives']
        
        # Should have all objective metrics
        expected_metrics = [
            'oxygen_ratio', 'total_oxygen_reward',
            'water_system_functional', 'total_water_reward',
            'happiness_ratio', 'total_happiness_reward',
            'overall_objective_score'
        ]
        
        for metric in expected_metrics:
            assert metric in objectives


if __name__ == "__main__":
    pytest.main([__file__])