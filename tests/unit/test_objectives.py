"""
Unit tests for Mini-ONI Objective System.

Tests the objective evaluation, scoring, and progress tracking functionality.
"""

import pytest
import numpy as np
from src.environments.mini_oni.objectives import (
    ObjectiveSystem, ObjectiveType, ObjectiveStatus, ObjectiveProgress, ObjectiveRewards
)
from src.environments.mini_oni.environment import MiniONIEnvironment
from src.environments.mini_oni.game_state import GameState, Duplicant, Resources
from src.environments.mini_oni.building_types import BuildingType


class TestObjectiveProgress:
    """Test cases for ObjectiveProgress tracking."""
    
    def test_progress_initialization(self):
        """Test objective progress initializes correctly."""
        progress = ObjectiveProgress(
            objective_type=ObjectiveType.PRIMARY_OXYGEN,
            target_value=0.3,
            cycles_required=10
        )
        
        assert progress.objective_type == ObjectiveType.PRIMARY_OXYGEN
        assert progress.status == ObjectiveStatus.NOT_STARTED
        assert progress.current_value == 0.0
        assert progress.target_value == 0.3
        assert progress.cycles_required == 10
        assert progress.cycles_maintained == 0
    
    def test_progress_update_not_started(self):
        """Test progress update when objective not started."""
        progress = ObjectiveProgress(
            objective_type=ObjectiveType.PRIMARY_OXYGEN,
            target_value=0.3
        )
        
        progress.update(0.0, 1)
        
        assert progress.status == ObjectiveStatus.NOT_STARTED
        assert progress.current_value == 0.0
        assert progress.cycles_maintained == 0
    
    def test_progress_update_in_progress(self):
        """Test progress update when objective in progress."""
        progress = ObjectiveProgress(
            objective_type=ObjectiveType.PRIMARY_OXYGEN,
            target_value=0.3
        )
        
        progress.update(0.15, 1)  # 50% of target
        
        assert progress.status == ObjectiveStatus.IN_PROGRESS
        assert progress.current_value == 0.15
        assert progress.completion_percentage == 50.0
        assert progress.cycles_maintained == 0
    
    def test_progress_update_completed(self):
        """Test progress update when objective completed."""
        progress = ObjectiveProgress(
            objective_type=ObjectiveType.PRIMARY_OXYGEN,
            target_value=0.3,
            cycles_required=2
        )
        
        # Meet target for required cycles
        progress.update(0.35, 1)
        assert progress.status == ObjectiveStatus.IN_PROGRESS
        assert progress.cycles_maintained == 1
        
        progress.update(0.4, 2)
        assert progress.status == ObjectiveStatus.COMPLETED
        assert progress.cycles_maintained == 2
        assert progress.completion_percentage >= 100.0
    
    def test_progress_reset_on_failure(self):
        """Test progress resets when objective fails after partial success."""
        progress = ObjectiveProgress(
            objective_type=ObjectiveType.PRIMARY_OXYGEN,
            target_value=0.3,
            cycles_required=3
        )
        
        # Build up some progress
        progress.update(0.35, 1)
        assert progress.cycles_maintained == 1
        
        # Fail to meet target
        progress.update(0.2, 2)
        assert progress.cycles_maintained == 0
        assert progress.status == ObjectiveStatus.IN_PROGRESS


class TestObjectiveSystem:
    """Test cases for ObjectiveSystem."""
    
    def test_objective_system_initialization(self):
        """Test objective system initializes correctly."""
        system = ObjectiveSystem()
        
        assert len(system.objectives) == 3
        assert ObjectiveType.PRIMARY_OXYGEN in system.objectives
        assert ObjectiveType.SECONDARY_WATER in system.objectives
        assert ObjectiveType.TERTIARY_HAPPINESS in system.objectives
        
        # Check initial status
        for progress in system.objectives.values():
            assert progress.status == ObjectiveStatus.NOT_STARTED
    
    def test_custom_rewards_configuration(self):
        """Test objective system with custom rewards."""
        custom_rewards = ObjectiveRewards(
            oxygen_tile_reward=0.2,
            happiness_reward=0.1
        )
        
        system = ObjectiveSystem(custom_rewards)
        
        assert system.rewards.oxygen_tile_reward == 0.2
        assert system.rewards.happiness_reward == 0.1
    
    def test_oxygen_objective_evaluation(self):
        """Test primary oxygen objective evaluation."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        system = ObjectiveSystem()
        metrics = system.evaluate_objectives(env.game_state)
        
        # Check oxygen metrics are present
        assert 'oxygen_ratio' in metrics
        assert 'breathable_tiles' in metrics
        assert 'total_oxygen_reward' in metrics
        assert 'oxygen_objective_status' in metrics
        
        # Values should be reasonable
        assert 0.0 <= metrics['oxygen_ratio'] <= 1.0
        assert metrics['breathable_tiles'] >= 0
    
    def test_water_objective_evaluation(self):
        """Test secondary water objective evaluation."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        system = ObjectiveSystem()
        metrics = system.evaluate_objectives(env.game_state)
        
        # Check water metrics are present
        assert 'water_buildings_count' in metrics
        assert 'has_water_sieve' in metrics
        assert 'has_liquid_pump' in metrics
        assert 'water_system_functional' in metrics
        assert 'total_water_reward' in metrics
        
        # Initially should have no water buildings
        assert metrics['water_buildings_count'] == 0
        assert not metrics['has_water_sieve']
        assert not metrics['has_liquid_pump']
        assert not metrics['water_system_functional']
    
    def test_happiness_objective_evaluation(self):
        """Test tertiary happiness objective evaluation."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        system = ObjectiveSystem()
        metrics = system.evaluate_objectives(env.game_state)
        
        # Check happiness metrics are present
        assert 'happiness_ratio' in metrics
        assert 'happy_duplicants' in metrics
        assert 'stressed_duplicants' in metrics
        assert 'total_happiness_reward' in metrics
        
        # Values should be reasonable
        assert 0.0 <= metrics['happiness_ratio'] <= 1.0
        assert metrics['happy_duplicants'] >= 0
        assert metrics['stressed_duplicants'] >= 0
    
    def test_overall_score_calculation(self):
        """Test overall objective score calculation."""
        system = ObjectiveSystem()
        
        # Set some progress values
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].completion_percentage = 80.0
        system.objectives[ObjectiveType.SECONDARY_WATER].completion_percentage = 60.0
        system.objectives[ObjectiveType.TERTIARY_HAPPINESS].completion_percentage = 40.0
        
        score = system._calculate_overall_score()
        
        # Should be weighted average: 0.8*0.5 + 0.6*0.3 + 0.4*0.2 = 0.66
        expected_score = 0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2
        assert abs(score - expected_score) < 0.01
    
    def test_objective_status_checks(self):
        """Test objective status checking methods."""
        system = ObjectiveSystem()
        
        # Initially no objectives met
        assert not system.is_primary_objective_met()
        assert not system.is_secondary_objective_met()
        assert not system.is_tertiary_objective_met()
        assert not system.are_all_objectives_met()
        
        # Meet primary objective
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].current_value = 0.5
        assert system.is_primary_objective_met()
        assert not system.are_all_objectives_met()
        
        # Meet all objectives
        system.objectives[ObjectiveType.SECONDARY_WATER].current_value = 15.0
        system.objectives[ObjectiveType.TERTIARY_HAPPINESS].current_value = 0.8
        
        assert system.is_primary_objective_met()
        assert system.is_secondary_objective_met()
        assert system.is_tertiary_objective_met()
        assert system.are_all_objectives_met()
    
    def test_objective_reset(self):
        """Test objective system reset functionality."""
        system = ObjectiveSystem()
        
        # Set some progress
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].current_value = 0.5
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].status = ObjectiveStatus.IN_PROGRESS
        system.episode_stats['total_oxygen_reward'] = 100.0
        
        # Reset
        system.reset()
        
        # Should be back to initial state
        for progress in system.objectives.values():
            assert progress.status == ObjectiveStatus.NOT_STARTED
            assert progress.current_value == 0.0
            assert progress.cycles_maintained == 0
        
        assert system.episode_stats['total_oxygen_reward'] == 0.0
    
    def test_objective_summary(self):
        """Test objective summary generation."""
        system = ObjectiveSystem()
        
        # Set some progress
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].current_value = 0.25
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].status = ObjectiveStatus.IN_PROGRESS
        
        summary = system.get_objective_summary()
        
        # Check summary structure
        assert 'objectives' in summary
        assert 'episode_stats' in summary
        assert 'overall_score' in summary
        
        # Check objective details
        oxygen_obj = summary['objectives']['primary_oxygen']
        assert oxygen_obj['status'] == 'in_progress'
        assert oxygen_obj['current_value'] == 0.25
        assert oxygen_obj['target_value'] == 0.3
    
    def test_objective_progress_text(self):
        """Test objective progress text generation."""
        system = ObjectiveSystem()
        
        # Set some progress
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].status = ObjectiveStatus.IN_PROGRESS
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].completion_percentage = 75.0
        system.objectives[ObjectiveType.PRIMARY_OXYGEN].cycles_maintained = 3
        
        text = system.get_objective_progress_text()
        
        # Should contain progress information
        assert "Objective Progress" in text
        assert "Primary Oxygen" in text
        assert "75.0%" in text
        assert "3/10" in text  # cycles maintained/required


class TestObjectiveIntegration:
    """Test cases for objective system integration with environment."""
    
    def test_environment_objective_integration(self):
        """Test objective system integration with Mini-ONI environment."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        obs = env.reset()
        
        # Take a step
        obs, reward, done, info = env.step(0)  # No-op action
        
        # Check objective information in info
        assert 'objectives' in info
        assert 'objective_summary' in info
        
        objectives = info['objectives']
        assert 'oxygen_ratio' in objectives
        assert 'total_oxygen_reward' in objectives
        assert 'total_water_reward' in objectives
        assert 'total_happiness_reward' in objectives
    
    def test_environment_objective_methods(self):
        """Test environment objective status methods."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        # Test objective status methods
        assert isinstance(env.is_primary_objective_met(), bool)
        assert isinstance(env.is_secondary_objective_met(), bool)
        assert isinstance(env.is_tertiary_objective_met(), bool)
        assert isinstance(env.are_all_objectives_met(), bool)
        
        # Get objective status
        status = env.get_objective_status()
        assert 'objectives' in status
        assert 'overall_score' in status
    
    def test_objective_reward_calculation(self):
        """Test objective-based reward calculation."""
        env = MiniONIEnvironment(map_width=32, map_height=32)
        env.reset()
        
        # Take a step and check reward includes objective components
        obs, reward, done, info = env.step(0)
        
        # Reward should be influenced by objectives
        assert isinstance(reward, (int, float))
        
        # Check objective rewards are calculated
        objectives = info['objectives']
        total_objective_reward = (
            objectives['total_oxygen_reward'] +
            objectives['total_water_reward'] +
            objectives['total_happiness_reward']
        )
        
        # Objective reward should be part of total reward
        assert total_objective_reward != 0  # Should have some objective-based reward
    
    def test_render_with_objectives(self):
        """Test rendering includes objective information."""
        env = MiniONIEnvironment(map_width=16, map_height=16)
        env.reset()
        
        # Should not raise exception and include objective progress
        env.render(mode='human')


if __name__ == "__main__":
    pytest.main([__file__])