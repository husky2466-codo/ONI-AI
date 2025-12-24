"""
Objective system for Mini-ONI environment.

Implements the three main objectives:
1. Primary: Oxygen maintenance (>500g/tile)
2. Secondary: Polluted water routing
3. Tertiary: Duplicant happiness (>50%)

Provides objective evaluation, scoring, and progress tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class ObjectiveType(Enum):
    """Types of objectives in the Mini-ONI environment."""
    
    PRIMARY_OXYGEN = "primary_oxygen"
    SECONDARY_WATER = "secondary_water"
    TERTIARY_HAPPINESS = "tertiary_happiness"


class ObjectiveStatus(Enum):
    """Status of objective completion."""
    
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ObjectiveProgress:
    """Progress tracking for a single objective."""
    
    objective_type: ObjectiveType
    status: ObjectiveStatus = ObjectiveStatus.NOT_STARTED
    current_value: float = 0.0
    target_value: float = 0.0
    max_achieved: float = 0.0
    completion_percentage: float = 0.0
    cycles_maintained: int = 0
    cycles_required: int = 10  # Cycles needed to maintain for completion
    last_updated_cycle: int = 0
    
    def update(self, current_value: float, cycle: int):
        """Update progress with current measurement."""
        self.current_value = current_value
        self.max_achieved = max(self.max_achieved, current_value)
        self.last_updated_cycle = cycle
        
        # Calculate completion percentage
        if self.target_value > 0:
            self.completion_percentage = min(100.0, (current_value / self.target_value) * 100.0)
        
        # Update status based on progress
        if current_value >= self.target_value:
            self.cycles_maintained += 1
            if self.cycles_maintained >= self.cycles_required:
                self.status = ObjectiveStatus.COMPLETED
            else:
                self.status = ObjectiveStatus.IN_PROGRESS
        else:
            self.cycles_maintained = 0
            if current_value > 0:
                self.status = ObjectiveStatus.IN_PROGRESS
            else:
                self.status = ObjectiveStatus.NOT_STARTED


@dataclass
class ObjectiveRewards:
    """Reward configuration for objectives."""
    
    # Primary objective rewards
    oxygen_tile_reward: float = 0.1  # Per breathable tile
    oxygen_threshold_bonus: float = 50.0  # Bonus for meeting threshold
    oxygen_maintenance_bonus: float = 100.0  # Bonus for sustained maintenance
    
    # Secondary objective rewards
    water_routing_reward: float = 0.05  # Per unit of clean water produced
    water_system_bonus: float = 30.0  # Bonus for functional water system
    
    # Tertiary objective rewards
    happiness_reward: float = 0.05  # Per happy duplicant
    happiness_threshold_bonus: float = 20.0  # Bonus for meeting threshold
    
    # Penalty weights
    oxygen_shortage_penalty: float = -0.2  # Per tile below threshold
    duplicant_stress_penalty: float = -0.1  # Per stressed duplicant
    system_failure_penalty: float = -100.0  # Major system failures


class ObjectiveSystem:
    """
    Manages and evaluates objectives for the Mini-ONI environment.
    
    Tracks progress on three main objectives:
    1. Primary: Maintain oxygen levels >500g/tile in living areas
    2. Secondary: Route polluted water through water sieve system
    3. Tertiary: Maintain duplicant happiness >50%
    """
    
    def __init__(self, rewards: Optional[ObjectiveRewards] = None):
        """
        Initialize objective system.
        
        Args:
            rewards: Custom reward configuration
        """
        self.rewards = rewards or ObjectiveRewards()
        
        # Initialize objective progress tracking
        self.objectives: Dict[ObjectiveType, ObjectiveProgress] = {
            ObjectiveType.PRIMARY_OXYGEN: ObjectiveProgress(
                objective_type=ObjectiveType.PRIMARY_OXYGEN,
                target_value=0.3,  # 30% of tiles should be breathable
                cycles_required=10
            ),
            ObjectiveType.SECONDARY_WATER: ObjectiveProgress(
                objective_type=ObjectiveType.SECONDARY_WATER,
                target_value=10.0,  # 10kg of clean water produced
                cycles_required=5
            ),
            ObjectiveType.TERTIARY_HAPPINESS: ObjectiveProgress(
                objective_type=ObjectiveType.TERTIARY_HAPPINESS,
                target_value=0.5,  # 50% of duplicants should be happy
                cycles_required=15
            )
        }
        
        # Episode statistics
        self.episode_stats = {
            'total_oxygen_reward': 0.0,
            'total_water_reward': 0.0,
            'total_happiness_reward': 0.0,
            'objectives_completed': 0,
            'peak_oxygen_ratio': 0.0,
            'peak_happiness_ratio': 0.0,
            'water_systems_built': 0,
            'cycles_all_objectives_met': 0
        }
    
    def evaluate_objectives(self, game_state) -> Dict[str, float]:
        """
        Evaluate all objectives and return detailed metrics.
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with objective metrics and scores
        """
        metrics = {}
        
        # Evaluate primary objective: oxygen maintenance
        oxygen_metrics = self._evaluate_oxygen_objective(game_state)
        metrics.update(oxygen_metrics)
        
        # Evaluate secondary objective: water routing
        water_metrics = self._evaluate_water_objective(game_state)
        metrics.update(water_metrics)
        
        # Evaluate tertiary objective: duplicant happiness
        happiness_metrics = self._evaluate_happiness_objective(game_state)
        metrics.update(happiness_metrics)
        
        # Calculate overall objective score
        metrics['overall_objective_score'] = self._calculate_overall_score()
        
        # Update episode statistics
        self._update_episode_stats(metrics)
        
        return metrics
    
    def _evaluate_oxygen_objective(self, game_state) -> Dict[str, float]:
        """Evaluate primary oxygen maintenance objective."""
        total_tiles = game_state.width * game_state.height
        breathable_tiles = game_state.get_breathable_tiles()
        oxygen_ratio = breathable_tiles / total_tiles if total_tiles > 0 else 0.0
        
        # Update progress
        self.objectives[ObjectiveType.PRIMARY_OXYGEN].update(oxygen_ratio, game_state.cycle)
        
        # Calculate rewards
        base_reward = breathable_tiles * self.rewards.oxygen_tile_reward
        
        # Threshold bonus
        threshold_bonus = 0.0
        if oxygen_ratio >= 0.3:  # 30% threshold
            threshold_bonus = self.rewards.oxygen_threshold_bonus
        
        # Maintenance bonus
        maintenance_bonus = 0.0
        progress = self.objectives[ObjectiveType.PRIMARY_OXYGEN]
        if progress.status == ObjectiveStatus.COMPLETED:
            maintenance_bonus = self.rewards.oxygen_maintenance_bonus
        
        # Shortage penalty
        shortage_penalty = 0.0
        if oxygen_ratio < 0.1:  # Critical shortage
            shortage_tiles = max(0, int(total_tiles * 0.1) - breathable_tiles)
            shortage_penalty = shortage_tiles * self.rewards.oxygen_shortage_penalty
        
        total_oxygen_reward = base_reward + threshold_bonus + maintenance_bonus + shortage_penalty
        
        return {
            'oxygen_ratio': oxygen_ratio,
            'breathable_tiles': breathable_tiles,
            'oxygen_base_reward': base_reward,
            'oxygen_threshold_bonus': threshold_bonus,
            'oxygen_maintenance_bonus': maintenance_bonus,
            'oxygen_shortage_penalty': shortage_penalty,
            'total_oxygen_reward': total_oxygen_reward,
            'oxygen_objective_status': progress.status.value,
            'oxygen_completion_percentage': progress.completion_percentage,
            'oxygen_cycles_maintained': progress.cycles_maintained
        }
    
    def _evaluate_water_objective(self, game_state) -> Dict[str, float]:
        """Evaluate secondary water routing objective."""
        # Count water-related buildings
        water_buildings = 0
        has_water_sieve = False
        has_liquid_pump = False
        
        from .building_types import BuildingType
        
        for building in game_state.buildings:
            if building.building_type == BuildingType.WATER_SIEVE:
                water_buildings += 1
                has_water_sieve = True
            elif building.building_type == BuildingType.LIQUID_PUMP:
                water_buildings += 1
                has_liquid_pump = True
        
        # Calculate water system functionality
        water_system_functional = has_water_sieve and has_liquid_pump
        clean_water_produced = game_state.resources.water
        
        # Update progress (based on clean water production)
        water_progress_value = clean_water_produced / 10.0  # Normalize to target
        self.objectives[ObjectiveType.SECONDARY_WATER].update(water_progress_value, game_state.cycle)
        
        # Calculate rewards
        base_reward = clean_water_produced * self.rewards.water_routing_reward
        
        # System bonus
        system_bonus = 0.0
        if water_system_functional:
            system_bonus = self.rewards.water_system_bonus
        
        total_water_reward = base_reward + system_bonus
        
        progress = self.objectives[ObjectiveType.SECONDARY_WATER]
        
        return {
            'water_buildings_count': water_buildings,
            'has_water_sieve': has_water_sieve,
            'has_liquid_pump': has_liquid_pump,
            'water_system_functional': water_system_functional,
            'clean_water_produced': clean_water_produced,
            'polluted_water_amount': game_state.resources.polluted_water,
            'water_base_reward': base_reward,
            'water_system_bonus': system_bonus,
            'total_water_reward': total_water_reward,
            'water_objective_status': progress.status.value,
            'water_completion_percentage': progress.completion_percentage,
            'water_cycles_maintained': progress.cycles_maintained
        }
    
    def _evaluate_happiness_objective(self, game_state) -> Dict[str, float]:
        """Evaluate tertiary duplicant happiness objective."""
        total_duplicants = len(game_state.duplicants)
        happy_duplicants = game_state.get_happy_duplicants()
        living_duplicants = game_state.get_living_duplicants()
        
        # Calculate happiness ratio (only count living duplicants)
        happiness_ratio = happy_duplicants / living_duplicants if living_duplicants > 0 else 0.0
        
        # Update progress
        self.objectives[ObjectiveType.TERTIARY_HAPPINESS].update(happiness_ratio, game_state.cycle)
        
        # Calculate rewards
        base_reward = happy_duplicants * self.rewards.happiness_reward
        
        # Threshold bonus
        threshold_bonus = 0.0
        if happiness_ratio >= 0.5:  # 50% threshold
            threshold_bonus = self.rewards.happiness_threshold_bonus
        
        # Stress penalty
        stressed_duplicants = sum(1 for dup in game_state.duplicants 
                                if dup.is_alive and dup.stress_level >= 80.0)
        stress_penalty = stressed_duplicants * self.rewards.duplicant_stress_penalty
        
        total_happiness_reward = base_reward + threshold_bonus + stress_penalty
        
        progress = self.objectives[ObjectiveType.TERTIARY_HAPPINESS]
        
        return {
            'total_duplicants': total_duplicants,
            'living_duplicants': living_duplicants,
            'happy_duplicants': happy_duplicants,
            'stressed_duplicants': stressed_duplicants,
            'happiness_ratio': happiness_ratio,
            'happiness_base_reward': base_reward,
            'happiness_threshold_bonus': threshold_bonus,
            'happiness_stress_penalty': stress_penalty,
            'total_happiness_reward': total_happiness_reward,
            'happiness_objective_status': progress.status.value,
            'happiness_completion_percentage': progress.completion_percentage,
            'happiness_cycles_maintained': progress.cycles_maintained
        }
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall objective completion score."""
        weights = {
            ObjectiveType.PRIMARY_OXYGEN: 0.5,    # 50% weight
            ObjectiveType.SECONDARY_WATER: 0.3,   # 30% weight
            ObjectiveType.TERTIARY_HAPPINESS: 0.2  # 20% weight
        }
        
        total_score = 0.0
        for obj_type, weight in weights.items():
            progress = self.objectives[obj_type]
            completion = progress.completion_percentage / 100.0
            total_score += completion * weight
        
        return total_score
    
    def _update_episode_stats(self, metrics: Dict[str, float]):
        """Update episode-level statistics."""
        self.episode_stats['total_oxygen_reward'] += metrics.get('total_oxygen_reward', 0.0)
        self.episode_stats['total_water_reward'] += metrics.get('total_water_reward', 0.0)
        self.episode_stats['total_happiness_reward'] += metrics.get('total_happiness_reward', 0.0)
        
        # Track peaks
        self.episode_stats['peak_oxygen_ratio'] = max(
            self.episode_stats['peak_oxygen_ratio'],
            metrics.get('oxygen_ratio', 0.0)
        )
        self.episode_stats['peak_happiness_ratio'] = max(
            self.episode_stats['peak_happiness_ratio'],
            metrics.get('happiness_ratio', 0.0)
        )
        
        # Count completed objectives
        completed_count = sum(
            1 for progress in self.objectives.values()
            if progress.status == ObjectiveStatus.COMPLETED
        )
        self.episode_stats['objectives_completed'] = completed_count
        
        # Track cycles where all objectives are met simultaneously
        all_met = all(
            progress.current_value >= progress.target_value
            for progress in self.objectives.values()
        )
        if all_met:
            self.episode_stats['cycles_all_objectives_met'] += 1
    
    def get_objective_summary(self) -> Dict[str, any]:
        """Get summary of all objectives and their current status."""
        summary = {
            'objectives': {},
            'episode_stats': self.episode_stats.copy(),
            'overall_score': self._calculate_overall_score()
        }
        
        for obj_type, progress in self.objectives.items():
            summary['objectives'][obj_type.value] = {
                'status': progress.status.value,
                'current_value': progress.current_value,
                'target_value': progress.target_value,
                'completion_percentage': progress.completion_percentage,
                'cycles_maintained': progress.cycles_maintained,
                'cycles_required': progress.cycles_required,
                'max_achieved': progress.max_achieved
            }
        
        return summary
    
    def reset(self):
        """Reset objective system for new episode."""
        # Reset all objective progress
        for progress in self.objectives.values():
            progress.status = ObjectiveStatus.NOT_STARTED
            progress.current_value = 0.0
            progress.max_achieved = 0.0
            progress.completion_percentage = 0.0
            progress.cycles_maintained = 0
            progress.last_updated_cycle = 0
        
        # Reset episode statistics
        self.episode_stats = {
            'total_oxygen_reward': 0.0,
            'total_water_reward': 0.0,
            'total_happiness_reward': 0.0,
            'objectives_completed': 0,
            'peak_oxygen_ratio': 0.0,
            'peak_happiness_ratio': 0.0,
            'water_systems_built': 0,
            'cycles_all_objectives_met': 0
        }
    
    def get_objective_rewards(self, game_state) -> float:
        """
        Calculate total reward from all objectives.
        
        Args:
            game_state: Current game state
            
        Returns:
            Total objective-based reward
        """
        metrics = self.evaluate_objectives(game_state)
        
        total_reward = (
            metrics.get('total_oxygen_reward', 0.0) +
            metrics.get('total_water_reward', 0.0) +
            metrics.get('total_happiness_reward', 0.0)
        )
        
        return total_reward
    
    def is_primary_objective_met(self) -> bool:
        """Check if primary objective is currently met."""
        progress = self.objectives[ObjectiveType.PRIMARY_OXYGEN]
        return progress.current_value >= progress.target_value
    
    def is_secondary_objective_met(self) -> bool:
        """Check if secondary objective is currently met."""
        progress = self.objectives[ObjectiveType.SECONDARY_WATER]
        return progress.current_value >= progress.target_value
    
    def is_tertiary_objective_met(self) -> bool:
        """Check if tertiary objective is currently met."""
        progress = self.objectives[ObjectiveType.TERTIARY_HAPPINESS]
        return progress.current_value >= progress.target_value
    
    def are_all_objectives_met(self) -> bool:
        """Check if all objectives are currently met."""
        return (self.is_primary_objective_met() and 
                self.is_secondary_objective_met() and 
                self.is_tertiary_objective_met())
    
    def get_objective_progress_text(self) -> str:
        """Get human-readable progress text for all objectives."""
        lines = []
        lines.append("=== Objective Progress ===")
        
        for obj_type, progress in self.objectives.items():
            name = obj_type.value.replace('_', ' ').title()
            status_icon = {
                ObjectiveStatus.NOT_STARTED: "⚪",
                ObjectiveStatus.IN_PROGRESS: "🟡", 
                ObjectiveStatus.COMPLETED: "🟢",
                ObjectiveStatus.FAILED: "🔴"
            }.get(progress.status, "❓")
            
            lines.append(f"{status_icon} {name}: {progress.completion_percentage:.1f}% "
                        f"({progress.cycles_maintained}/{progress.cycles_required} cycles)")
        
        lines.append(f"Overall Score: {self._calculate_overall_score():.2f}")
        
        return "\n".join(lines)