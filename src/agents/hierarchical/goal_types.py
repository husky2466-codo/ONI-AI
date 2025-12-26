"""
Goal and subgoal type definitions for hierarchical planning.

Task 5.1: Hierarchical Planning Architecture

Defines:
- AbstractGoal: High-level goals aligned with ObjectiveSystem
- SubgoalType: Mid-level subgoals from HierarchicalDecisionTree patterns
- Termination conditions for each goal/subgoal
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Callable, Optional, Any, List
import numpy as np


class AbstractGoal(Enum):
    """
    High-level abstract goals for the planner.

    Aligned with ObjectiveSystem objectives:
    - Primary: MAINTAIN_OXYGEN
    - Secondary: MANAGE_WATER
    - Tertiary: ENSURE_HAPPINESS

    Plus strategic goals from HierarchicalDecisionTree:
    - BUILD_INFRASTRUCTURE
    - SECURE_RESOURCES
    """
    MAINTAIN_OXYGEN = 0      # Primary objective
    MANAGE_WATER = 1         # Secondary objective
    ENSURE_HAPPINESS = 2     # Tertiary objective
    BUILD_INFRASTRUCTURE = 3  # Strategic expansion
    SECURE_RESOURCES = 4     # Power/food security

    @classmethod
    def count(cls) -> int:
        """Return number of goal types."""
        return len(cls)

    def get_priority(self) -> int:
        """Return goal priority (higher = more urgent)."""
        priorities = {
            AbstractGoal.MAINTAIN_OXYGEN: 100,
            AbstractGoal.MANAGE_WATER: 80,
            AbstractGoal.ENSURE_HAPPINESS: 60,
            AbstractGoal.BUILD_INFRASTRUCTURE: 40,
            AbstractGoal.SECURE_RESOURCES: 50,
        }
        return priorities[self]


class SubgoalType(Enum):
    """
    Mid-level subgoals for the controller.

    Derived from HierarchicalDecisionTree tactical templates.
    """
    # Oxygen management
    BUILD_OXYGEN_ROOM = 0
    IMPROVE_VENTILATION = 1

    # Infrastructure
    BUILD_BATHROOM = 2
    BUILD_BEDROOM = 3
    BUILD_KITCHEN = 4

    # Resources
    BUILD_POWER_ROOM = 5
    BUILD_FARM = 6
    BUILD_STORAGE = 7

    # Expansion
    EXPAND_LIVING_SPACE = 8
    BUILD_WORKSHOP = 9
    IMPROVE_LOGISTICS = 10

    # Optimization
    OPTIMIZE_POWER = 11
    OPTIMIZE_WATER = 12
    OPTIMIZE_AIR = 13

    # Emergency
    EMERGENCY_DIG = 14

    @classmethod
    def count(cls) -> int:
        """Return number of subgoal types."""
        return len(cls)

    def get_parent_goals(self) -> List[AbstractGoal]:
        """Return which abstract goals this subgoal can serve."""
        mapping = {
            SubgoalType.BUILD_OXYGEN_ROOM: [AbstractGoal.MAINTAIN_OXYGEN],
            SubgoalType.IMPROVE_VENTILATION: [AbstractGoal.MAINTAIN_OXYGEN],
            SubgoalType.BUILD_BATHROOM: [AbstractGoal.ENSURE_HAPPINESS, AbstractGoal.BUILD_INFRASTRUCTURE],
            SubgoalType.BUILD_BEDROOM: [AbstractGoal.ENSURE_HAPPINESS, AbstractGoal.BUILD_INFRASTRUCTURE],
            SubgoalType.BUILD_KITCHEN: [AbstractGoal.ENSURE_HAPPINESS, AbstractGoal.SECURE_RESOURCES],
            SubgoalType.BUILD_POWER_ROOM: [AbstractGoal.SECURE_RESOURCES, AbstractGoal.BUILD_INFRASTRUCTURE],
            SubgoalType.BUILD_FARM: [AbstractGoal.SECURE_RESOURCES],
            SubgoalType.BUILD_STORAGE: [AbstractGoal.SECURE_RESOURCES, AbstractGoal.BUILD_INFRASTRUCTURE],
            SubgoalType.EXPAND_LIVING_SPACE: [AbstractGoal.BUILD_INFRASTRUCTURE],
            SubgoalType.BUILD_WORKSHOP: [AbstractGoal.BUILD_INFRASTRUCTURE],
            SubgoalType.IMPROVE_LOGISTICS: [AbstractGoal.BUILD_INFRASTRUCTURE],
            SubgoalType.OPTIMIZE_POWER: [AbstractGoal.SECURE_RESOURCES],
            SubgoalType.OPTIMIZE_WATER: [AbstractGoal.MANAGE_WATER],
            SubgoalType.OPTIMIZE_AIR: [AbstractGoal.MAINTAIN_OXYGEN],
            SubgoalType.EMERGENCY_DIG: [
                AbstractGoal.MAINTAIN_OXYGEN,
                AbstractGoal.BUILD_INFRASTRUCTURE,
            ],
        }
        return mapping.get(self, [])


@dataclass
class GoalTerminationCondition:
    """
    Defines when a goal should terminate.

    Termination can be based on:
    - Achievement: goal completed successfully
    - Timeout: maximum duration exceeded
    - Failure: goal became impossible
    """
    # Threshold for success (goal-specific metric >= threshold)
    success_threshold: float

    # Metric name to check (from game state)
    metric_name: str

    # Number of cycles the threshold must be maintained
    maintenance_cycles: int = 3

    # Maximum cycles before forced termination
    max_cycles: int = 50

    def is_achieved(self, current_value: float, cycles_maintained: int) -> bool:
        """Check if goal is successfully achieved."""
        return (current_value >= self.success_threshold and
                cycles_maintained >= self.maintenance_cycles)

    def is_timeout(self, cycles_active: int) -> bool:
        """Check if goal has timed out."""
        return cycles_active >= self.max_cycles


@dataclass
class SubgoalTerminationCondition:
    """
    Defines when a subgoal should terminate.
    """
    # Success condition
    success_metric: str
    success_threshold: float

    # Maximum steps before timeout
    max_steps: int = 20

    # Minimum steps before allowing completion
    min_steps: int = 3

    def is_achieved(self, metrics: Dict[str, float]) -> bool:
        """Check if subgoal is successfully achieved."""
        if self.success_metric not in metrics:
            return False
        return metrics[self.success_metric] >= self.success_threshold

    def is_timeout(self, steps_active: int) -> bool:
        """Check if subgoal has timed out."""
        return steps_active >= self.max_steps


# Goal termination conditions
GOAL_TERMINATION_CONDITIONS: Dict[AbstractGoal, GoalTerminationCondition] = {
    AbstractGoal.MAINTAIN_OXYGEN: GoalTerminationCondition(
        success_threshold=0.3,
        metric_name='oxygen_ratio',
        maintenance_cycles=3,
        max_cycles=50
    ),
    AbstractGoal.MANAGE_WATER: GoalTerminationCondition(
        success_threshold=1.0,  # water_system_functional (boolean as float)
        metric_name='water_system_functional',
        maintenance_cycles=1,
        max_cycles=40
    ),
    AbstractGoal.ENSURE_HAPPINESS: GoalTerminationCondition(
        success_threshold=0.5,
        metric_name='happiness_ratio',
        maintenance_cycles=5,
        max_cycles=60
    ),
    AbstractGoal.BUILD_INFRASTRUCTURE: GoalTerminationCondition(
        success_threshold=10.0,  # number of essential buildings
        metric_name='essential_buildings_count',
        maintenance_cycles=1,
        max_cycles=50
    ),
    AbstractGoal.SECURE_RESOURCES: GoalTerminationCondition(
        success_threshold=1.0,  # power_balance > 0 as boolean
        metric_name='power_positive',
        maintenance_cycles=5,
        max_cycles=40
    ),
}


# Subgoal termination conditions
SUBGOAL_TERMINATION_CONDITIONS: Dict[SubgoalType, SubgoalTerminationCondition] = {
    SubgoalType.BUILD_OXYGEN_ROOM: SubgoalTerminationCondition(
        success_metric='has_oxygen_generator',
        success_threshold=1.0,
        max_steps=20
    ),
    SubgoalType.IMPROVE_VENTILATION: SubgoalTerminationCondition(
        success_metric='ventilation_coverage',
        success_threshold=0.5,
        max_steps=15
    ),
    SubgoalType.BUILD_BATHROOM: SubgoalTerminationCondition(
        success_metric='has_bathroom',
        success_threshold=1.0,
        max_steps=15
    ),
    SubgoalType.BUILD_BEDROOM: SubgoalTerminationCondition(
        success_metric='has_bedroom',
        success_threshold=1.0,
        max_steps=15
    ),
    SubgoalType.BUILD_KITCHEN: SubgoalTerminationCondition(
        success_metric='has_kitchen',
        success_threshold=1.0,
        max_steps=20
    ),
    SubgoalType.BUILD_POWER_ROOM: SubgoalTerminationCondition(
        success_metric='has_power_generation',
        success_threshold=1.0,
        max_steps=20
    ),
    SubgoalType.BUILD_FARM: SubgoalTerminationCondition(
        success_metric='has_farm',
        success_threshold=1.0,
        max_steps=25
    ),
    SubgoalType.BUILD_STORAGE: SubgoalTerminationCondition(
        success_metric='has_storage',
        success_threshold=1.0,
        max_steps=15
    ),
    SubgoalType.EXPAND_LIVING_SPACE: SubgoalTerminationCondition(
        success_metric='living_space_tiles',
        success_threshold=50.0,
        max_steps=30
    ),
    SubgoalType.BUILD_WORKSHOP: SubgoalTerminationCondition(
        success_metric='has_workshop',
        success_threshold=1.0,
        max_steps=20
    ),
    SubgoalType.IMPROVE_LOGISTICS: SubgoalTerminationCondition(
        success_metric='logistics_score',
        success_threshold=0.5,
        max_steps=25
    ),
    SubgoalType.OPTIMIZE_POWER: SubgoalTerminationCondition(
        success_metric='power_efficiency',
        success_threshold=0.7,
        max_steps=15
    ),
    SubgoalType.OPTIMIZE_WATER: SubgoalTerminationCondition(
        success_metric='water_efficiency',
        success_threshold=0.7,
        max_steps=15
    ),
    SubgoalType.OPTIMIZE_AIR: SubgoalTerminationCondition(
        success_metric='air_quality',
        success_threshold=0.8,
        max_steps=15
    ),
    SubgoalType.EMERGENCY_DIG: SubgoalTerminationCondition(
        success_metric='tiles_dug',
        success_threshold=5.0,
        max_steps=10
    ),
}


def get_goal_termination(goal: AbstractGoal) -> GoalTerminationCondition:
    """Get termination condition for a goal."""
    return GOAL_TERMINATION_CONDITIONS.get(
        goal,
        GoalTerminationCondition(
            success_threshold=1.0,
            metric_name='default',
            max_cycles=50
        )
    )


def get_subgoal_termination(subgoal: SubgoalType) -> SubgoalTerminationCondition:
    """Get termination condition for a subgoal."""
    return SUBGOAL_TERMINATION_CONDITIONS.get(
        subgoal,
        SubgoalTerminationCondition(
            success_metric='default',
            success_threshold=1.0,
            max_steps=20
        )
    )


def get_valid_subgoals_for_goal(goal: AbstractGoal) -> List[SubgoalType]:
    """Get list of subgoals that can serve a given goal."""
    valid_subgoals = []
    for subgoal in SubgoalType:
        if goal in subgoal.get_parent_goals():
            valid_subgoals.append(subgoal)
    return valid_subgoals


def goal_to_one_hot(goal: AbstractGoal) -> np.ndarray:
    """Convert goal to one-hot encoding."""
    one_hot = np.zeros(AbstractGoal.count(), dtype=np.float32)
    one_hot[goal.value] = 1.0
    return one_hot


def subgoal_to_one_hot(subgoal: SubgoalType) -> np.ndarray:
    """Convert subgoal to one-hot encoding."""
    one_hot = np.zeros(SubgoalType.count(), dtype=np.float32)
    one_hot[subgoal.value] = 1.0
    return one_hot


@dataclass
class GoalStatus:
    """Tracks the status of an active goal."""
    goal: AbstractGoal
    started_cycle: int
    cycles_active: int = 0
    cycles_maintained: int = 0
    is_completed: bool = False
    is_failed: bool = False
    completion_value: float = 0.0

    def update(self, current_value: float, threshold: float):
        """Update goal status with new observation."""
        self.cycles_active += 1
        self.completion_value = current_value

        if current_value >= threshold:
            self.cycles_maintained += 1
        else:
            self.cycles_maintained = 0


@dataclass
class SubgoalStatus:
    """Tracks the status of an active subgoal."""
    subgoal: SubgoalType
    started_step: int
    target_location: Optional[tuple] = None  # (x, y) if spatial
    steps_active: int = 0
    is_completed: bool = False
    is_failed: bool = False
    progress: float = 0.0

    def update(self, metrics: Dict[str, float], condition: SubgoalTerminationCondition):
        """Update subgoal status with new metrics."""
        self.steps_active += 1

        if condition.success_metric in metrics:
            self.progress = metrics[condition.success_metric] / condition.success_threshold
            self.progress = min(1.0, self.progress)
