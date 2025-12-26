"""
Intrinsic reward system for hierarchical planning.

Task 5.1: Hierarchical Planning Architecture

Provides intrinsic motivation for subgoal/goal completion
to guide learning in the hierarchical architecture.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
import numpy as np

from .goal_types import (
    AbstractGoal, SubgoalType,
    get_goal_termination, get_subgoal_termination
)
from .config import IntrinsicRewardConfig


@dataclass
class IntrinsicRewardBreakdown:
    """Breakdown of intrinsic reward components."""
    subgoal_completion: float = 0.0
    goal_completion: float = 0.0
    subgoal_progress: float = 0.0
    goal_progress: float = 0.0
    exploration_bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'subgoal_completion': self.subgoal_completion,
            'goal_completion': self.goal_completion,
            'subgoal_progress': self.subgoal_progress,
            'goal_progress': self.goal_progress,
            'exploration_bonus': self.exploration_bonus,
            'total': self.total,
        }


class HierarchicalIntrinsicRewards:
    """
    Computes intrinsic rewards for the hierarchical architecture.

    Intrinsic rewards include:
    1. Subgoal completion bonus
    2. Goal completion bonus
    3. Progress-based shaping rewards
    4. (Optional) Exploration bonuses
    """

    def __init__(self, config: Optional[IntrinsicRewardConfig] = None):
        """
        Initialize intrinsic reward calculator.

        Args:
            config: Intrinsic reward configuration
        """
        self.config = config or IntrinsicRewardConfig()

        # Tracking for progress-based rewards
        self._prev_goal_progress: Optional[float] = None
        self._prev_subgoal_progress: Optional[float] = None

        # State visitation counts for exploration bonus
        self._state_counts: Dict[int, int] = {}

    def reset(self):
        """Reset for new episode."""
        self._prev_goal_progress = None
        self._prev_subgoal_progress = None
        self._state_counts.clear()

    def compute_reward(
        self,
        current_goal: Optional[AbstractGoal],
        current_subgoal: Optional[SubgoalType],
        game_metrics: Dict[str, float],
        goal_completed: bool = False,
        goal_failed: bool = False,
        subgoal_completed: bool = False,
        subgoal_failed: bool = False,
        state_hash: Optional[int] = None
    ) -> Tuple[float, IntrinsicRewardBreakdown]:
        """
        Compute total intrinsic reward.

        Args:
            current_goal: Current active goal
            current_subgoal: Current active subgoal
            game_metrics: Dictionary of game state metrics
            goal_completed: Whether goal was just completed
            goal_failed: Whether goal just failed
            subgoal_completed: Whether subgoal was just completed
            subgoal_failed: Whether subgoal just failed
            state_hash: Optional hash of current state for exploration

        Returns:
            total_reward: Total intrinsic reward
            breakdown: Breakdown of reward components
        """
        if not self.config.use_intrinsic_rewards:
            return 0.0, IntrinsicRewardBreakdown()

        breakdown = IntrinsicRewardBreakdown()

        # Subgoal completion bonus
        if subgoal_completed:
            breakdown.subgoal_completion = self.config.subgoal_completion_bonus
        elif subgoal_failed:
            # Small penalty for failed subgoals
            breakdown.subgoal_completion = -self.config.subgoal_completion_bonus * 0.1

        # Goal completion bonus
        if goal_completed:
            breakdown.goal_completion = self.config.goal_completion_bonus
        elif goal_failed:
            # Small penalty for failed goals
            breakdown.goal_completion = -self.config.goal_completion_bonus * 0.1

        # Progress-based shaping
        if self.config.use_progress_shaping:
            goal_shaping, subgoal_shaping = self._compute_progress_shaping(
                current_goal, current_subgoal, game_metrics
            )
            breakdown.goal_progress = goal_shaping
            breakdown.subgoal_progress = subgoal_shaping

        # Exploration bonus
        if self.config.use_exploration_bonus and state_hash is not None:
            breakdown.exploration_bonus = self._compute_exploration_bonus(state_hash)

        # Total
        breakdown.total = (
            breakdown.subgoal_completion +
            breakdown.goal_completion +
            breakdown.subgoal_progress +
            breakdown.goal_progress +
            breakdown.exploration_bonus
        )

        return breakdown.total, breakdown

    def _compute_progress_shaping(
        self,
        current_goal: Optional[AbstractGoal],
        current_subgoal: Optional[SubgoalType],
        game_metrics: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Compute progress-based shaping rewards.

        Uses potential-based shaping: F(s,s') = gamma * Phi(s') - Phi(s)
        where Phi is the progress toward goal/subgoal completion.

        Args:
            current_goal: Current active goal
            current_subgoal: Current active subgoal
            game_metrics: Dictionary of game state metrics

        Returns:
            goal_shaping: Goal progress shaping reward
            subgoal_shaping: Subgoal progress shaping reward
        """
        gamma = 0.99
        scale = self.config.progress_shaping_scale

        goal_shaping = 0.0
        subgoal_shaping = 0.0

        # Goal progress shaping
        if current_goal is not None:
            termination = get_goal_termination(current_goal)
            metric_value = game_metrics.get(termination.metric_name, 0.0)
            current_progress = min(1.0, metric_value / termination.success_threshold)

            if self._prev_goal_progress is not None:
                goal_shaping = scale * (gamma * current_progress - self._prev_goal_progress)

            self._prev_goal_progress = current_progress

        # Subgoal progress shaping
        if current_subgoal is not None:
            termination = get_subgoal_termination(current_subgoal)
            metric_value = game_metrics.get(termination.success_metric, 0.0)
            current_progress = min(1.0, metric_value / termination.success_threshold)

            if self._prev_subgoal_progress is not None:
                subgoal_shaping = scale * (gamma * current_progress - self._prev_subgoal_progress)

            self._prev_subgoal_progress = current_progress

        return goal_shaping, subgoal_shaping

    def _compute_exploration_bonus(self, state_hash: int) -> float:
        """
        Compute exploration bonus based on state visitation counts.

        Uses count-based exploration: bonus = scale / sqrt(count)

        Args:
            state_hash: Hash of current state

        Returns:
            Exploration bonus
        """
        # Increment count
        self._state_counts[state_hash] = self._state_counts.get(state_hash, 0) + 1
        count = self._state_counts[state_hash]

        # Bonus decreases with visitation count
        bonus = self.config.exploration_bonus_scale / np.sqrt(count)

        return bonus

    def on_goal_change(self):
        """Called when goal changes."""
        self._prev_goal_progress = None
        self._prev_subgoal_progress = None

    def on_subgoal_change(self):
        """Called when subgoal changes."""
        self._prev_subgoal_progress = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get intrinsic reward statistics."""
        return {
            'num_visited_states': len(self._state_counts),
            'total_state_visits': sum(self._state_counts.values()),
        }


class SubgoalAchievementTracker:
    """
    Tracks subgoal achievements for curriculum and analysis.
    """

    def __init__(self):
        """Initialize tracker."""
        self._subgoal_attempts: Dict[SubgoalType, int] = {}
        self._subgoal_completions: Dict[SubgoalType, int] = {}
        self._subgoal_failures: Dict[SubgoalType, int] = {}
        self._goal_attempts: Dict[AbstractGoal, int] = {}
        self._goal_completions: Dict[AbstractGoal, int] = {}
        self._goal_failures: Dict[AbstractGoal, int] = {}

    def record_subgoal_attempt(self, subgoal: SubgoalType):
        """Record a subgoal attempt."""
        self._subgoal_attempts[subgoal] = self._subgoal_attempts.get(subgoal, 0) + 1

    def record_subgoal_completion(self, subgoal: SubgoalType):
        """Record a subgoal completion."""
        self._subgoal_completions[subgoal] = self._subgoal_completions.get(subgoal, 0) + 1

    def record_subgoal_failure(self, subgoal: SubgoalType):
        """Record a subgoal failure."""
        self._subgoal_failures[subgoal] = self._subgoal_failures.get(subgoal, 0) + 1

    def record_goal_attempt(self, goal: AbstractGoal):
        """Record a goal attempt."""
        self._goal_attempts[goal] = self._goal_attempts.get(goal, 0) + 1

    def record_goal_completion(self, goal: AbstractGoal):
        """Record a goal completion."""
        self._goal_completions[goal] = self._goal_completions.get(goal, 0) + 1

    def record_goal_failure(self, goal: AbstractGoal):
        """Record a goal failure."""
        self._goal_failures[goal] = self._goal_failures.get(goal, 0) + 1

    def get_subgoal_success_rate(self, subgoal: SubgoalType) -> float:
        """Get success rate for a subgoal."""
        attempts = self._subgoal_attempts.get(subgoal, 0)
        if attempts == 0:
            return 0.0
        completions = self._subgoal_completions.get(subgoal, 0)
        return completions / attempts

    def get_goal_success_rate(self, goal: AbstractGoal) -> float:
        """Get success rate for a goal."""
        attempts = self._goal_attempts.get(goal, 0)
        if attempts == 0:
            return 0.0
        completions = self._goal_completions.get(goal, 0)
        return completions / attempts

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        subgoal_rates = {
            sg.name: self.get_subgoal_success_rate(sg)
            for sg in SubgoalType
        }
        goal_rates = {
            g.name: self.get_goal_success_rate(g)
            for g in AbstractGoal
        }

        total_subgoal_attempts = sum(self._subgoal_attempts.values())
        total_subgoal_completions = sum(self._subgoal_completions.values())
        total_goal_attempts = sum(self._goal_attempts.values())
        total_goal_completions = sum(self._goal_completions.values())

        return {
            'subgoal_success_rates': subgoal_rates,
            'goal_success_rates': goal_rates,
            'overall_subgoal_rate': (
                total_subgoal_completions / max(1, total_subgoal_attempts)
            ),
            'overall_goal_rate': (
                total_goal_completions / max(1, total_goal_attempts)
            ),
            'total_subgoal_attempts': total_subgoal_attempts,
            'total_goal_attempts': total_goal_attempts,
        }

    def reset(self):
        """Reset all counters."""
        self._subgoal_attempts.clear()
        self._subgoal_completions.clear()
        self._subgoal_failures.clear()
        self._goal_attempts.clear()
        self._goal_completions.clear()
        self._goal_failures.clear()


def create_intrinsic_reward_calculator(
    config: Optional[IntrinsicRewardConfig] = None
) -> HierarchicalIntrinsicRewards:
    """
    Factory function to create intrinsic reward calculator.

    Args:
        config: Optional configuration

    Returns:
        HierarchicalIntrinsicRewards instance
    """
    return HierarchicalIntrinsicRewards(config)
