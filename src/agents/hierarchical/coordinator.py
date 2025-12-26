"""
Hierarchical coordinator for inter-level communication.

Task 5.1: Hierarchical Planning Architecture

Manages the communication and coordination between:
- High-level planner (goal selection)
- Mid-level controller (subgoal decomposition)
- Low-level executor (action execution)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch

from .goal_types import (
    AbstractGoal, SubgoalType,
    GoalStatus, SubgoalStatus,
    get_goal_termination, get_subgoal_termination,
    get_valid_subgoals_for_goal
)
from .config import HierarchicalConfig, TemporalConfig


@dataclass
class HierarchyState:
    """
    Tracks the current state of the hierarchy.

    Contains information about active goals/subgoals,
    timing, and execution history.
    """
    # Current active goal
    current_goal: Optional[AbstractGoal] = None
    goal_status: Optional[GoalStatus] = None
    goal_embedding: Optional[torch.Tensor] = None

    # Current active subgoal
    current_subgoal: Optional[SubgoalType] = None
    subgoal_status: Optional[SubgoalStatus] = None
    subgoal_embedding: Optional[torch.Tensor] = None
    subgoal_params: Optional[np.ndarray] = None  # Spatial parameters

    # Timing
    current_cycle: int = 0
    current_step: int = 0
    steps_since_goal_change: int = 0
    steps_since_subgoal_change: int = 0

    # History
    goal_history: List[AbstractGoal] = field(default_factory=list)
    subgoal_history: List[SubgoalType] = field(default_factory=list)
    completed_goals: int = 0
    completed_subgoals: int = 0
    failed_goals: int = 0
    failed_subgoals: int = 0


class HierarchicalCoordinator:
    """
    Coordinates communication between hierarchy levels.

    Responsibilities:
    1. Track current goal/subgoal state
    2. Manage goal/subgoal embeddings
    3. Determine when to request new goals/subgoals
    4. Handle termination conditions
    5. Compute intrinsic rewards for completions
    """

    def __init__(self, config: Optional[HierarchicalConfig] = None):
        """
        Initialize the coordinator.

        Args:
            config: Hierarchical configuration
        """
        self.config = config or HierarchicalConfig()
        self.temporal = self.config.temporal

        # Current hierarchy state
        self.state = HierarchyState()

        # Tracking for metrics
        self._episode_goals = 0
        self._episode_subgoals = 0
        self._episode_goal_completions = 0
        self._episode_subgoal_completions = 0

    def reset(self):
        """Reset coordinator for new episode."""
        self.state = HierarchyState()
        self._episode_goals = 0
        self._episode_subgoals = 0
        self._episode_goal_completions = 0
        self._episode_subgoal_completions = 0

    def needs_new_goal(self) -> bool:
        """
        Check if a new goal should be selected.

        Returns True if:
        - No current goal
        - Current goal completed
        - Current goal failed/timed out
        - High-level interval reached
        """
        # No current goal
        if self.state.current_goal is None:
            return True

        # Goal completed or failed
        if self.state.goal_status is not None:
            if self.state.goal_status.is_completed:
                return True
            if self.state.goal_status.is_failed:
                return True

        # Check timeout
        if self.state.goal_status is not None:
            termination = get_goal_termination(self.state.current_goal)
            if termination.is_timeout(self.state.goal_status.cycles_active):
                return True

        # High-level interval check (for re-evaluation)
        if self.state.steps_since_goal_change >= self.temporal.high_level_interval * 10:
            # Allow goal re-evaluation, but don't force change
            pass

        return False

    def needs_new_subgoal(self) -> bool:
        """
        Check if a new subgoal should be selected.

        Returns True if:
        - No current subgoal
        - Current subgoal completed
        - Current subgoal failed/timed out
        - Mid-level interval reached
        """
        # No current subgoal
        if self.state.current_subgoal is None:
            return True

        # Subgoal completed or failed
        if self.state.subgoal_status is not None:
            if self.state.subgoal_status.is_completed:
                return True
            if self.state.subgoal_status.is_failed:
                return True

        # Check timeout
        if self.state.subgoal_status is not None:
            termination = get_subgoal_termination(self.state.current_subgoal)
            if termination.is_timeout(self.state.subgoal_status.steps_active):
                return True

        return False

    def set_goal(
        self,
        goal: AbstractGoal,
        embedding: torch.Tensor,
        cycle: int
    ):
        """
        Set a new active goal.

        Args:
            goal: The new abstract goal
            embedding: Goal embedding tensor
            cycle: Current game cycle
        """
        # Record previous goal in history
        if self.state.current_goal is not None:
            self.state.goal_history.append(self.state.current_goal)

        # Set new goal
        self.state.current_goal = goal
        self.state.goal_embedding = embedding.detach() if isinstance(embedding, torch.Tensor) else embedding
        self.state.goal_status = GoalStatus(
            goal=goal,
            started_cycle=cycle
        )
        self.state.steps_since_goal_change = 0

        # Clear subgoal (new goal means new subgoal needed)
        self.state.current_subgoal = None
        self.state.subgoal_status = None
        self.state.subgoal_embedding = None
        self.state.subgoal_params = None
        self.state.steps_since_subgoal_change = 0

        self._episode_goals += 1

    def set_subgoal(
        self,
        subgoal: SubgoalType,
        embedding: torch.Tensor,
        params: Optional[np.ndarray] = None,
        step: int = 0
    ):
        """
        Set a new active subgoal.

        Args:
            subgoal: The new subgoal type
            embedding: Subgoal embedding tensor
            params: Spatial parameters (x, y, width, height)
            step: Current step
        """
        # Record previous subgoal in history
        if self.state.current_subgoal is not None:
            self.state.subgoal_history.append(self.state.current_subgoal)

        # Set new subgoal
        self.state.current_subgoal = subgoal
        self.state.subgoal_embedding = embedding.detach() if isinstance(embedding, torch.Tensor) else embedding
        self.state.subgoal_params = params
        self.state.subgoal_status = SubgoalStatus(
            subgoal=subgoal,
            started_step=step,
            target_location=tuple(params[:2]) if params is not None else None
        )
        self.state.steps_since_subgoal_change = 0

        self._episode_subgoals += 1

    def update(
        self,
        game_metrics: Dict[str, float],
        cycle: int,
        step: int
    ) -> Dict[str, Any]:
        """
        Update coordinator state based on game metrics.

        Args:
            game_metrics: Dictionary of current game metrics
            cycle: Current game cycle
            step: Current step

        Returns:
            Dictionary with update results including termination events
        """
        self.state.current_cycle = cycle
        self.state.current_step = step
        self.state.steps_since_goal_change += 1
        self.state.steps_since_subgoal_change += 1

        results = {
            'goal_completed': False,
            'goal_failed': False,
            'subgoal_completed': False,
            'subgoal_failed': False,
            'goal_progress': 0.0,
            'subgoal_progress': 0.0,
        }

        # Update goal status
        if self.state.current_goal is not None and self.state.goal_status is not None:
            termination = get_goal_termination(self.state.current_goal)

            # Get current metric value
            metric_value = game_metrics.get(termination.metric_name, 0.0)
            self.state.goal_status.update(metric_value, termination.success_threshold)

            # Check completion
            if termination.is_achieved(metric_value, self.state.goal_status.cycles_maintained):
                self.state.goal_status.is_completed = True
                self.state.completed_goals += 1
                self._episode_goal_completions += 1
                results['goal_completed'] = True

            # Check timeout
            elif termination.is_timeout(self.state.goal_status.cycles_active):
                self.state.goal_status.is_failed = True
                self.state.failed_goals += 1
                results['goal_failed'] = True

            # Compute progress
            results['goal_progress'] = min(1.0, metric_value / termination.success_threshold)

        # Update subgoal status
        if self.state.current_subgoal is not None and self.state.subgoal_status is not None:
            termination = get_subgoal_termination(self.state.current_subgoal)
            self.state.subgoal_status.update(game_metrics, termination)

            # Check completion
            if termination.is_achieved(game_metrics):
                self.state.subgoal_status.is_completed = True
                self.state.completed_subgoals += 1
                self._episode_subgoal_completions += 1
                results['subgoal_completed'] = True

            # Check timeout
            elif termination.is_timeout(self.state.subgoal_status.steps_active):
                self.state.subgoal_status.is_failed = True
                self.state.failed_subgoals += 1
                results['subgoal_failed'] = True

            results['subgoal_progress'] = self.state.subgoal_status.progress

        return results

    def get_valid_subgoals(self) -> List[SubgoalType]:
        """Get list of valid subgoals for current goal."""
        if self.state.current_goal is None:
            return list(SubgoalType)
        return get_valid_subgoals_for_goal(self.state.current_goal)

    def get_goal_embedding(self) -> Optional[torch.Tensor]:
        """Get current goal embedding."""
        return self.state.goal_embedding

    def get_subgoal_embedding(self) -> Optional[torch.Tensor]:
        """Get current subgoal embedding."""
        return self.state.subgoal_embedding

    def get_hierarchy_info(self) -> Dict[str, Any]:
        """
        Get information about current hierarchy state.

        Returns:
            Dictionary with hierarchy information for logging/debugging
        """
        return {
            'current_goal': self.state.current_goal.name if self.state.current_goal else None,
            'current_subgoal': self.state.current_subgoal.name if self.state.current_subgoal else None,
            'goal_cycles_active': self.state.goal_status.cycles_active if self.state.goal_status else 0,
            'subgoal_steps_active': self.state.subgoal_status.steps_active if self.state.subgoal_status else 0,
            'goal_progress': self.state.goal_status.completion_value if self.state.goal_status else 0.0,
            'subgoal_progress': self.state.subgoal_status.progress if self.state.subgoal_status else 0.0,
            'completed_goals': self.state.completed_goals,
            'completed_subgoals': self.state.completed_subgoals,
            'steps_since_goal_change': self.state.steps_since_goal_change,
            'steps_since_subgoal_change': self.state.steps_since_subgoal_change,
        }

    def get_episode_statistics(self) -> Dict[str, float]:
        """
        Get episode-level statistics.

        Returns:
            Dictionary with episode statistics
        """
        goal_completion_rate = (
            self._episode_goal_completions / max(1, self._episode_goals)
        )
        subgoal_completion_rate = (
            self._episode_subgoal_completions / max(1, self._episode_subgoals)
        )

        return {
            'episode_goals': self._episode_goals,
            'episode_subgoals': self._episode_subgoals,
            'episode_goal_completions': self._episode_goal_completions,
            'episode_subgoal_completions': self._episode_subgoal_completions,
            'goal_completion_rate': goal_completion_rate,
            'subgoal_completion_rate': subgoal_completion_rate,
        }

    def create_goal_mask(self, game_metrics: Dict[str, float]) -> np.ndarray:
        """
        Create mask for valid goals based on current state.

        Some goals may not be appropriate given current game state.

        Args:
            game_metrics: Current game metrics

        Returns:
            Boolean mask for valid goals
        """
        mask = np.ones(AbstractGoal.count(), dtype=bool)

        # Example: Don't pursue oxygen if already achieved
        if game_metrics.get('oxygen_ratio', 0) >= 0.4:
            # Can still select, but lower priority
            pass

        # All goals are generally valid
        return mask

    def create_subgoal_mask(self, game_metrics: Dict[str, float]) -> np.ndarray:
        """
        Create mask for valid subgoals based on current state and goal.

        Args:
            game_metrics: Current game metrics

        Returns:
            Boolean mask for valid subgoals
        """
        mask = np.zeros(SubgoalType.count(), dtype=bool)

        # Get subgoals valid for current goal
        valid_subgoals = self.get_valid_subgoals()
        for subgoal in valid_subgoals:
            mask[subgoal.value] = True

        # Additional filtering based on game state
        # (e.g., can't build bathroom if already have one)
        if game_metrics.get('has_bathroom', 0) >= 1:
            mask[SubgoalType.BUILD_BATHROOM.value] = False

        if game_metrics.get('has_bedroom', 0) >= 1:
            mask[SubgoalType.BUILD_BEDROOM.value] = False

        # Ensure at least one subgoal is valid
        if not mask.any():
            # Fall back to emergency dig
            mask[SubgoalType.EMERGENCY_DIG.value] = True

        return mask


class HierarchyBuffer:
    """
    Buffer for storing hierarchical trajectories.

    Stores transitions at each level of the hierarchy
    for training purposes.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize hierarchy buffer.

        Args:
            capacity: Maximum number of transitions per level
        """
        self.capacity = capacity

        # High-level buffer (goal transitions)
        self.hl_states: List[np.ndarray] = []
        self.hl_goals: List[int] = []
        self.hl_rewards: List[float] = []
        self.hl_values: List[float] = []
        self.hl_log_probs: List[float] = []
        self.hl_dones: List[bool] = []

        # Mid-level buffer (subgoal transitions)
        self.ml_states: List[np.ndarray] = []
        self.ml_goal_embeddings: List[np.ndarray] = []
        self.ml_subgoals: List[int] = []
        self.ml_params: List[np.ndarray] = []
        self.ml_rewards: List[float] = []
        self.ml_values: List[float] = []
        self.ml_log_probs: List[float] = []
        self.ml_dones: List[bool] = []

        # Low-level buffer (action transitions)
        self.ll_states: List[np.ndarray] = []
        self.ll_subgoal_embeddings: List[np.ndarray] = []
        self.ll_actions: List[int] = []
        self.ll_rewards: List[float] = []
        self.ll_values: List[float] = []
        self.ll_log_probs: List[float] = []
        self.ll_dones: List[bool] = []

    def add_high_level(
        self,
        state: np.ndarray,
        goal: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add high-level transition."""
        self.hl_states.append(state)
        self.hl_goals.append(goal)
        self.hl_rewards.append(reward)
        self.hl_values.append(value)
        self.hl_log_probs.append(log_prob)
        self.hl_dones.append(done)

        # Trim if over capacity
        if len(self.hl_states) > self.capacity:
            self.hl_states.pop(0)
            self.hl_goals.pop(0)
            self.hl_rewards.pop(0)
            self.hl_values.pop(0)
            self.hl_log_probs.pop(0)
            self.hl_dones.pop(0)

    def add_mid_level(
        self,
        state: np.ndarray,
        goal_embedding: np.ndarray,
        subgoal: int,
        params: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add mid-level transition."""
        self.ml_states.append(state)
        self.ml_goal_embeddings.append(goal_embedding)
        self.ml_subgoals.append(subgoal)
        self.ml_params.append(params)
        self.ml_rewards.append(reward)
        self.ml_values.append(value)
        self.ml_log_probs.append(log_prob)
        self.ml_dones.append(done)

        if len(self.ml_states) > self.capacity:
            self._trim_mid_level()

    def add_low_level(
        self,
        state: np.ndarray,
        subgoal_embedding: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add low-level transition."""
        self.ll_states.append(state)
        self.ll_subgoal_embeddings.append(subgoal_embedding)
        self.ll_actions.append(action)
        self.ll_rewards.append(reward)
        self.ll_values.append(value)
        self.ll_log_probs.append(log_prob)
        self.ll_dones.append(done)

        if len(self.ll_states) > self.capacity:
            self._trim_low_level()

    def _trim_mid_level(self):
        """Trim mid-level buffer."""
        self.ml_states.pop(0)
        self.ml_goal_embeddings.pop(0)
        self.ml_subgoals.pop(0)
        self.ml_params.pop(0)
        self.ml_rewards.pop(0)
        self.ml_values.pop(0)
        self.ml_log_probs.pop(0)
        self.ml_dones.pop(0)

    def _trim_low_level(self):
        """Trim low-level buffer."""
        self.ll_states.pop(0)
        self.ll_subgoal_embeddings.pop(0)
        self.ll_actions.pop(0)
        self.ll_rewards.pop(0)
        self.ll_values.pop(0)
        self.ll_log_probs.pop(0)
        self.ll_dones.pop(0)

    def clear(self):
        """Clear all buffers."""
        self.hl_states.clear()
        self.hl_goals.clear()
        self.hl_rewards.clear()
        self.hl_values.clear()
        self.hl_log_probs.clear()
        self.hl_dones.clear()

        self.ml_states.clear()
        self.ml_goal_embeddings.clear()
        self.ml_subgoals.clear()
        self.ml_params.clear()
        self.ml_rewards.clear()
        self.ml_values.clear()
        self.ml_log_probs.clear()
        self.ml_dones.clear()

        self.ll_states.clear()
        self.ll_subgoal_embeddings.clear()
        self.ll_actions.clear()
        self.ll_rewards.clear()
        self.ll_values.clear()
        self.ll_log_probs.clear()
        self.ll_dones.clear()

    def size(self) -> Dict[str, int]:
        """Get buffer sizes for each level."""
        return {
            'high_level': len(self.hl_states),
            'mid_level': len(self.ml_states),
            'low_level': len(self.ll_states),
        }
