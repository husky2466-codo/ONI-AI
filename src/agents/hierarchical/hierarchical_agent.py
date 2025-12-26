"""
Main hierarchical agent combining all components.

Task 5.1: Hierarchical Planning Architecture

Integrates:
- High-level planner (goal selection)
- Mid-level controller (subgoal decomposition)
- Low-level executor (action selection)
- Coordinator (inter-level communication)
- Intrinsic rewards (motivation system)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import os
import json

from .config import HierarchicalConfig
from .goal_types import AbstractGoal, SubgoalType, goal_to_one_hot, subgoal_to_one_hot
from .goal_embeddings import HierarchicalEmbeddingSystem
from .coordinator import HierarchicalCoordinator
from .high_level_planner import HighLevelPlanner, create_high_level_planner
from .mid_level_controller import MidLevelController, create_mid_level_controller
from .low_level_executor import LowLevelExecutor, create_low_level_executor
from .intrinsic_rewards import HierarchicalIntrinsicRewards, SubgoalAchievementTracker


class HierarchicalAgent(nn.Module):
    """
    Main hierarchical agent for the ONI-AI project.

    Combines high-level planning, mid-level control, and low-level execution
    with goal-based coordination and intrinsic motivation.
    """

    def __init__(self, config: Optional[HierarchicalConfig] = None):
        """
        Initialize hierarchical agent.

        Args:
            config: Hierarchical configuration
        """
        super().__init__()

        self.config = config or HierarchicalConfig()

        # Create hierarchical components
        self.embedding_system = HierarchicalEmbeddingSystem(
            num_goals=self.config.goals.num_abstract_goals,
            num_subgoals=self.config.goals.num_subgoals,
            state_dim=self.config.global_features_dim,
            goal_embed_dim=self.config.goals.goal_embedding_dim,
            subgoal_embed_dim=self.config.goals.subgoal_embedding_dim,
            spatial_channels=self.config.spatial_channels,
            spatial_size=self.config.spatial_size
        )

        self.high_level_planner = create_high_level_planner(self.config)
        self.mid_level_controller = create_mid_level_controller(self.config)
        self.low_level_executor = create_low_level_executor(self.config)

        # Coordination and rewards
        self.coordinator = HierarchicalCoordinator(self.config)
        self.intrinsic_rewards = HierarchicalIntrinsicRewards(self.config.intrinsic)
        self.achievement_tracker = SubgoalAchievementTracker()

        # Device
        self.device = torch.device('cpu')

        # Training mode flag
        self._is_training = True

    def to(self, device):
        """Move agent to device."""
        super().to(device)
        self.device = device
        return self

    def reset(self):
        """Reset agent for new episode."""
        self.coordinator.reset()
        self.intrinsic_rewards.reset()
        self.embedding_system.clear_cache()

    def _extract_global_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract global features from observation."""
        spatial_dim = (
            self.config.spatial_channels *
            self.config.spatial_size *
            self.config.spatial_size
        )
        return observation[:, spatial_dim:]

    def select_action(
        self,
        observation: np.ndarray,
        game_metrics: Dict[str, float],
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action using the full hierarchy.

        Args:
            observation: Environment observation
            game_metrics: Current game metrics
            action_mask: Optional mask for valid actions
            deterministic: If True, use greedy action selection

        Returns:
            action: Selected action index
            info: Dictionary with hierarchy information
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        global_features = self._extract_global_features(obs_tensor)

        # Get masks
        goal_mask = self.coordinator.create_goal_mask(game_metrics)
        subgoal_mask = self.coordinator.create_subgoal_mask(game_metrics)

        goal_mask_tensor = torch.BoolTensor(goal_mask).unsqueeze(0).to(self.device)
        subgoal_mask_tensor = torch.BoolTensor(subgoal_mask).unsqueeze(0).to(self.device)

        if action_mask is not None:
            action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
        else:
            action_mask_tensor = None

        info = {}

        # Step 1: Check if we need a new goal
        if self.coordinator.needs_new_goal():
            goal, goal_log_prob, goal_value = self.high_level_planner.select_goal(
                global_features, goal_mask_tensor, deterministic=deterministic
            )
            goal_enum = AbstractGoal(goal.item())

            # Compute goal embedding
            goal_embedding = self.embedding_system.compute_goal_embedding(
                goal, global_features, cache=True
            )

            # Set new goal in coordinator
            self.coordinator.set_goal(
                goal_enum,
                goal_embedding,
                game_metrics.get('cycle', 0)
            )

            # Track goal attempt
            self.achievement_tracker.record_goal_attempt(goal_enum)
            self.intrinsic_rewards.on_goal_change()

            info['goal_selected'] = True
            info['goal_log_prob'] = goal_log_prob.item()
            info['goal_value'] = goal_value.item()
        else:
            info['goal_selected'] = False

        # Get current goal embedding
        goal_embedding = self.coordinator.get_goal_embedding()
        if goal_embedding is None:
            # Fallback: select a goal
            goal, _, _ = self.high_level_planner.select_goal(
                global_features, goal_mask_tensor, deterministic=True
            )
            goal_embedding = self.embedding_system.compute_goal_embedding(
                goal, global_features, cache=True
            )

        # Step 2: Check if we need a new subgoal
        if self.coordinator.needs_new_subgoal():
            subgoal, spatial_params, subgoal_log_prob, subgoal_value = \
                self.mid_level_controller.select_subgoal(
                    obs_tensor, goal_embedding, subgoal_mask_tensor,
                    deterministic=deterministic
                )
            subgoal_enum = SubgoalType(subgoal.item())

            # Convert spatial params to numpy
            spatial_params_np = spatial_params.squeeze(0).detach().cpu().numpy()

            # Compute subgoal embedding
            subgoal_embedding = self.embedding_system.compute_subgoal_embedding(
                subgoal, spatial_params, goal_embedding, cache=True
            )

            # Set new subgoal in coordinator
            self.coordinator.set_subgoal(
                subgoal_enum,
                subgoal_embedding,
                spatial_params_np,
                step=game_metrics.get('step', 0)
            )

            # Track subgoal attempt
            self.achievement_tracker.record_subgoal_attempt(subgoal_enum)
            self.intrinsic_rewards.on_subgoal_change()

            info['subgoal_selected'] = True
            info['subgoal_log_prob'] = subgoal_log_prob.item()
            info['subgoal_value'] = subgoal_value.item()
            info['subgoal_params'] = spatial_params_np.tolist()
        else:
            info['subgoal_selected'] = False

        # Get current subgoal embedding
        subgoal_embedding = self.coordinator.get_subgoal_embedding()
        if subgoal_embedding is None:
            # Fallback: select a subgoal
            subgoal, spatial_params, _, _ = self.mid_level_controller.select_subgoal(
                obs_tensor, goal_embedding, subgoal_mask_tensor, deterministic=True
            )
            subgoal_embedding = self.embedding_system.compute_subgoal_embedding(
                subgoal, spatial_params, goal_embedding, cache=True
            )

        # Step 3: Select low-level action
        action, action_log_prob, action_value = self.low_level_executor.select_action(
            obs_tensor, subgoal_embedding, action_mask_tensor,
            deterministic=deterministic
        )

        info['action_log_prob'] = action_log_prob.item()
        info['action_value'] = action_value.item()

        # Add hierarchy info
        hierarchy_info = self.coordinator.get_hierarchy_info()
        info.update(hierarchy_info)

        return action.item(), info

    def update_and_get_intrinsic_reward(
        self,
        game_metrics: Dict[str, float],
        cycle: int,
        step: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Update coordinator and compute intrinsic reward.

        Args:
            game_metrics: Current game metrics
            cycle: Current game cycle
            step: Current step

        Returns:
            intrinsic_reward: Total intrinsic reward
            update_info: Dictionary with update information
        """
        # Update coordinator
        update_results = self.coordinator.update(game_metrics, cycle, step)

        # Track achievements
        if update_results['goal_completed']:
            self.achievement_tracker.record_goal_completion(
                self.coordinator.state.current_goal
            )
        elif update_results['goal_failed']:
            self.achievement_tracker.record_goal_failure(
                self.coordinator.state.current_goal
            )

        if update_results['subgoal_completed']:
            self.achievement_tracker.record_subgoal_completion(
                self.coordinator.state.current_subgoal
            )
        elif update_results['subgoal_failed']:
            self.achievement_tracker.record_subgoal_failure(
                self.coordinator.state.current_subgoal
            )

        # Compute intrinsic reward
        intrinsic_reward, breakdown = self.intrinsic_rewards.compute_reward(
            current_goal=self.coordinator.state.current_goal,
            current_subgoal=self.coordinator.state.current_subgoal,
            game_metrics=game_metrics,
            goal_completed=update_results['goal_completed'],
            goal_failed=update_results['goal_failed'],
            subgoal_completed=update_results['subgoal_completed'],
            subgoal_failed=update_results['subgoal_failed']
        )

        update_info = {
            **update_results,
            'intrinsic_reward': intrinsic_reward,
            'reward_breakdown': breakdown.to_dict()
        }

        return intrinsic_reward, update_info

    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get episode-level statistics."""
        coordinator_stats = self.coordinator.get_episode_statistics()
        achievement_stats = self.achievement_tracker.get_summary()
        intrinsic_stats = self.intrinsic_rewards.get_statistics()

        return {
            'coordinator': coordinator_stats,
            'achievements': achievement_stats,
            'intrinsic': intrinsic_stats,
        }

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self._is_training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def save(self, path: str):
        """
        Save agent state.

        Args:
            path: Path to save directory or file
        """
        # Create directory if needed
        save_dir = os.path.dirname(path) if '.' in os.path.basename(path) else path
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Determine file paths
        if path.endswith('.pt') or path.endswith('.pth'):
            model_path = path
            config_path = path.replace('.pt', '_config.json').replace('.pth', '_config.json')
        else:
            model_path = os.path.join(path, 'hierarchical_agent.pt')
            config_path = os.path.join(path, 'config.json')

        # Save model state
        state_dict = {
            'embedding_system': self.embedding_system.state_dict(),
            'high_level_planner': self.high_level_planner.state_dict(),
            'mid_level_controller': self.mid_level_controller.state_dict(),
            'low_level_executor': self.low_level_executor.state_dict(),
        }
        torch.save(state_dict, model_path)

        # Save config
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def load(self, path: str):
        """
        Load agent state.

        Args:
            path: Path to saved model
        """
        # Determine file path
        if path.endswith('.pt') or path.endswith('.pth'):
            model_path = path
        else:
            model_path = os.path.join(path, 'hierarchical_agent.pt')

        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)

        self.embedding_system.load_state_dict(state_dict['embedding_system'])
        self.high_level_planner.load_state_dict(state_dict['high_level_planner'])
        self.mid_level_controller.load_state_dict(state_dict['mid_level_controller'])
        self.low_level_executor.load_state_dict(state_dict['low_level_executor'])


def create_hierarchical_agent(
    config: Optional[HierarchicalConfig] = None,
    device: str = 'auto'
) -> HierarchicalAgent:
    """
    Factory function to create hierarchical agent.

    Args:
        config: Optional configuration
        device: Device to use ('cpu', 'cuda', or 'auto')

    Returns:
        HierarchicalAgent instance
    """
    agent = HierarchicalAgent(config)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent.to(torch.device(device))

    return agent
