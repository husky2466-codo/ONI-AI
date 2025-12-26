"""
Training infrastructure for hierarchical agent.

Task 5.1: Hierarchical Planning Architecture

Provides training loops and utilities for the hierarchical agent
with asymmetric updates across levels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import os
import json
from collections import deque

from .config import HierarchicalConfig, TrainingConfig
from .hierarchical_agent import HierarchicalAgent
from .coordinator import HierarchyBuffer


@dataclass
class TrainingState:
    """Tracks training progress."""
    total_timesteps: int = 0
    total_episodes: int = 0
    current_curriculum_stage: int = 0
    best_mean_reward: float = float('-inf')
    training_started: bool = False
    training_completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.total_episodes,
            'current_curriculum_stage': self.current_curriculum_stage,
            'best_mean_reward': self.best_mean_reward,
            'training_completed': self.training_completed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingState':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class HierarchicalPPOTrainer:
    """
    PPO trainer for the hierarchical agent.

    Features:
    - Asymmetric update frequencies for each level
    - Separate optimizers per level
    - Curriculum learning support
    - Integrated logging and checkpointing
    """

    def __init__(
        self,
        agent: HierarchicalAgent,
        config: Optional[HierarchicalConfig] = None
    ):
        """
        Initialize trainer.

        Args:
            agent: Hierarchical agent to train
            config: Training configuration
        """
        self.agent = agent
        self.config = config or agent.config
        self.training_config = self.config.training

        # Create optimizers
        self.hl_optimizer = optim.Adam(
            agent.high_level_planner.parameters(),
            lr=self.training_config.hl_learning_rate
        )
        self.ml_optimizer = optim.Adam(
            agent.mid_level_controller.parameters(),
            lr=self.training_config.ml_learning_rate
        )
        self.ll_optimizer = optim.Adam(
            agent.low_level_executor.parameters(),
            lr=self.training_config.ll_learning_rate
        )

        # Training state
        self.state = TrainingState()

        # Buffers for each level
        self.buffer = HierarchyBuffer(capacity=10000)

        # Metrics tracking
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.goal_completion_rates: deque = deque(maxlen=100)
        self.subgoal_completion_rates: deque = deque(maxlen=100)

        # Checkpointing
        self.checkpoint_dir = self.config.checkpoint_dir

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            advantages: Advantage estimates
            returns: Return estimates
        """
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        last_gae = 0
        last_value = values[-1] if not dones[-1] else 0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            if dones[t]:
                next_value = 0
                last_gae = 0

            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update_low_level(
        self,
        observations: torch.Tensor,
        subgoal_embeddings: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Update low-level executor with PPO.

        Args:
            observations: Batch of observations
            subgoal_embeddings: Batch of subgoal embeddings
            actions: Batch of actions
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            action_masks: Optional action masks

        Returns:
            Dictionary of loss values
        """
        # Evaluate actions
        log_probs, entropy, values = self.agent.low_level_executor.evaluate_actions(
            observations, subgoal_embeddings, actions, action_masks
        )

        # PPO ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped objective
        clip_range = self.training_config.clip_range
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss +
            self.training_config.vf_coef * value_loss +
            self.training_config.ent_coef * entropy_loss
        )

        # Update
        self.ll_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.low_level_executor.parameters(),
            max_norm=0.5
        )
        self.ll_optimizer.step()

        return {
            'll_policy_loss': policy_loss.item(),
            'll_value_loss': value_loss.item(),
            'll_entropy': -entropy_loss.item(),
            'll_total_loss': loss.item(),
        }

    def update_mid_level(
        self,
        observations: torch.Tensor,
        goal_embeddings: torch.Tensor,
        subgoals: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        subgoal_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Update mid-level controller with PPO.

        Args:
            observations: Batch of observations
            goal_embeddings: Batch of goal embeddings
            subgoals: Batch of subgoal indices
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            subgoal_masks: Optional subgoal masks

        Returns:
            Dictionary of loss values
        """
        # Evaluate subgoals
        log_probs, entropy, values, _ = self.agent.mid_level_controller.evaluate_subgoals(
            observations, goal_embeddings, subgoals, subgoal_masks
        )

        # PPO ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped objective
        clip_range = self.training_config.clip_range
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss +
            self.training_config.vf_coef * value_loss +
            self.training_config.ent_coef * entropy_loss
        )

        # Update
        self.ml_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.mid_level_controller.parameters(),
            max_norm=0.5
        )
        self.ml_optimizer.step()

        return {
            'ml_policy_loss': policy_loss.item(),
            'ml_value_loss': value_loss.item(),
            'ml_entropy': -entropy_loss.item(),
            'ml_total_loss': loss.item(),
        }

    def update_high_level(
        self,
        global_features: torch.Tensor,
        goals: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        goal_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Update high-level planner with PPO.

        Args:
            global_features: Batch of global features
            goals: Batch of goal indices
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            goal_masks: Optional goal masks

        Returns:
            Dictionary of loss values
        """
        # Evaluate goals
        log_probs, entropy, values = self.agent.high_level_planner.evaluate_goals(
            global_features, goals, goal_masks
        )

        # PPO ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped objective
        clip_range = self.training_config.clip_range
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss +
            self.training_config.vf_coef * value_loss +
            self.training_config.ent_coef * entropy_loss
        )

        # Update
        self.hl_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.high_level_planner.parameters(),
            max_norm=0.5
        )
        self.hl_optimizer.step()

        return {
            'hl_policy_loss': policy_loss.item(),
            'hl_value_loss': value_loss.item(),
            'hl_entropy': -entropy_loss.item(),
            'hl_total_loss': loss.item(),
        }

    def record_episode(
        self,
        episode_reward: float,
        episode_length: int,
        goal_completion_rate: float,
        subgoal_completion_rate: float
    ):
        """Record episode statistics."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.goal_completion_rates.append(goal_completion_rate)
        self.subgoal_completion_rates.append(subgoal_completion_rate)
        self.state.total_episodes += 1

    def get_training_statistics(self) -> Dict[str, float]:
        """Get current training statistics."""
        if len(self.episode_rewards) == 0:
            return {}

        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'mean_goal_completion': np.mean(self.goal_completion_rates),
            'mean_subgoal_completion': np.mean(self.subgoal_completion_rates),
            'total_timesteps': self.state.total_timesteps,
            'total_episodes': self.state.total_episodes,
        }

    def should_update_curriculum(self) -> bool:
        """Check if curriculum should advance."""
        if not self.training_config.use_curriculum:
            return False

        stages = self.training_config.curriculum_stages
        if self.state.current_curriculum_stage >= len(stages):
            return False

        threshold = stages[self.state.current_curriculum_stage]
        return self.state.total_episodes >= threshold

    def advance_curriculum(self):
        """Advance to next curriculum stage."""
        self.state.current_curriculum_stage += 1

    def save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint."""
        if name is None:
            name = f"checkpoint_{self.state.total_timesteps}"

        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save agent
        self.agent.save(checkpoint_path)

        # Save training state
        state_path = os.path.join(checkpoint_path, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

        # Save optimizer states
        torch.save({
            'hl_optimizer': self.hl_optimizer.state_dict(),
            'ml_optimizer': self.ml_optimizer.state_dict(),
            'll_optimizer': self.ll_optimizer.state_dict(),
        }, os.path.join(checkpoint_path, 'optimizers.pt'))

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        # Load agent
        self.agent.load(path)

        # Load training state
        state_path = os.path.join(path, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                self.state = TrainingState.from_dict(json.load(f))

        # Load optimizer states
        opt_path = os.path.join(path, 'optimizers.pt')
        if os.path.exists(opt_path):
            opt_states = torch.load(opt_path)
            self.hl_optimizer.load_state_dict(opt_states['hl_optimizer'])
            self.ml_optimizer.load_state_dict(opt_states['ml_optimizer'])
            self.ll_optimizer.load_state_dict(opt_states['ll_optimizer'])


def create_trainer(
    agent: HierarchicalAgent,
    config: Optional[HierarchicalConfig] = None
) -> HierarchicalPPOTrainer:
    """
    Factory function to create trainer.

    Args:
        agent: Agent to train
        config: Optional configuration

    Returns:
        HierarchicalPPOTrainer instance
    """
    return HierarchicalPPOTrainer(agent, config)
