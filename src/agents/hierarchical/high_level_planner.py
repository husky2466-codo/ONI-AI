"""
High-level planner for abstract goal selection.

Task 5.1: Hierarchical Planning Architecture

The high-level planner operates on a slow timescale (every 20 cycles)
and selects abstract goals that guide the mid-level controller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .goal_types import AbstractGoal
from .config import HierarchicalConfig


class HighLevelPlanner(nn.Module):
    """
    High-level planner for selecting abstract goals.

    Architecture:
    - Input: Global state features (64-dim)
    - Hidden: MLP with configurable layers
    - Output: Goal distribution (5 goals) + Value estimate

    Operates every ~20 game cycles and selects persistent goals
    that guide the lower levels of the hierarchy.
    """

    def __init__(
        self,
        state_dim: int = 64,
        num_goals: int = 5,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = 'relu',
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize high-level planner.

        Args:
            state_dim: Dimension of global state features
            num_goals: Number of abstract goal types
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            use_layer_norm: Whether to use layer normalization
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.num_goals = num_goals

        # Select activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        else:
            act_fn = nn.ReLU

        # Build shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)
        self.feature_dim = prev_dim

        # Policy head (goal selection)
        self.policy_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            act_fn(),
            nn.Linear(self.feature_dim // 2, num_goals)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            act_fn(),
            nn.Linear(self.feature_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Smaller initialization for output layers
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
        goal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Global state features (batch_size, state_dim)
            goal_mask: Optional mask for valid goals (batch_size, num_goals)

        Returns:
            goal_logits: Logits for goal selection (batch_size, num_goals)
            value: Value estimate (batch_size, 1)
        """
        features = self.shared_net(state)

        goal_logits = self.policy_head(features)
        value = self.value_head(features)

        # Apply goal mask if provided
        if goal_mask is not None:
            goal_logits = goal_logits.masked_fill(~goal_mask, float('-inf'))

        return goal_logits, value

    def get_goal_distribution(
        self,
        state: torch.Tensor,
        goal_mask: Optional[torch.Tensor] = None
    ) -> torch.distributions.Categorical:
        """
        Get goal distribution for sampling.

        Args:
            state: Global state features
            goal_mask: Optional mask for valid goals

        Returns:
            Categorical distribution over goals
        """
        goal_logits, _ = self.forward(state, goal_mask)
        return torch.distributions.Categorical(logits=goal_logits)

    def select_goal(
        self,
        state: torch.Tensor,
        goal_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select a goal given current state.

        Args:
            state: Global state features (batch_size, state_dim)
            goal_mask: Optional mask for valid goals
            deterministic: If True, select most likely goal

        Returns:
            goal: Selected goal index (batch_size,)
            log_prob: Log probability of selected goal
            value: Value estimate
        """
        goal_logits, value = self.forward(state, goal_mask)

        if deterministic:
            goal = goal_logits.argmax(dim=-1)
            dist = torch.distributions.Categorical(logits=goal_logits)
            log_prob = dist.log_prob(goal)
        else:
            dist = torch.distributions.Categorical(logits=goal_logits)
            goal = dist.sample()
            log_prob = dist.log_prob(goal)

        return goal, log_prob, value.squeeze(-1)

    def evaluate_goals(
        self,
        state: torch.Tensor,
        goals: torch.Tensor,
        goal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given goals.

        Used during PPO training.

        Args:
            state: Global state features (batch_size, state_dim)
            goals: Goal indices to evaluate (batch_size,)
            goal_mask: Optional mask for valid goals

        Returns:
            log_prob: Log probabilities of goals
            entropy: Distribution entropy
            value: Value estimates
        """
        goal_logits, value = self.forward(state, goal_mask)

        dist = torch.distributions.Categorical(logits=goal_logits)
        log_prob = dist.log_prob(goals)
        entropy = dist.entropy()

        return log_prob, entropy, value.squeeze(-1)

    def get_goal_probabilities(
        self,
        state: torch.Tensor,
        goal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get probability distribution over goals.

        Args:
            state: Global state features
            goal_mask: Optional mask for valid goals

        Returns:
            Probability distribution (batch_size, num_goals)
        """
        goal_logits, _ = self.forward(state, goal_mask)
        return F.softmax(goal_logits, dim=-1)


class HighLevelPlannerWithContext(HighLevelPlanner):
    """
    Extended high-level planner with history context.

    Includes information about previous goals and their outcomes
    to make better planning decisions.
    """

    def __init__(
        self,
        state_dim: int = 64,
        num_goals: int = 5,
        context_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (256, 256),
        **kwargs
    ):
        """
        Initialize planner with context.

        Args:
            state_dim: Dimension of global state features
            num_goals: Number of abstract goal types
            context_dim: Dimension of context embedding
            hidden_dims: Hidden layer dimensions
        """
        # Adjust state_dim to include context
        super().__init__(
            state_dim=state_dim + context_dim,
            num_goals=num_goals,
            hidden_dims=hidden_dims,
            **kwargs
        )

        self.base_state_dim = state_dim
        self.context_dim = context_dim

        # Context encoder (encodes goal history)
        self.context_encoder = nn.Sequential(
            nn.Linear(num_goals * 3, context_dim),  # last 3 goals one-hot
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )

    def forward_with_context(
        self,
        state: torch.Tensor,
        goal_history: torch.Tensor,
        goal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with goal history context.

        Args:
            state: Global state features (batch_size, base_state_dim)
            goal_history: Last 3 goals one-hot encoded (batch_size, num_goals * 3)
            goal_mask: Optional mask for valid goals

        Returns:
            goal_logits: Logits for goal selection
            value: Value estimate
        """
        # Encode context
        context = self.context_encoder(goal_history)

        # Concatenate state and context
        augmented_state = torch.cat([state, context], dim=-1)

        return self.forward(augmented_state, goal_mask)


def create_high_level_planner(config: HierarchicalConfig) -> HighLevelPlanner:
    """
    Factory function to create high-level planner from config.

    Args:
        config: Hierarchical configuration

    Returns:
        Configured HighLevelPlanner instance
    """
    return HighLevelPlanner(
        state_dim=config.global_features_dim,
        num_goals=config.goals.num_abstract_goals,
        hidden_dims=tuple(config.network.hl_hidden_dims),
        activation=config.network.hl_activation,
        use_layer_norm=config.network.use_layer_norm,
        dropout_rate=config.network.dropout_rate
    )
