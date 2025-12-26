"""
Mid-level controller for goal decomposition.

Task 5.1: Hierarchical Planning Architecture

The mid-level controller decomposes abstract goals into concrete subgoals,
using goal-conditioned attention to focus on relevant parts of the state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .goal_types import SubgoalType
from .goal_embeddings import GoalConditionedAttention
from .config import HierarchicalConfig


class MidLevelController(nn.Module):
    """
    Mid-level controller for decomposing goals into subgoals.

    Architecture:
    - Input: Goal embedding (64-dim) + Full state (8256-dim)
    - Attention: Goal-conditioned attention on spatial features
    - Output: Subgoal distribution (15 types) + Spatial parameters (4-dim)

    Operates every ~5 steps and selects subgoals with spatial targets.
    """

    def __init__(
        self,
        goal_dim: int = 64,
        spatial_channels: int = 8,
        spatial_size: int = 32,
        global_dim: int = 64,
        num_subgoals: int = 15,
        hidden_dims: Tuple[int, ...] = (256, 128),
        use_attention: bool = True,
        attention_heads: int = 4,
        dropout_rate: float = 0.1
    ):
        """
        Initialize mid-level controller.

        Args:
            goal_dim: Dimension of goal embedding
            spatial_channels: Number of spatial feature channels
            spatial_size: Spatial feature map size (H=W)
            global_dim: Dimension of global features
            num_subgoals: Number of subgoal types
            hidden_dims: Hidden layer dimensions
            use_attention: Whether to use goal-conditioned attention
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.goal_dim = goal_dim
        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size
        self.global_dim = global_dim
        self.num_subgoals = num_subgoals
        self.use_attention = use_attention

        # Goal-conditioned attention
        if use_attention:
            self.goal_attention = GoalConditionedAttention(
                goal_dim=goal_dim,
                spatial_channels=spatial_channels,
                spatial_size=spatial_size,
                hidden_dim=hidden_dims[0],
                num_heads=attention_heads
            )
            attention_output_dim = hidden_dims[0]
        else:
            # Simple spatial pooling
            self.spatial_pool = nn.Sequential(
                nn.Conv2d(spatial_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            attention_output_dim = 64

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Combined feature dimension
        combined_dim = attention_output_dim + 64 + 64  # attention + global + goal

        # Shared trunk
        trunk_layers = []
        prev_dim = combined_dim
        for hidden_dim in hidden_dims:
            trunk_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*trunk_layers)
        self.feature_dim = prev_dim

        # Subgoal selection head (policy)
        self.subgoal_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, num_subgoals)
        )

        # Spatial parameter head (predicts x, y, width, height)
        self.spatial_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 4),
            nn.Sigmoid()  # Normalize to [0, 1]
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Smaller initialization for output layers
        nn.init.orthogonal_(self.subgoal_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.spatial_head[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def _reshape_spatial(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Extract and reshape spatial features from observation.

        Args:
            observation: Full observation (batch, spatial_dim + global_dim)

        Returns:
            Spatial features (batch, C, H, W)
        """
        batch_size = observation.shape[0]
        spatial_dim = self.spatial_channels * self.spatial_size * self.spatial_size

        # Extract spatial part
        spatial_flat = observation[:, :spatial_dim]

        # Reshape to (batch, C, H, W)
        spatial = spatial_flat.view(
            batch_size,
            self.spatial_channels,
            self.spatial_size,
            self.spatial_size
        )

        return spatial

    def _extract_global(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Extract global features from observation.

        Args:
            observation: Full observation

        Returns:
            Global features (batch, global_dim)
        """
        spatial_dim = self.spatial_channels * self.spatial_size * self.spatial_size
        return observation[:, spatial_dim:]

    def forward(
        self,
        observation: torch.Tensor,
        goal_embedding: torch.Tensor,
        subgoal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            observation: Full state observation (batch, obs_dim)
            goal_embedding: Goal embedding (batch, goal_dim)
            subgoal_mask: Optional mask for valid subgoals (batch, num_subgoals)

        Returns:
            subgoal_logits: Logits for subgoal selection
            spatial_params: Predicted spatial parameters (x, y, w, h) in [0, 1]
            value: Value estimate
            attention_weights: Attention weights if using attention, else None
        """
        # Extract spatial and global features
        spatial = self._reshape_spatial(observation)
        global_features = self._extract_global(observation)

        # Apply goal-conditioned attention or pooling
        attention_weights = None
        if self.use_attention:
            attended, attention_weights = self.goal_attention(goal_embedding, spatial)
        else:
            attended = self.spatial_pool(spatial)

        # Encode global features and goal
        global_encoded = self.global_encoder(global_features)
        goal_encoded = self.goal_encoder(goal_embedding)

        # Combine all features
        combined = torch.cat([attended, global_encoded, goal_encoded], dim=-1)

        # Pass through trunk
        features = self.trunk(combined)

        # Get outputs
        subgoal_logits = self.subgoal_head(features)
        spatial_params = self.spatial_head(features)
        value = self.value_head(features)

        # Apply subgoal mask if provided
        if subgoal_mask is not None:
            subgoal_logits = subgoal_logits.masked_fill(~subgoal_mask, float('-inf'))

        return subgoal_logits, spatial_params, value, attention_weights

    def select_subgoal(
        self,
        observation: torch.Tensor,
        goal_embedding: torch.Tensor,
        subgoal_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select a subgoal given current state and goal.

        Args:
            observation: Full state observation
            goal_embedding: Goal embedding
            subgoal_mask: Optional mask for valid subgoals
            deterministic: If True, select most likely subgoal

        Returns:
            subgoal: Selected subgoal index (batch,)
            spatial_params: Predicted spatial parameters
            log_prob: Log probability of selected subgoal
            value: Value estimate
        """
        subgoal_logits, spatial_params, value, _ = self.forward(
            observation, goal_embedding, subgoal_mask
        )

        if deterministic:
            subgoal = subgoal_logits.argmax(dim=-1)
            dist = torch.distributions.Categorical(logits=subgoal_logits)
            log_prob = dist.log_prob(subgoal)
        else:
            dist = torch.distributions.Categorical(logits=subgoal_logits)
            subgoal = dist.sample()
            log_prob = dist.log_prob(subgoal)

        return subgoal, spatial_params, log_prob, value.squeeze(-1)

    def evaluate_subgoals(
        self,
        observation: torch.Tensor,
        goal_embedding: torch.Tensor,
        subgoals: torch.Tensor,
        subgoal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given subgoals.

        Used during PPO training.

        Args:
            observation: Full state observation
            goal_embedding: Goal embedding
            subgoals: Subgoal indices to evaluate
            subgoal_mask: Optional mask for valid subgoals

        Returns:
            log_prob: Log probabilities of subgoals
            entropy: Distribution entropy
            value: Value estimates
            spatial_params: Predicted spatial parameters
        """
        subgoal_logits, spatial_params, value, _ = self.forward(
            observation, goal_embedding, subgoal_mask
        )

        dist = torch.distributions.Categorical(logits=subgoal_logits)
        log_prob = dist.log_prob(subgoals)
        entropy = dist.entropy()

        return log_prob, entropy, value.squeeze(-1), spatial_params

    def get_attention_map(
        self,
        observation: torch.Tensor,
        goal_embedding: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Get attention map for visualization.

        Args:
            observation: Full state observation
            goal_embedding: Goal embedding

        Returns:
            Attention weights reshaped to (batch, H, W) or None
        """
        if not self.use_attention:
            return None

        _, _, _, attention_weights = self.forward(observation, goal_embedding)

        if attention_weights is not None:
            batch_size = observation.shape[0]
            attention_map = attention_weights.view(
                batch_size, self.spatial_size, self.spatial_size
            )
            return attention_map

        return None


class MidLevelControllerSimple(nn.Module):
    """
    Simplified mid-level controller without attention.

    Uses direct concatenation of goal embedding with flattened observation.
    """

    def __init__(
        self,
        observation_dim: int = 8256,
        goal_dim: int = 64,
        num_subgoals: int = 15,
        hidden_dims: Tuple[int, ...] = (512, 256, 128)
    ):
        """Initialize simplified controller."""
        super().__init__()

        self.num_subgoals = num_subgoals

        # MLP trunk
        layers = []
        prev_dim = observation_dim + goal_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.subgoal_head = nn.Linear(prev_dim, num_subgoals)
        self.spatial_head = nn.Sequential(
            nn.Linear(prev_dim, 4),
            nn.Sigmoid()
        )
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(
        self,
        observation: torch.Tensor,
        goal_embedding: torch.Tensor,
        subgoal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        combined = torch.cat([observation, goal_embedding], dim=-1)
        features = self.trunk(combined)

        subgoal_logits = self.subgoal_head(features)
        spatial_params = self.spatial_head(features)
        value = self.value_head(features)

        if subgoal_mask is not None:
            subgoal_logits = subgoal_logits.masked_fill(~subgoal_mask, float('-inf'))

        return subgoal_logits, spatial_params, value


def create_mid_level_controller(config: HierarchicalConfig) -> MidLevelController:
    """
    Factory function to create mid-level controller from config.

    Args:
        config: Hierarchical configuration

    Returns:
        Configured MidLevelController instance
    """
    return MidLevelController(
        goal_dim=config.goals.goal_embedding_dim,
        spatial_channels=config.spatial_channels,
        spatial_size=config.spatial_size,
        global_dim=config.global_features_dim,
        num_subgoals=config.goals.num_subgoals,
        hidden_dims=tuple(config.network.ml_hidden_dims),
        use_attention=config.network.ml_use_attention,
        attention_heads=config.network.ml_attention_heads,
        dropout_rate=config.network.dropout_rate
    )
