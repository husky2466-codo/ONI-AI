"""
Low-level executor for primitive action selection.

Task 5.1: Hierarchical Planning Architecture

The low-level executor maps subgoal embeddings to primitive actions,
operating at every environment step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .config import HierarchicalConfig


class ResidualBlock(nn.Module):
    """Residual block for spatial feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=not use_batch_norm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=not use_batch_norm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class SpatialFeatureExtractor(nn.Module):
    """
    CNN-based spatial feature extractor.

    Similar to MiniONIFeaturesExtractor but optimized for
    subgoal-conditioned action selection.
    """

    def __init__(
        self,
        spatial_channels: int = 8,
        spatial_size: int = 32,
        conv_channels: Tuple[int, ...] = (32, 64, 128),
        features_dim: int = 256,
        use_residual: bool = True,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size

        if use_residual:
            # Build residual CNN
            layers = []
            prev_channels = spatial_channels
            for i, out_channels in enumerate(conv_channels):
                stride = 2 if i > 0 else 1
                layers.append(ResidualBlock(
                    prev_channels, out_channels,
                    stride=stride,
                    use_batch_norm=use_batch_norm
                ))
                prev_channels = out_channels

            self.cnn = nn.Sequential(*layers)
        else:
            # Simple CNN
            layers = []
            prev_channels = spatial_channels
            for out_channels in conv_channels:
                layers.extend([
                    nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                    nn.ReLU()
                ])
                prev_channels = out_channels

            self.cnn = nn.Sequential(*layers)

        # Calculate output size
        with torch.no_grad():
            dummy = torch.zeros(1, spatial_channels, spatial_size, spatial_size)
            cnn_out = self.cnn(dummy)
            cnn_flat_dim = cnn_out.numel()

        # Final projection
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(cnn_flat_dim, features_dim),
            nn.ReLU()
        )

        self.features_dim = features_dim

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        """
        Extract features from spatial input.

        Args:
            spatial: Spatial features (batch, C, H, W)

        Returns:
            Features (batch, features_dim)
        """
        x = self.cnn(spatial)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class LowLevelExecutor(nn.Module):
    """
    Low-level executor for primitive action selection.

    Architecture:
    - Input: Subgoal embedding (32-dim) + Full state (8256-dim)
    - CNN: Spatial feature extraction with ResNet-style blocks
    - Output: Action distribution (200 actions) + Value estimate

    Operates at every environment step, conditioned on the current subgoal.
    """

    def __init__(
        self,
        subgoal_dim: int = 32,
        spatial_channels: int = 8,
        spatial_size: int = 32,
        global_dim: int = 64,
        num_actions: int = 200,
        features_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_residual: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize low-level executor.

        Args:
            subgoal_dim: Dimension of subgoal embedding
            spatial_channels: Number of spatial feature channels
            spatial_size: Spatial feature map size
            global_dim: Dimension of global features
            num_actions: Number of primitive actions
            features_dim: CNN feature output dimension
            hidden_dims: Hidden layer dimensions for policy/value heads
            use_residual: Use residual connections in CNN
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.subgoal_dim = subgoal_dim
        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size
        self.global_dim = global_dim
        self.num_actions = num_actions

        # Spatial feature extractor
        self.spatial_extractor = SpatialFeatureExtractor(
            spatial_channels=spatial_channels,
            spatial_size=spatial_size,
            features_dim=features_dim,
            use_residual=use_residual
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Subgoal encoder
        self.subgoal_encoder = nn.Sequential(
            nn.Linear(subgoal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Combined dimension
        combined_dim = features_dim + 64 + 64  # spatial + global + subgoal

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

        # Policy head (action selection)
        self.policy_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, num_actions)
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
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

        # Smaller initialization for output layers
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def _reshape_observation(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape observation into spatial and global components.

        Args:
            observation: Full observation (batch, obs_dim)

        Returns:
            spatial: Spatial features (batch, C, H, W)
            global_features: Global features (batch, global_dim)
        """
        batch_size = observation.shape[0]
        spatial_dim = self.spatial_channels * self.spatial_size * self.spatial_size

        # Extract and reshape spatial
        spatial_flat = observation[:, :spatial_dim]
        spatial = spatial_flat.view(
            batch_size,
            self.spatial_channels,
            self.spatial_size,
            self.spatial_size
        )

        # Extract global
        global_features = observation[:, spatial_dim:]

        return spatial, global_features

    def forward(
        self,
        observation: torch.Tensor,
        subgoal_embedding: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            observation: Full state observation (batch, obs_dim)
            subgoal_embedding: Subgoal embedding (batch, subgoal_dim)
            action_mask: Optional mask for valid actions (batch, num_actions)

        Returns:
            action_logits: Logits for action selection
            value: Value estimate
        """
        # Reshape observation
        spatial, global_features = self._reshape_observation(observation)

        # Extract features
        spatial_features = self.spatial_extractor(spatial)
        global_encoded = self.global_encoder(global_features)
        subgoal_encoded = self.subgoal_encoder(subgoal_embedding)

        # Combine features
        combined = torch.cat([spatial_features, global_encoded, subgoal_encoded], dim=-1)

        # Pass through trunk
        features = self.trunk(combined)

        # Get outputs
        action_logits = self.policy_head(features)
        value = self.value_head(features)

        # Apply action mask
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))

        return action_logits, value

    def select_action(
        self,
        observation: torch.Tensor,
        subgoal_embedding: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select an action given current state and subgoal.

        Args:
            observation: Full state observation
            subgoal_embedding: Subgoal embedding
            action_mask: Optional mask for valid actions
            deterministic: If True, select most likely action

        Returns:
            action: Selected action index (batch,)
            log_prob: Log probability of selected action
            value: Value estimate
        """
        action_logits, value = self.forward(observation, subgoal_embedding, action_mask)

        if deterministic:
            action = action_logits.argmax(dim=-1)
            dist = torch.distributions.Categorical(logits=action_logits)
            log_prob = dist.log_prob(action)
        else:
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        subgoal_embedding: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given actions.

        Used during PPO training.

        Args:
            observation: Full state observation
            subgoal_embedding: Subgoal embedding
            actions: Action indices to evaluate
            action_mask: Optional mask for valid actions

        Returns:
            log_prob: Log probabilities of actions
            entropy: Distribution entropy
            value: Value estimates
        """
        action_logits, value = self.forward(observation, subgoal_embedding, action_mask)

        dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy, value.squeeze(-1)

    def get_action_probabilities(
        self,
        observation: torch.Tensor,
        subgoal_embedding: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get probability distribution over actions.

        Args:
            observation: Full state observation
            subgoal_embedding: Subgoal embedding
            action_mask: Optional mask for valid actions

        Returns:
            Action probabilities (batch, num_actions)
        """
        action_logits, _ = self.forward(observation, subgoal_embedding, action_mask)
        return F.softmax(action_logits, dim=-1)


class LowLevelExecutorSimple(nn.Module):
    """
    Simplified low-level executor without CNN.

    Uses direct MLP on flattened observation.
    """

    def __init__(
        self,
        observation_dim: int = 8256,
        subgoal_dim: int = 32,
        num_actions: int = 200,
        hidden_dims: Tuple[int, ...] = (512, 256, 256)
    ):
        """Initialize simplified executor."""
        super().__init__()

        self.num_actions = num_actions

        # MLP trunk
        layers = []
        prev_dim = observation_dim + subgoal_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.policy_head = nn.Linear(prev_dim, num_actions)
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(
        self,
        observation: torch.Tensor,
        subgoal_embedding: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        combined = torch.cat([observation, subgoal_embedding], dim=-1)
        features = self.trunk(combined)

        action_logits = self.policy_head(features)
        value = self.value_head(features)

        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))

        return action_logits, value


def create_low_level_executor(config: HierarchicalConfig) -> LowLevelExecutor:
    """
    Factory function to create low-level executor from config.

    Args:
        config: Hierarchical configuration

    Returns:
        Configured LowLevelExecutor instance
    """
    return LowLevelExecutor(
        subgoal_dim=config.goals.subgoal_embedding_dim,
        spatial_channels=config.spatial_channels,
        spatial_size=config.spatial_size,
        global_dim=config.global_features_dim,
        num_actions=config.num_actions,
        features_dim=256,
        hidden_dims=tuple(config.network.ll_hidden_dims),
        use_residual=config.network.ll_use_residual,
        dropout_rate=config.network.dropout_rate
    )
