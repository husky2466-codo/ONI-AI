"""
Goal embedding networks for hierarchical planning.

Task 5.1: Hierarchical Planning Architecture

Provides learnable embeddings for goals and subgoals that enable
communication between hierarchy levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

from .goal_types import AbstractGoal, SubgoalType


class GoalEmbeddingNetwork(nn.Module):
    """
    Learns dense embeddings for abstract goals.

    The embedding combines:
    1. A learned goal-specific embedding
    2. A state-conditioned context vector

    This allows goals to have both fixed identity and
    dynamic context-aware representations.
    """

    def __init__(
        self,
        num_goals: int = 5,
        state_dim: int = 64,
        embed_dim: int = 64,
        hidden_dim: int = 128
    ):
        """
        Initialize goal embedding network.

        Args:
            num_goals: Number of abstract goal types
            state_dim: Dimension of global state features
            embed_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.num_goals = num_goals
        self.embed_dim = embed_dim

        # Learnable goal embeddings (identity component)
        self.goal_embed = nn.Embedding(num_goals, embed_dim // 2)

        # State encoder (context component)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim // 2)
        )

        # Fusion layer to combine goal identity and context
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize embeddings with small random values
        nn.init.normal_(self.goal_embed.weight, mean=0.0, std=0.1)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        goal_id: torch.Tensor,
        state_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute goal embedding.

        Args:
            goal_id: Goal index tensor (batch_size,) or (batch_size, 1)
            state_features: Global state features (batch_size, state_dim)

        Returns:
            Goal embedding tensor (batch_size, embed_dim)
        """
        # Ensure goal_id is 1D
        if goal_id.dim() > 1:
            goal_id = goal_id.squeeze(-1)

        # Get goal identity embedding
        goal_identity = self.goal_embed(goal_id)  # (batch, embed_dim // 2)

        # Get state-conditioned context
        state_context = self.state_encoder(state_features)  # (batch, embed_dim // 2)

        # Concatenate and fuse
        combined = torch.cat([goal_identity, state_context], dim=-1)
        embedding = self.fusion(combined)

        return embedding

    def get_all_embeddings(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for all goals given current state.

        Args:
            state_features: Global state features (batch_size, state_dim)

        Returns:
            All goal embeddings (batch_size, num_goals, embed_dim)
        """
        batch_size = state_features.shape[0]

        # Create goal indices for all goals
        goal_ids = torch.arange(
            self.num_goals,
            device=state_features.device
        ).unsqueeze(0).expand(batch_size, -1)  # (batch, num_goals)

        # Get embeddings for each goal
        embeddings = []
        for i in range(self.num_goals):
            goal_id = goal_ids[:, i]
            embed = self.forward(goal_id, state_features)
            embeddings.append(embed)

        return torch.stack(embeddings, dim=1)  # (batch, num_goals, embed_dim)


class SubgoalEmbeddingNetwork(nn.Module):
    """
    Creates embeddings for subgoals.

    Combines subgoal type with spatial target information
    to create a unified subgoal representation.
    """

    def __init__(
        self,
        num_subgoals: int = 15,
        goal_embed_dim: int = 64,
        embed_dim: int = 32,
        spatial_param_dim: int = 4,  # x, y, width, height
        hidden_dim: int = 64
    ):
        """
        Initialize subgoal embedding network.

        Args:
            num_subgoals: Number of subgoal types
            goal_embed_dim: Dimension of parent goal embedding
            embed_dim: Output embedding dimension
            spatial_param_dim: Dimension of spatial parameters
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.num_subgoals = num_subgoals
        self.embed_dim = embed_dim

        # Learnable subgoal type embeddings
        self.subgoal_embed = nn.Embedding(num_subgoals, embed_dim // 2)

        # Spatial parameter encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_param_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim // 4)
        )

        # Goal context encoder (from parent goal)
        self.goal_context_encoder = nn.Sequential(
            nn.Linear(goal_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim // 4)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        nn.init.normal_(self.subgoal_embed.weight, mean=0.0, std=0.1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        subgoal_id: torch.Tensor,
        spatial_params: torch.Tensor,
        goal_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute subgoal embedding.

        Args:
            subgoal_id: Subgoal index tensor (batch_size,)
            spatial_params: Spatial parameters (batch_size, 4) - x, y, w, h
            goal_embedding: Parent goal embedding (batch_size, goal_embed_dim)

        Returns:
            Subgoal embedding tensor (batch_size, embed_dim)
        """
        # Ensure subgoal_id is 1D
        if subgoal_id.dim() > 1:
            subgoal_id = subgoal_id.squeeze(-1)

        # Get subgoal type embedding
        subgoal_type = self.subgoal_embed(subgoal_id)  # (batch, embed_dim // 2)

        # Encode spatial parameters
        spatial_features = self.spatial_encoder(spatial_params)  # (batch, embed_dim // 4)

        # Encode goal context
        goal_context = self.goal_context_encoder(goal_embedding)  # (batch, embed_dim // 4)

        # Combine all components
        combined = torch.cat([subgoal_type, spatial_features, goal_context], dim=-1)
        embedding = self.fusion(combined)

        return embedding

    def get_subgoal_type_embedding(self, subgoal_id: torch.Tensor) -> torch.Tensor:
        """Get just the subgoal type embedding (without spatial/goal context)."""
        if subgoal_id.dim() > 1:
            subgoal_id = subgoal_id.squeeze(-1)
        return self.subgoal_embed(subgoal_id)


class GoalConditionedAttention(nn.Module):
    """
    Attention mechanism conditioned on goal embedding.

    Used by the mid-level controller to focus on goal-relevant
    regions of the spatial state representation.
    """

    def __init__(
        self,
        goal_dim: int = 64,
        spatial_channels: int = 8,
        spatial_size: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4
    ):
        """
        Initialize goal-conditioned attention.

        Args:
            goal_dim: Goal embedding dimension
            spatial_channels: Number of spatial feature channels
            spatial_size: Spatial feature map size
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
        """
        super().__init__()

        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project goal to query
        self.goal_to_query = nn.Linear(goal_dim, hidden_dim)

        # Project spatial features to key/value
        self.spatial_to_key = nn.Conv2d(spatial_channels, hidden_dim, kernel_size=1)
        self.spatial_to_value = nn.Conv2d(spatial_channels, hidden_dim, kernel_size=1)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        goal_embedding: torch.Tensor,
        spatial_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply goal-conditioned attention to spatial features.

        Args:
            goal_embedding: Goal embedding (batch_size, goal_dim)
            spatial_features: Spatial features (batch_size, C, H, W)

        Returns:
            attended_features: Attended spatial features (batch_size, hidden_dim)
            attention_weights: Attention weights (batch_size, H*W)
        """
        batch_size = goal_embedding.shape[0]

        # Goal as query
        query = self.goal_to_query(goal_embedding)  # (batch, hidden)
        query = query.unsqueeze(1)  # (batch, 1, hidden)

        # Spatial features as key/value
        key = self.spatial_to_key(spatial_features)    # (batch, hidden, H, W)
        value = self.spatial_to_value(spatial_features)  # (batch, hidden, H, W)

        # Reshape to sequence format
        key = key.flatten(2).permute(0, 2, 1)    # (batch, H*W, hidden)
        value = value.flatten(2).permute(0, 2, 1)  # (batch, H*W, hidden)

        # Apply attention
        attended, attention_weights = self.attention(
            query, key, value,
            need_weights=True
        )

        # Squeeze and project
        attended = attended.squeeze(1)  # (batch, hidden)
        attended = self.output_proj(attended)

        # Reshape attention weights for visualization
        attention_weights = attention_weights.squeeze(1)  # (batch, H*W)

        return attended, attention_weights


class HierarchicalEmbeddingSystem(nn.Module):
    """
    Combined embedding system for the hierarchical architecture.

    Manages both goal and subgoal embeddings with proper caching
    and efficient computation.
    """

    def __init__(
        self,
        num_goals: int = 5,
        num_subgoals: int = 15,
        state_dim: int = 64,
        goal_embed_dim: int = 64,
        subgoal_embed_dim: int = 32,
        spatial_channels: int = 8,
        spatial_size: int = 32
    ):
        """Initialize the hierarchical embedding system."""
        super().__init__()

        self.goal_embedder = GoalEmbeddingNetwork(
            num_goals=num_goals,
            state_dim=state_dim,
            embed_dim=goal_embed_dim
        )

        self.subgoal_embedder = SubgoalEmbeddingNetwork(
            num_subgoals=num_subgoals,
            goal_embed_dim=goal_embed_dim,
            embed_dim=subgoal_embed_dim
        )

        self.goal_attention = GoalConditionedAttention(
            goal_dim=goal_embed_dim,
            spatial_channels=spatial_channels,
            spatial_size=spatial_size
        )

        # Cache for current embeddings
        self._cached_goal_embedding: Optional[torch.Tensor] = None
        self._cached_subgoal_embedding: Optional[torch.Tensor] = None

    def compute_goal_embedding(
        self,
        goal_id: torch.Tensor,
        state_features: torch.Tensor,
        cache: bool = True
    ) -> torch.Tensor:
        """Compute and optionally cache goal embedding."""
        embedding = self.goal_embedder(goal_id, state_features)
        if cache:
            self._cached_goal_embedding = embedding.detach()
        return embedding

    def compute_subgoal_embedding(
        self,
        subgoal_id: torch.Tensor,
        spatial_params: torch.Tensor,
        goal_embedding: Optional[torch.Tensor] = None,
        cache: bool = True
    ) -> torch.Tensor:
        """Compute and optionally cache subgoal embedding."""
        if goal_embedding is None:
            if self._cached_goal_embedding is None:
                raise ValueError("No goal embedding available")
            goal_embedding = self._cached_goal_embedding

        embedding = self.subgoal_embedder(subgoal_id, spatial_params, goal_embedding)
        if cache:
            self._cached_subgoal_embedding = embedding.detach()
        return embedding

    def get_goal_attended_features(
        self,
        goal_embedding: torch.Tensor,
        spatial_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get goal-attended spatial features."""
        return self.goal_attention(goal_embedding, spatial_features)

    def clear_cache(self):
        """Clear cached embeddings."""
        self._cached_goal_embedding = None
        self._cached_subgoal_embedding = None

    @property
    def cached_goal_embedding(self) -> Optional[torch.Tensor]:
        """Get cached goal embedding."""
        return self._cached_goal_embedding

    @property
    def cached_subgoal_embedding(self) -> Optional[torch.Tensor]:
        """Get cached subgoal embedding."""
        return self._cached_subgoal_embedding
