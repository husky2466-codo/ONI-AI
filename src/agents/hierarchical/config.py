"""
Configuration for hierarchical planning architecture.

Task 5.1: Hierarchical Planning Architecture
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os


@dataclass
class GoalConfig:
    """Configuration for goal system."""
    num_abstract_goals: int = 5
    num_subgoals: int = 15
    goal_embedding_dim: int = 64
    subgoal_embedding_dim: int = 32

    # Goal-specific settings
    goal_hidden_dim: int = 256
    subgoal_hidden_dim: int = 128


@dataclass
class TemporalConfig:
    """Configuration for temporal abstraction."""
    # How often each level makes decisions
    high_level_interval: int = 20  # game cycles
    mid_level_interval: int = 5    # environment steps

    # Maximum duration before forced termination
    max_goal_duration: int = 50    # cycles
    max_subgoal_duration: int = 20  # steps

    # Minimum duration before allowing termination
    min_goal_duration: int = 5     # cycles
    min_subgoal_duration: int = 3  # steps


@dataclass
class IntrinsicRewardConfig:
    """Configuration for intrinsic motivation."""
    use_intrinsic_rewards: bool = True

    # Bonus for completing subgoals/goals
    subgoal_completion_bonus: float = 10.0
    goal_completion_bonus: float = 50.0

    # Progress-based shaping
    use_progress_shaping: bool = True
    progress_shaping_scale: float = 5.0

    # Exploration bonus
    use_exploration_bonus: bool = False
    exploration_bonus_scale: float = 0.1


@dataclass
class NetworkConfig:
    """Configuration for neural network architectures."""
    # High-level planner
    hl_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    hl_activation: str = 'relu'

    # Mid-level controller
    ml_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    ml_use_attention: bool = True
    ml_attention_heads: int = 4

    # Low-level executor
    ll_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    ll_use_residual: bool = True

    # Shared settings
    dropout_rate: float = 0.1
    use_layer_norm: bool = True


@dataclass
class TrainingConfig:
    """Configuration for hierarchical training."""
    # Learning rates for each level
    hl_learning_rate: float = 1e-4
    ml_learning_rate: float = 3e-4
    ll_learning_rate: float = 3e-4

    # PPO hyperparameters (can differ by level)
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # Batch sizes
    batch_size: int = 256
    n_steps: int = 2048

    # Update frequencies (relative to low-level)
    hl_update_interval: int = 20
    ml_update_interval: int = 5

    # Total training steps
    total_timesteps: int = 10_000_000

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[int] = field(default_factory=lambda: [
        1000,   # Stage 1: Single goal
        3000,   # Stage 2: Infrastructure goals
        5000,   # Stage 3: Full goal set, short horizons
    ])


@dataclass
class HierarchicalConfig:
    """
    Master configuration for hierarchical planning architecture.

    Combines all sub-configurations for easy management.
    """
    # Sub-configurations
    goals: GoalConfig = field(default_factory=GoalConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    intrinsic: IntrinsicRewardConfig = field(default_factory=IntrinsicRewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # State dimensions (from Mini-ONI environment)
    spatial_channels: int = 8
    spatial_size: int = 32
    global_features_dim: int = 64
    num_actions: int = 200

    # Paths
    checkpoint_dir: str = 'checkpoints/hierarchical'
    log_dir: str = 'logs/hierarchical'
    tensorboard_log: str = 'runs/hierarchical'

    # Misc
    device: str = 'auto'  # 'cpu', 'cuda', or 'auto'
    seed: Optional[int] = None
    verbose: int = 1

    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.checkpoint_dir, self.log_dir, self.tensorboard_log]:
            os.makedirs(path, exist_ok=True)

    @property
    def observation_dim(self) -> int:
        """Total observation dimension."""
        return (self.spatial_channels * self.spatial_size * self.spatial_size +
                self.global_features_dim)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'goals': {
                'num_abstract_goals': self.goals.num_abstract_goals,
                'num_subgoals': self.goals.num_subgoals,
                'goal_embedding_dim': self.goals.goal_embedding_dim,
                'subgoal_embedding_dim': self.goals.subgoal_embedding_dim,
            },
            'temporal': {
                'high_level_interval': self.temporal.high_level_interval,
                'mid_level_interval': self.temporal.mid_level_interval,
                'max_goal_duration': self.temporal.max_goal_duration,
                'max_subgoal_duration': self.temporal.max_subgoal_duration,
            },
            'intrinsic': {
                'use_intrinsic_rewards': self.intrinsic.use_intrinsic_rewards,
                'subgoal_completion_bonus': self.intrinsic.subgoal_completion_bonus,
                'goal_completion_bonus': self.intrinsic.goal_completion_bonus,
            },
            'training': {
                'total_timesteps': self.training.total_timesteps,
                'batch_size': self.training.batch_size,
            },
            'num_actions': self.num_actions,
            'observation_dim': self.observation_dim,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HierarchicalConfig':
        """Create configuration from dictionary."""
        goals = GoalConfig(**config_dict.get('goals', {}))
        temporal = TemporalConfig(**config_dict.get('temporal', {}))
        intrinsic = IntrinsicRewardConfig(**config_dict.get('intrinsic', {}))
        network = NetworkConfig(**config_dict.get('network', {}))
        training = TrainingConfig(**config_dict.get('training', {}))

        return cls(
            goals=goals,
            temporal=temporal,
            intrinsic=intrinsic,
            network=network,
            training=training,
            num_actions=config_dict.get('num_actions', 200),
            seed=config_dict.get('seed'),
            verbose=config_dict.get('verbose', 1),
        )


def create_default_config() -> HierarchicalConfig:
    """Create default hierarchical configuration."""
    return HierarchicalConfig()


def create_fast_config() -> HierarchicalConfig:
    """Create configuration for fast testing/debugging."""
    return HierarchicalConfig(
        training=TrainingConfig(
            total_timesteps=10_000,
            batch_size=64,
            n_steps=256,
        ),
        temporal=TemporalConfig(
            high_level_interval=10,
            mid_level_interval=3,
        ),
    )
