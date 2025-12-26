"""
Hierarchical Planning Architecture for ONI-AI.

Task 5.1: Hierarchical Planning Architecture

This module implements a 3-level hierarchical reinforcement learning
architecture for the Oxygen Not Included AI agent:

- High-Level Planner: Selects abstract goals (5 goal types)
- Mid-Level Controller: Decomposes goals into subgoals (15 subgoal types)
- Low-Level Executor: Maps subgoals to primitive actions (~200 actions)

Architecture:
    +---------------------------+
    |    HIGH-LEVEL PLANNER     |  Operates every 20 cycles
    |  - 5 abstract goals       |  Goal embedding: 64-dim
    +-------------+-------------+
                  |
        Goal Embedding (64-dim)
                  |
    +-------------v-------------+
    |   MID-LEVEL CONTROLLER    |  Operates every 5 steps
    |  - 15 subgoal types       |  Subgoal embedding: 32-dim
    +-------------+-------------+
                  |
       Subgoal Embedding (32-dim)
                  |
    +-------------v-------------+
    |   LOW-LEVEL EXECUTOR      |  Operates every step
    |  - ~200 primitive actions |  Uses existing action space
    +---------------------------+

Usage:
    from src.agents.hierarchical import (
        HierarchicalAgent,
        HierarchicalConfig,
        create_hierarchical_agent,
        HierarchicalPPOTrainer,
    )

    # Create agent with default config
    agent = create_hierarchical_agent()

    # Create trainer
    trainer = HierarchicalPPOTrainer(agent)

    # Select action in environment loop
    action, info = agent.select_action(
        observation,
        game_metrics={'oxygen_ratio': 0.2, 'cycle': 5},
        action_mask=valid_actions
    )
"""

# Configuration
from .config import (
    GoalConfig,
    TemporalConfig,
    IntrinsicRewardConfig,
    NetworkConfig,
    TrainingConfig,
    HierarchicalConfig,
)

# Goal types and termination conditions
from .goal_types import (
    AbstractGoal,
    SubgoalType,
    GoalTerminationCondition,
    SubgoalTerminationCondition,
    GOAL_TERMINATION_CONDITIONS,
    SUBGOAL_TERMINATION_CONDITIONS,
    get_goal_termination,
    get_subgoal_termination,
    goal_to_one_hot,
    subgoal_to_one_hot,
)

# Goal embeddings
from .goal_embeddings import (
    GoalEmbeddingNetwork,
    SubgoalEmbeddingNetwork,
    GoalConditionedAttention,
    HierarchicalEmbeddingSystem,
)

# Hierarchical components
from .high_level_planner import (
    HighLevelPlanner,
    create_high_level_planner,
)

from .mid_level_controller import (
    MidLevelController,
    MidLevelControllerSimple,
    create_mid_level_controller,
)

from .low_level_executor import (
    LowLevelExecutor,
    LowLevelExecutorSimple,
    SpatialFeatureExtractor,
    ResidualBlock,
    create_low_level_executor,
)

# Coordination
from .coordinator import (
    HierarchyState,
    HierarchicalCoordinator,
    HierarchyBuffer,
)

# Intrinsic rewards
from .intrinsic_rewards import (
    IntrinsicRewardBreakdown,
    HierarchicalIntrinsicRewards,
    SubgoalAchievementTracker,
    create_intrinsic_reward_calculator,
)

# Main agent
from .hierarchical_agent import (
    HierarchicalAgent,
    create_hierarchical_agent,
)

# Training
from .training import (
    TrainingState,
    HierarchicalPPOTrainer,
    create_trainer,
)


__all__ = [
    # Configuration
    'GoalConfig',
    'TemporalConfig',
    'IntrinsicRewardConfig',
    'NetworkConfig',
    'TrainingConfig',
    'HierarchicalConfig',

    # Goal types
    'AbstractGoal',
    'SubgoalType',
    'GoalTerminationCondition',
    'SubgoalTerminationCondition',
    'GOAL_TERMINATION_CONDITIONS',
    'SUBGOAL_TERMINATION_CONDITIONS',
    'get_goal_termination',
    'get_subgoal_termination',
    'goal_to_one_hot',
    'subgoal_to_one_hot',

    # Embeddings
    'GoalEmbeddingNetwork',
    'SubgoalEmbeddingNetwork',
    'GoalConditionedAttention',
    'HierarchicalEmbeddingSystem',

    # Hierarchical components
    'HighLevelPlanner',
    'create_high_level_planner',
    'MidLevelController',
    'MidLevelControllerSimple',
    'create_mid_level_controller',
    'LowLevelExecutor',
    'LowLevelExecutorSimple',
    'SpatialFeatureExtractor',
    'ResidualBlock',
    'create_low_level_executor',

    # Coordination
    'HierarchyState',
    'HierarchicalCoordinator',
    'HierarchyBuffer',

    # Intrinsic rewards
    'IntrinsicRewardBreakdown',
    'HierarchicalIntrinsicRewards',
    'SubgoalAchievementTracker',
    'create_intrinsic_reward_calculator',

    # Main agent
    'HierarchicalAgent',
    'create_hierarchical_agent',

    # Training
    'TrainingState',
    'HierarchicalPPOTrainer',
    'create_trainer',
]
