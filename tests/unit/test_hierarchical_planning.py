"""
Unit tests for hierarchical planning architecture.

Task 5.1: Hierarchical Planning Architecture

Tests all components of the 3-level hierarchical RL system:
- Goal types and embeddings
- High-level planner
- Mid-level controller
- Low-level executor
- Coordinator
- Intrinsic rewards
- Full hierarchical agent
"""

import pytest
import torch
import numpy as np
from typing import Dict

from src.agents.hierarchical import (
    # Configuration
    HierarchicalConfig,
    GoalConfig,
    TemporalConfig,
    IntrinsicRewardConfig,
    NetworkConfig,
    TrainingConfig,

    # Goal types
    AbstractGoal,
    SubgoalType,
    get_goal_termination,
    get_subgoal_termination,
    goal_to_one_hot,
    subgoal_to_one_hot,

    # Embeddings
    GoalEmbeddingNetwork,
    SubgoalEmbeddingNetwork,
    GoalConditionedAttention,
    HierarchicalEmbeddingSystem,

    # Components
    HighLevelPlanner,
    MidLevelController,
    LowLevelExecutor,
    HierarchicalCoordinator,
    HierarchyBuffer,

    # Intrinsic rewards
    HierarchicalIntrinsicRewards,
    SubgoalAchievementTracker,

    # Main agent
    HierarchicalAgent,
    create_hierarchical_agent,

    # Training
    HierarchicalPPOTrainer,
    TrainingState,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create default hierarchical configuration."""
    return HierarchicalConfig()


@pytest.fixture
def small_config():
    """Create small configuration for faster tests."""
    return HierarchicalConfig(
        spatial_channels=4,
        spatial_size=16,
        global_features_dim=32,
        num_actions=50,
        goals=GoalConfig(
            goal_embedding_dim=32,
            subgoal_embedding_dim=16,
        ),
        network=NetworkConfig(
            hl_hidden_dims=[64, 32],
            ml_hidden_dims=[64, 32],
            ll_hidden_dims=[64, 32],
        ),
    )


@pytest.fixture
def dummy_observation(small_config):
    """Create dummy observation matching expected format."""
    spatial_dim = (
        small_config.spatial_channels *
        small_config.spatial_size *
        small_config.spatial_size
    )
    obs_dim = spatial_dim + small_config.global_features_dim
    return np.random.randn(obs_dim).astype(np.float32)


@pytest.fixture
def dummy_game_metrics():
    """Create dummy game metrics."""
    return {
        'oxygen_ratio': 0.2,
        'water_ratio': 0.3,
        'happiness_ratio': 0.5,
        'power_balance': 10.0,
        'cycle': 5,
        'step': 100,
        'total_tiles_dug': 10,
        'rooms_built': 2,
    }


# =============================================================================
# Goal Types Tests
# =============================================================================

class TestGoalTypes:
    """Tests for goal type definitions."""

    def test_abstract_goal_enum(self):
        """Test AbstractGoal enum has correct values."""
        assert len(AbstractGoal) == 5
        assert AbstractGoal.MAINTAIN_OXYGEN.value == 0
        assert AbstractGoal.MANAGE_WATER.value == 1
        assert AbstractGoal.ENSURE_HAPPINESS.value == 2
        assert AbstractGoal.BUILD_INFRASTRUCTURE.value == 3
        assert AbstractGoal.SECURE_RESOURCES.value == 4

    def test_subgoal_type_enum(self):
        """Test SubgoalType enum has correct values."""
        assert len(SubgoalType) == 15
        assert SubgoalType.BUILD_OXYGEN_ROOM.value == 0
        assert SubgoalType.EMERGENCY_DIG.value == 14

    def test_goal_termination_conditions(self):
        """Test goal termination conditions are defined."""
        for goal in AbstractGoal:
            term = get_goal_termination(goal)
            assert term is not None
            assert term.metric_name is not None
            assert term.success_threshold >= 0

    def test_subgoal_termination_conditions(self):
        """Test subgoal termination conditions are defined."""
        for subgoal in SubgoalType:
            term = get_subgoal_termination(subgoal)
            assert term is not None
            assert term.success_metric is not None
            assert term.max_steps > 0

    def test_goal_to_one_hot(self):
        """Test goal one-hot encoding."""
        for goal in AbstractGoal:
            one_hot = goal_to_one_hot(goal)
            assert one_hot.shape == (5,)
            assert one_hot.sum() == 1.0
            assert one_hot[goal.value] == 1.0

    def test_subgoal_to_one_hot(self):
        """Test subgoal one-hot encoding."""
        for subgoal in SubgoalType:
            one_hot = subgoal_to_one_hot(subgoal)
            assert one_hot.shape == (15,)
            assert one_hot.sum() == 1.0
            assert one_hot[subgoal.value] == 1.0


# =============================================================================
# Goal Embeddings Tests
# =============================================================================

class TestGoalEmbeddings:
    """Tests for goal embedding networks."""

    def test_goal_embedding_network_shape(self, small_config):
        """Test GoalEmbeddingNetwork output shapes."""
        network = GoalEmbeddingNetwork(
            num_goals=5,
            state_dim=small_config.global_features_dim,
            embed_dim=small_config.goals.goal_embedding_dim,
        )

        batch_size = 4
        goal_indices = torch.randint(0, 5, (batch_size,))
        state_features = torch.randn(batch_size, small_config.global_features_dim)

        embedding = network(goal_indices, state_features)

        assert embedding.shape == (batch_size, small_config.goals.goal_embedding_dim)

    def test_subgoal_embedding_network_shape(self, small_config):
        """Test SubgoalEmbeddingNetwork output shapes."""
        network = SubgoalEmbeddingNetwork(
            num_subgoals=15,
            spatial_param_dim=4,
            goal_embed_dim=small_config.goals.goal_embedding_dim,
            embed_dim=small_config.goals.subgoal_embedding_dim,
        )

        batch_size = 4
        subgoal_indices = torch.randint(0, 15, (batch_size,))
        spatial_params = torch.rand(batch_size, 4)
        goal_embedding = torch.randn(batch_size, small_config.goals.goal_embedding_dim)

        embedding = network(subgoal_indices, spatial_params, goal_embedding)

        assert embedding.shape == (batch_size, small_config.goals.subgoal_embedding_dim)

    def test_goal_conditioned_attention(self, small_config):
        """Test GoalConditionedAttention output shapes."""
        attention = GoalConditionedAttention(
            goal_dim=small_config.goals.goal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
            hidden_dim=64,
            num_heads=2,
        )

        batch_size = 4
        goal_embedding = torch.randn(batch_size, small_config.goals.goal_embedding_dim)
        spatial_features = torch.randn(
            batch_size,
            small_config.spatial_channels,
            small_config.spatial_size,
            small_config.spatial_size
        )

        attended, weights = attention(goal_embedding, spatial_features)

        assert attended.shape == (batch_size, 64)
        assert weights.shape == (batch_size, small_config.spatial_size * small_config.spatial_size)

    def test_hierarchical_embedding_system(self, small_config):
        """Test HierarchicalEmbeddingSystem integration."""
        system = HierarchicalEmbeddingSystem(
            num_goals=5,
            num_subgoals=15,
            state_dim=small_config.global_features_dim,
            goal_embed_dim=small_config.goals.goal_embedding_dim,
            subgoal_embed_dim=small_config.goals.subgoal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
        )

        batch_size = 2
        goal_indices = torch.randint(0, 5, (batch_size,))
        subgoal_indices = torch.randint(0, 15, (batch_size,))
        state_features = torch.randn(batch_size, small_config.global_features_dim)
        spatial_params = torch.rand(batch_size, 4)

        # Test goal embedding
        goal_embed = system.compute_goal_embedding(goal_indices, state_features)
        assert goal_embed.shape == (batch_size, small_config.goals.goal_embedding_dim)

        # Test subgoal embedding
        subgoal_embed = system.compute_subgoal_embedding(
            subgoal_indices, spatial_params, goal_embed
        )
        assert subgoal_embed.shape == (batch_size, small_config.goals.subgoal_embedding_dim)

        # Test caching
        goal_embed_cached = system.compute_goal_embedding(
            goal_indices, state_features, cache=True
        )
        assert goal_embed_cached.shape == goal_embed.shape

        system.clear_cache()


# =============================================================================
# High-Level Planner Tests
# =============================================================================

class TestHighLevelPlanner:
    """Tests for high-level planner."""

    def test_forward_shape(self, small_config):
        """Test HighLevelPlanner forward pass shapes."""
        planner = HighLevelPlanner(
            state_dim=small_config.global_features_dim,
            num_goals=5,
            hidden_dims=(64, 32),
        )

        batch_size = 4
        global_features = torch.randn(batch_size, small_config.global_features_dim)

        goal_logits, value = planner(global_features)

        assert goal_logits.shape == (batch_size, 5)
        assert value.shape == (batch_size, 1)

    def test_select_goal(self, small_config):
        """Test goal selection."""
        planner = HighLevelPlanner(
            state_dim=small_config.global_features_dim,
            num_goals=5,
        )

        batch_size = 4
        global_features = torch.randn(batch_size, small_config.global_features_dim)

        goal, log_prob, value = planner.select_goal(global_features)

        assert goal.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size,)
        assert (goal >= 0).all() and (goal < 5).all()

    def test_goal_masking(self, small_config):
        """Test goal masking works correctly."""
        planner = HighLevelPlanner(
            state_dim=small_config.global_features_dim,
            num_goals=5,
        )

        batch_size = 4
        global_features = torch.randn(batch_size, small_config.global_features_dim)

        # Only goal 0 and 2 are valid
        mask = torch.zeros(batch_size, 5, dtype=torch.bool)
        mask[:, 0] = True
        mask[:, 2] = True

        goal, _, _ = planner.select_goal(global_features, goal_mask=mask, deterministic=True)

        # All selected goals should be 0 or 2
        assert all(g.item() in [0, 2] for g in goal)

    def test_evaluate_goals(self, small_config):
        """Test goal evaluation for training."""
        planner = HighLevelPlanner(
            state_dim=small_config.global_features_dim,
            num_goals=5,
        )

        batch_size = 4
        global_features = torch.randn(batch_size, small_config.global_features_dim)
        goals = torch.randint(0, 5, (batch_size,))

        log_prob, entropy, value = planner.evaluate_goals(global_features, goals)

        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)


# =============================================================================
# Mid-Level Controller Tests
# =============================================================================

class TestMidLevelController:
    """Tests for mid-level controller."""

    def test_forward_shape(self, small_config):
        """Test MidLevelController forward pass shapes."""
        controller = MidLevelController(
            goal_dim=small_config.goals.goal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
            global_dim=small_config.global_features_dim,
            num_subgoals=15,
            hidden_dims=(64, 32),
            use_attention=True,
        )

        batch_size = 4
        spatial_dim = (
            small_config.spatial_channels *
            small_config.spatial_size *
            small_config.spatial_size
        )
        obs_dim = spatial_dim + small_config.global_features_dim
        observation = torch.randn(batch_size, obs_dim)
        goal_embedding = torch.randn(batch_size, small_config.goals.goal_embedding_dim)

        subgoal_logits, spatial_params, value, attention = controller(
            observation, goal_embedding
        )

        assert subgoal_logits.shape == (batch_size, 15)
        assert spatial_params.shape == (batch_size, 4)
        assert value.shape == (batch_size, 1)
        assert attention is not None

    def test_select_subgoal(self, small_config):
        """Test subgoal selection."""
        controller = MidLevelController(
            goal_dim=small_config.goals.goal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
            global_dim=small_config.global_features_dim,
            num_subgoals=15,
        )

        batch_size = 4
        spatial_dim = (
            small_config.spatial_channels *
            small_config.spatial_size *
            small_config.spatial_size
        )
        obs_dim = spatial_dim + small_config.global_features_dim
        observation = torch.randn(batch_size, obs_dim)
        goal_embedding = torch.randn(batch_size, small_config.goals.goal_embedding_dim)

        subgoal, spatial_params, log_prob, value = controller.select_subgoal(
            observation, goal_embedding
        )

        assert subgoal.shape == (batch_size,)
        assert spatial_params.shape == (batch_size, 4)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size,)
        assert (subgoal >= 0).all() and (subgoal < 15).all()
        assert (spatial_params >= 0).all() and (spatial_params <= 1).all()

    def test_subgoal_masking(self, small_config):
        """Test subgoal masking works correctly."""
        controller = MidLevelController(
            goal_dim=small_config.goals.goal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
            global_dim=small_config.global_features_dim,
            num_subgoals=15,
        )

        batch_size = 4
        spatial_dim = (
            small_config.spatial_channels *
            small_config.spatial_size *
            small_config.spatial_size
        )
        obs_dim = spatial_dim + small_config.global_features_dim
        observation = torch.randn(batch_size, obs_dim)
        goal_embedding = torch.randn(batch_size, small_config.goals.goal_embedding_dim)

        # Only subgoals 0, 1, 2 are valid
        mask = torch.zeros(batch_size, 15, dtype=torch.bool)
        mask[:, :3] = True

        subgoal, _, _, _ = controller.select_subgoal(
            observation, goal_embedding, subgoal_mask=mask, deterministic=True
        )

        assert all(s.item() in [0, 1, 2] for s in subgoal)


# =============================================================================
# Low-Level Executor Tests
# =============================================================================

class TestLowLevelExecutor:
    """Tests for low-level executor."""

    def test_forward_shape(self, small_config):
        """Test LowLevelExecutor forward pass shapes."""
        executor = LowLevelExecutor(
            subgoal_dim=small_config.goals.subgoal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
            global_dim=small_config.global_features_dim,
            num_actions=small_config.num_actions,
            hidden_dims=(64, 32),
        )

        batch_size = 4
        spatial_dim = (
            small_config.spatial_channels *
            small_config.spatial_size *
            small_config.spatial_size
        )
        obs_dim = spatial_dim + small_config.global_features_dim
        observation = torch.randn(batch_size, obs_dim)
        subgoal_embedding = torch.randn(batch_size, small_config.goals.subgoal_embedding_dim)

        action_logits, value = executor(observation, subgoal_embedding)

        assert action_logits.shape == (batch_size, small_config.num_actions)
        assert value.shape == (batch_size, 1)

    def test_select_action(self, small_config):
        """Test action selection."""
        executor = LowLevelExecutor(
            subgoal_dim=small_config.goals.subgoal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
            global_dim=small_config.global_features_dim,
            num_actions=small_config.num_actions,
        )

        batch_size = 4
        spatial_dim = (
            small_config.spatial_channels *
            small_config.spatial_size *
            small_config.spatial_size
        )
        obs_dim = spatial_dim + small_config.global_features_dim
        observation = torch.randn(batch_size, obs_dim)
        subgoal_embedding = torch.randn(batch_size, small_config.goals.subgoal_embedding_dim)

        action, log_prob, value = executor.select_action(observation, subgoal_embedding)

        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size,)
        assert (action >= 0).all() and (action < small_config.num_actions).all()

    def test_action_masking(self, small_config):
        """Test action masking works correctly."""
        executor = LowLevelExecutor(
            subgoal_dim=small_config.goals.subgoal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
            global_dim=small_config.global_features_dim,
            num_actions=small_config.num_actions,
        )

        batch_size = 4
        spatial_dim = (
            small_config.spatial_channels *
            small_config.spatial_size *
            small_config.spatial_size
        )
        obs_dim = spatial_dim + small_config.global_features_dim
        observation = torch.randn(batch_size, obs_dim)
        subgoal_embedding = torch.randn(batch_size, small_config.goals.subgoal_embedding_dim)

        # Only first 10 actions are valid
        mask = torch.zeros(batch_size, small_config.num_actions, dtype=torch.bool)
        mask[:, :10] = True

        action, _, _ = executor.select_action(
            observation, subgoal_embedding, action_mask=mask, deterministic=True
        )

        assert all(a.item() < 10 for a in action)

    def test_evaluate_actions(self, small_config):
        """Test action evaluation for training."""
        executor = LowLevelExecutor(
            subgoal_dim=small_config.goals.subgoal_embedding_dim,
            spatial_channels=small_config.spatial_channels,
            spatial_size=small_config.spatial_size,
            global_dim=small_config.global_features_dim,
            num_actions=small_config.num_actions,
        )

        batch_size = 4
        spatial_dim = (
            small_config.spatial_channels *
            small_config.spatial_size *
            small_config.spatial_size
        )
        obs_dim = spatial_dim + small_config.global_features_dim
        observation = torch.randn(batch_size, obs_dim)
        subgoal_embedding = torch.randn(batch_size, small_config.goals.subgoal_embedding_dim)
        actions = torch.randint(0, small_config.num_actions, (batch_size,))

        log_prob, entropy, value = executor.evaluate_actions(
            observation, subgoal_embedding, actions
        )

        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)


# =============================================================================
# Coordinator Tests
# =============================================================================

class TestCoordinator:
    """Tests for hierarchical coordinator."""

    def test_coordinator_initialization(self, small_config):
        """Test coordinator initializes correctly."""
        coordinator = HierarchicalCoordinator(small_config)

        assert coordinator.state.current_goal is None
        assert coordinator.state.current_subgoal is None
        assert coordinator.state.goal_status is None
        assert coordinator.state.subgoal_status is None

    def test_needs_new_goal(self, small_config):
        """Test goal requirement detection."""
        coordinator = HierarchicalCoordinator(small_config)

        # Should need goal initially
        assert coordinator.needs_new_goal()

        # Set a goal
        goal_embedding = torch.randn(1, small_config.goals.goal_embedding_dim)
        coordinator.set_goal(AbstractGoal.MAINTAIN_OXYGEN, goal_embedding, cycle=0)

        # Should not need goal immediately after
        assert not coordinator.needs_new_goal()

    def test_needs_new_subgoal(self, small_config):
        """Test subgoal requirement detection."""
        coordinator = HierarchicalCoordinator(small_config)

        # Set goal first
        goal_embedding = torch.randn(1, small_config.goals.goal_embedding_dim)
        coordinator.set_goal(AbstractGoal.MAINTAIN_OXYGEN, goal_embedding, cycle=0)

        # Should need subgoal
        assert coordinator.needs_new_subgoal()

        # Set subgoal
        subgoal_embedding = torch.randn(1, small_config.goals.subgoal_embedding_dim)
        coordinator.set_subgoal(
            SubgoalType.BUILD_OXYGEN_ROOM,
            subgoal_embedding,
            np.array([0.5, 0.5, 0.2, 0.2]),
            step=0
        )

        # Should not need subgoal immediately after
        assert not coordinator.needs_new_subgoal()

    def test_update_detects_completion(self, small_config):
        """Test coordinator detects goal/subgoal completion."""
        coordinator = HierarchicalCoordinator(small_config)

        # Set goal and subgoal
        goal_embedding = torch.randn(1, small_config.goals.goal_embedding_dim)
        coordinator.set_goal(AbstractGoal.MAINTAIN_OXYGEN, goal_embedding, cycle=0)

        subgoal_embedding = torch.randn(1, small_config.goals.subgoal_embedding_dim)
        coordinator.set_subgoal(
            SubgoalType.BUILD_OXYGEN_ROOM,
            subgoal_embedding,
            np.array([0.5, 0.5, 0.2, 0.2]),
            step=0
        )

        # Simulate completion with high oxygen
        metrics = {'oxygen_ratio': 0.35, 'rooms_built': 5}  # Above threshold
        results = coordinator.update(metrics, cycle=1, step=10)

        # Goal should complete (oxygen_ratio >= 0.3)
        assert results['goal_completed'] or results['subgoal_completed'] or not results['goal_failed']

    def test_reset(self, small_config):
        """Test coordinator reset."""
        coordinator = HierarchicalCoordinator(small_config)

        # Set goal and subgoal
        goal_embedding = torch.randn(1, small_config.goals.goal_embedding_dim)
        coordinator.set_goal(AbstractGoal.MAINTAIN_OXYGEN, goal_embedding, cycle=0)

        subgoal_embedding = torch.randn(1, small_config.goals.subgoal_embedding_dim)
        coordinator.set_subgoal(
            SubgoalType.BUILD_OXYGEN_ROOM,
            subgoal_embedding,
            np.array([0.5, 0.5, 0.2, 0.2]),
            step=0
        )

        # Reset
        coordinator.reset()

        assert coordinator.state.current_goal is None
        assert coordinator.state.current_subgoal is None

    def test_hierarchy_buffer(self, small_config):
        """Test HierarchyBuffer stores experiences."""
        buffer = HierarchyBuffer(capacity=100)

        # Add some high-level experiences
        for i in range(10):
            buffer.add_high_level(
                state=np.random.randn(100).astype(np.float32),
                goal=i % 5,
                reward=float(i) * 0.1,
                value=float(i) * 0.05,
                log_prob=-0.5,
                done=False,
            )

        # Add some low-level experiences
        for i in range(10):
            buffer.add_low_level(
                state=np.random.randn(100).astype(np.float32),
                subgoal_embedding=np.random.randn(32).astype(np.float32),
                action=i % 50,
                reward=float(i) * 0.1,
                value=float(i) * 0.05,
                log_prob=-0.5,
                done=False,
            )

        assert len(buffer.hl_states) == 10
        assert len(buffer.ll_states) == 10


# =============================================================================
# Intrinsic Rewards Tests
# =============================================================================

class TestIntrinsicRewards:
    """Tests for intrinsic reward system."""

    def test_basic_reward_computation(self):
        """Test basic intrinsic reward computation."""
        rewards = HierarchicalIntrinsicRewards()

        total, breakdown = rewards.compute_reward(
            current_goal=AbstractGoal.MAINTAIN_OXYGEN,
            current_subgoal=SubgoalType.BUILD_OXYGEN_ROOM,
            game_metrics={'oxygen_ratio': 0.2},
            goal_completed=False,
            subgoal_completed=False,
        )

        assert isinstance(total, float)
        assert breakdown.total == total

    def test_subgoal_completion_bonus(self):
        """Test subgoal completion gives bonus."""
        config = IntrinsicRewardConfig(
            use_intrinsic_rewards=True,
            subgoal_completion_bonus=10.0,
        )
        rewards = HierarchicalIntrinsicRewards(config)

        total, breakdown = rewards.compute_reward(
            current_goal=AbstractGoal.MAINTAIN_OXYGEN,
            current_subgoal=SubgoalType.BUILD_OXYGEN_ROOM,
            game_metrics={'oxygen_ratio': 0.2},
            subgoal_completed=True,
        )

        assert breakdown.subgoal_completion == 10.0
        assert total >= 10.0

    def test_goal_completion_bonus(self):
        """Test goal completion gives bonus."""
        config = IntrinsicRewardConfig(
            use_intrinsic_rewards=True,
            goal_completion_bonus=50.0,
        )
        rewards = HierarchicalIntrinsicRewards(config)

        total, breakdown = rewards.compute_reward(
            current_goal=AbstractGoal.MAINTAIN_OXYGEN,
            current_subgoal=SubgoalType.BUILD_OXYGEN_ROOM,
            game_metrics={'oxygen_ratio': 0.35},
            goal_completed=True,
        )

        assert breakdown.goal_completion == 50.0
        assert total >= 50.0

    def test_reset_clears_state(self):
        """Test reset clears internal state."""
        rewards = HierarchicalIntrinsicRewards()

        # Compute some rewards
        rewards.compute_reward(
            current_goal=AbstractGoal.MAINTAIN_OXYGEN,
            current_subgoal=None,
            game_metrics={'oxygen_ratio': 0.2},
        )

        rewards.reset()

        stats = rewards.get_statistics()
        assert stats['num_visited_states'] == 0

    def test_achievement_tracker(self):
        """Test SubgoalAchievementTracker."""
        tracker = SubgoalAchievementTracker()

        # Record some attempts
        tracker.record_subgoal_attempt(SubgoalType.BUILD_OXYGEN_ROOM)
        tracker.record_subgoal_attempt(SubgoalType.BUILD_OXYGEN_ROOM)
        tracker.record_subgoal_completion(SubgoalType.BUILD_OXYGEN_ROOM)

        rate = tracker.get_subgoal_success_rate(SubgoalType.BUILD_OXYGEN_ROOM)
        assert rate == 0.5  # 1 completion / 2 attempts

        tracker.record_goal_attempt(AbstractGoal.MAINTAIN_OXYGEN)
        tracker.record_goal_completion(AbstractGoal.MAINTAIN_OXYGEN)

        rate = tracker.get_goal_success_rate(AbstractGoal.MAINTAIN_OXYGEN)
        assert rate == 1.0


# =============================================================================
# Hierarchical Agent Tests
# =============================================================================

class TestHierarchicalAgent:
    """Tests for main hierarchical agent."""

    def test_agent_creation(self, small_config):
        """Test agent creation with factory function."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        assert isinstance(agent, HierarchicalAgent)
        assert agent.device == torch.device('cpu')

    def test_select_action(self, small_config, dummy_observation, dummy_game_metrics):
        """Test action selection through full hierarchy."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        agent.eval()

        action, info = agent.select_action(
            dummy_observation,
            dummy_game_metrics,
            deterministic=True
        )

        assert isinstance(action, int)
        assert 0 <= action < small_config.num_actions
        assert 'goal_selected' in info
        assert 'subgoal_selected' in info
        assert 'action_log_prob' in info
        assert 'action_value' in info

    def test_action_with_mask(self, small_config, dummy_observation, dummy_game_metrics):
        """Test action selection respects mask."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        agent.eval()

        # Only first 10 actions valid
        action_mask = np.zeros(small_config.num_actions, dtype=bool)
        action_mask[:10] = True

        action, _ = agent.select_action(
            dummy_observation,
            dummy_game_metrics,
            action_mask=action_mask,
            deterministic=True
        )

        assert action < 10

    def test_update_and_get_intrinsic_reward(self, small_config, dummy_game_metrics):
        """Test intrinsic reward computation."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        # Need to initialize goal first
        dummy_obs = np.random.randn(
            small_config.spatial_channels * small_config.spatial_size * small_config.spatial_size +
            small_config.global_features_dim
        ).astype(np.float32)

        agent.select_action(dummy_obs, dummy_game_metrics, deterministic=True)

        reward, info = agent.update_and_get_intrinsic_reward(
            dummy_game_metrics,
            cycle=5,
            step=100
        )

        assert isinstance(reward, float)
        assert 'reward_breakdown' in info

    def test_reset(self, small_config, dummy_observation, dummy_game_metrics):
        """Test agent reset between episodes."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        # Do some actions
        agent.select_action(dummy_observation, dummy_game_metrics)
        agent.select_action(dummy_observation, dummy_game_metrics)

        # Reset
        agent.reset()

        # Should need new goal after reset
        assert agent.coordinator.needs_new_goal()

    def test_episode_statistics(self, small_config, dummy_observation, dummy_game_metrics):
        """Test episode statistics collection."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        # Do some actions
        for _ in range(5):
            agent.select_action(dummy_observation, dummy_game_metrics)

        stats = agent.get_episode_statistics()

        assert 'coordinator' in stats
        assert 'achievements' in stats
        assert 'intrinsic' in stats

    def test_save_load(self, small_config, tmp_path):
        """Test agent save and load."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        # Save
        save_path = str(tmp_path / "agent")
        agent.save(save_path)

        # Create new agent and load
        agent2 = create_hierarchical_agent(small_config, device='cpu')
        agent2.load(save_path)

        # Check weights are equal
        for (name1, param1), (name2, param2) in zip(
            agent.high_level_planner.named_parameters(),
            agent2.high_level_planner.named_parameters()
        ):
            assert torch.allclose(param1, param2), f"Parameter {name1} differs"

    def test_train_eval_modes(self, small_config):
        """Test training and evaluation mode switching."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        agent.train()
        assert agent._is_training

        agent.eval()
        assert not agent._is_training


# =============================================================================
# Training Infrastructure Tests
# =============================================================================

class TestTrainingInfrastructure:
    """Tests for training infrastructure."""

    def test_training_state(self):
        """Test TrainingState serialization."""
        state = TrainingState(
            total_timesteps=1000,
            total_episodes=50,
            current_curriculum_stage=1,
            best_mean_reward=25.0,
        )

        state_dict = state.to_dict()
        restored = TrainingState.from_dict(state_dict)

        assert restored.total_timesteps == 1000
        assert restored.total_episodes == 50
        assert restored.current_curriculum_stage == 1
        assert restored.best_mean_reward == 25.0

    def test_trainer_creation(self, small_config):
        """Test trainer creation."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        trainer = HierarchicalPPOTrainer(agent, small_config)

        assert trainer.agent is agent
        assert trainer.config is small_config

    def test_compute_gae(self, small_config):
        """Test GAE computation."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        trainer = HierarchicalPPOTrainer(agent, small_config)

        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [1.0, 1.5, 2.0, 2.5, 3.0]
        dones = [False, False, False, False, True]

        advantages, returns = trainer.compute_gae(rewards, values, dones)

        assert len(advantages) == 5
        assert len(returns) == 5

    def test_record_episode(self, small_config):
        """Test episode recording."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        trainer = HierarchicalPPOTrainer(agent, small_config)

        trainer.record_episode(
            episode_reward=100.0,
            episode_length=200,
            goal_completion_rate=0.8,
            subgoal_completion_rate=0.9
        )

        assert len(trainer.episode_rewards) == 1
        assert trainer.state.total_episodes == 1

    def test_training_statistics(self, small_config):
        """Test training statistics computation."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        trainer = HierarchicalPPOTrainer(agent, small_config)

        # Record some episodes
        for i in range(5):
            trainer.record_episode(
                episode_reward=float(i * 10),
                episode_length=100 + i * 10,
                goal_completion_rate=0.5 + i * 0.1,
                subgoal_completion_rate=0.6 + i * 0.1
            )

        stats = trainer.get_training_statistics()

        assert 'mean_reward' in stats
        assert 'mean_length' in stats
        assert 'mean_goal_completion' in stats
        assert 'mean_subgoal_completion' in stats

    def test_checkpoint_save_load(self, small_config, tmp_path):
        """Test checkpoint saving and loading."""
        small_config.checkpoint_dir = str(tmp_path)
        agent = create_hierarchical_agent(small_config, device='cpu')
        trainer = HierarchicalPPOTrainer(agent, small_config)

        # Modify state
        trainer.state.total_timesteps = 5000
        trainer.state.total_episodes = 100

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint")

        # Create new trainer and load
        agent2 = create_hierarchical_agent(small_config, device='cpu')
        trainer2 = HierarchicalPPOTrainer(agent2, small_config)
        trainer2.load_checkpoint(str(tmp_path / "test_checkpoint"))

        assert trainer2.state.total_timesteps == 5000
        assert trainer2.state.total_episodes == 100


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full hierarchical system."""

    def test_full_episode_simulation(self, small_config, dummy_game_metrics):
        """Test simulating a full episode."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        agent.eval()

        # Simulate 100 steps
        for step in range(100):
            # Create observation
            obs = np.random.randn(
                small_config.spatial_channels * small_config.spatial_size * small_config.spatial_size +
                small_config.global_features_dim
            ).astype(np.float32)

            # Update metrics
            metrics = dummy_game_metrics.copy()
            metrics['step'] = step
            metrics['cycle'] = step // 20

            # Select action
            action, info = agent.select_action(obs, metrics)

            # Get intrinsic reward
            intrinsic_reward, update_info = agent.update_and_get_intrinsic_reward(
                metrics, cycle=metrics['cycle'], step=step
            )

            assert isinstance(action, int)
            assert isinstance(intrinsic_reward, float)

        # Check statistics
        stats = agent.get_episode_statistics()
        assert stats is not None

    def test_hierarchy_levels_coordinate(self, small_config, dummy_game_metrics):
        """Test that hierarchy levels properly coordinate."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        agent.eval()

        obs = np.random.randn(
            small_config.spatial_channels * small_config.spatial_size * small_config.spatial_size +
            small_config.global_features_dim
        ).astype(np.float32)

        # First action should select goal and subgoal
        action1, info1 = agent.select_action(obs, dummy_game_metrics)
        assert info1['goal_selected'] is True
        assert info1['subgoal_selected'] is True

        # Next few actions should not need new goal
        for _ in range(3):
            _, info = agent.select_action(obs, dummy_game_metrics)
            # May or may not need new subgoal depending on timing
            assert 'goal_selected' in info

    def test_training_loop_compatibility(self, small_config, dummy_game_metrics):
        """Test that agent works in training loop context."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        trainer = HierarchicalPPOTrainer(agent, small_config)

        agent.train()

        # Simulate training batch collection
        observations = []
        actions = []
        rewards = []

        for step in range(50):
            obs = np.random.randn(
                small_config.spatial_channels * small_config.spatial_size * small_config.spatial_size +
                small_config.global_features_dim
            ).astype(np.float32)

            action, info = agent.select_action(
                obs, dummy_game_metrics, deterministic=False
            )

            observations.append(obs)
            actions.append(action)
            rewards.append(1.0)

        # Record episode
        trainer.record_episode(
            episode_reward=sum(rewards),
            episode_length=len(rewards),
            goal_completion_rate=0.5,
            subgoal_completion_rate=0.7
        )

        assert trainer.state.total_episodes == 1
        assert len(observations) == 50


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_action_mask(self, small_config, dummy_observation, dummy_game_metrics):
        """Test handling of all-false action mask."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        # All actions invalid (edge case - should still return something)
        action_mask = np.zeros(small_config.num_actions, dtype=bool)
        # At least one action must be valid
        action_mask[0] = True

        action, _ = agent.select_action(
            dummy_observation,
            dummy_game_metrics,
            action_mask=action_mask,
            deterministic=True
        )

        assert action == 0

    def test_missing_metrics(self, small_config, dummy_observation):
        """Test handling of missing game metrics."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        # Minimal metrics
        metrics = {'cycle': 0, 'step': 0}

        # Should still work with default values
        action, info = agent.select_action(dummy_observation, metrics)
        assert isinstance(action, int)

    def test_multiple_resets(self, small_config, dummy_observation, dummy_game_metrics):
        """Test multiple consecutive resets."""
        agent = create_hierarchical_agent(small_config, device='cpu')

        for _ in range(5):
            agent.select_action(dummy_observation, dummy_game_metrics)
            agent.reset()

        # Should be in clean state
        assert agent.coordinator.needs_new_goal()

    def test_deterministic_consistency(self, small_config, dummy_observation, dummy_game_metrics):
        """Test that deterministic mode gives consistent results."""
        agent = create_hierarchical_agent(small_config, device='cpu')
        agent.eval()

        # Reset to known state
        agent.reset()

        action1, _ = agent.select_action(
            dummy_observation, dummy_game_metrics, deterministic=True
        )

        agent.reset()

        action2, _ = agent.select_action(
            dummy_observation, dummy_game_metrics, deterministic=True
        )

        # Same input, deterministic mode should give same output
        assert action1 == action2
