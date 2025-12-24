# Mini-ONI Environment Implementation

## Overview

The Mini-ONI Environment is a simplified version of Oxygen Not Included designed for reinforcement learning training. It implements the core requirements from Task 2.1 of the ONI AI Agent specification.

## Implementation Status

✅ **COMPLETED** - All sub-tasks of Task 2.1 have been implemented:

1. ✅ Environment scope (64x64 tile maximum)
2. ✅ Rectangular starter base constraints  
3. ✅ Essential building type restrictions (15 types)
4. ✅ Time horizon limits (100 cycles maximum)
5. ✅ Environment reset and initialization

## Architecture

### Core Components

#### 1. MiniONIEnvironment (`src/environments/mini_oni/environment.py`)
- Main environment class implementing the RL interface
- Enforces 64x64 tile maximum and 100 cycle limits
- Provides standard methods: `reset()`, `step()`, `render()`
- Manages game simulation and reward calculation

#### 2. Game State (`src/environments/mini_oni/game_state.py`)
- `GameState`: Complete world state representation
- `Tile`: Individual grid cell with material, temperature, buildings
- `Building`: Building instances with type, position, operational status
- `Duplicant`: Colonist with position, happiness, stress, skills
- `Resources`: Colony resource tracking (oxygen, food, water, power)

#### 3. Building Types (`src/environments/mini_oni/building_types.py`)
- `BuildingType`: Enum of 15 essential building types
- `BuildingProperties`: Configuration for each building type
- Categories: life_support, infrastructure, utilities, food, power, water, ventilation

#### 4. Actions (`src/environments/mini_oni/actions.py`)
- `Action`: Base action class with validation
- `PlaceBuildingAction`: Place buildings in regions
- `DigAction`: Excavate solid tiles
- `PriorityAction`: Set task priorities
- `DuplicantAction`: Assign duplicant skills
- `NoOpAction`: Do nothing action

## Key Features

### Environment Constraints
- **Map Size**: Maximum 64x64 tiles (configurable, clamped to limit)
- **Episode Length**: Maximum 100 cycles (configurable, clamped to limit)
- **Building Types**: Limited to 15 essential types only
- **Action Space**: ~200 discrete actions (manageable for RL)

### Starter Base
- **Rectangular Area**: Configurable size (default 16x12 tiles)
- **Centered Placement**: Positioned in middle of map
- **Pre-cleared**: Gas/air tiles with breathable oxygen
- **Foundation**: Solid base for building placement
- **Starting Buildings**: Cot, outhouse, manual generator

### Objectives & Rewards
- **Dense Rewards**: +0.1 per breathable tile, +0.05 per happy duplicant
- **Penalties**: -0.1 per dangerous temperature tile, -50 per dead duplicant
- **Sparse Rewards**: +100 for 100-cycle survival, +20 for infrastructure milestones
- **Success Metrics**: Survival rate, oxygen coverage, duplicant happiness

### State Representation
- **Spatial**: 32x32x8 downscaled multi-channel tensor
  - Channels: material states (3), element ID (1), temperature (1), buildings (1), duplicants (1), oxygen (1)
- **Global**: 64-dimensional feature vector
  - Basic stats, resources, building counts, duplicant stats, task priorities
- **Combined**: Flattened observation (8,256 dimensions)

### Action Space
- **High-Level**: Coarse commands avoiding micromanagement
- **Building Placement**: Place specific building types in regions
- **Digging**: Excavate rectangular areas
- **Priority Setting**: Adjust task priorities (1-9 scale)
- **Duplicant Assignment**: Assign skills to duplicants
- **Action Masking**: Invalid actions filtered out

## Usage Example

```python
from src.environments.mini_oni.environment import MiniONIEnvironment

# Create environment
env = MiniONIEnvironment(
    map_width=32,
    map_height=32,
    max_cycles=50,
    num_duplicants=3
)

# Reset and run
obs = env.reset()
done = False

while not done:
    # Get valid actions
    action_mask = env.get_action_mask()
    valid_actions = np.where(action_mask)[0]
    
    # Choose action
    action_idx = np.random.choice(valid_actions)
    
    # Step environment
    obs, reward, done, info = env.step(action_idx)
    
    # Render state
    env.render(mode='human')
```

## Testing

Comprehensive unit tests verify all functionality:
- Environment initialization and constraints
- Reset and step functionality
- Action space generation and validation
- Building placement and digging
- Episode termination conditions
- Observation shape and values
- Rendering capabilities
- Resource tracking

Run tests with: `python -m pytest tests/unit/test_mini_oni_environment.py -v`

## Files Created

### Core Implementation
- `src/environments/__init__.py` - Environment module exports
- `src/environments/mini_oni/__init__.py` - Mini-ONI module exports
- `src/environments/mini_oni/environment.py` - Main environment class
- `src/environments/mini_oni/game_state.py` - Game state representation
- `src/environments/mini_oni/building_types.py` - Building definitions
- `src/environments/mini_oni/actions.py` - Action space definition

### Testing & Examples
- `tests/unit/test_mini_oni_environment.py` - Comprehensive unit tests
- `examples/mini_oni_environment_demo.py` - Usage demonstration
- `docs/mini_oni_environment_implementation.md` - This documentation

## Requirements Validation

### R2.1.1: Limit map area to rectangular starter base (64x64 tiles)
✅ **IMPLEMENTED**: Environment enforces 64x64 maximum, creates rectangular starter base

### R2.1.2: Restrict available buildings to essential subset (10-15 types)  
✅ **IMPLEMENTED**: Exactly 15 essential building types defined and enforced

### R2.1.3: Set clear episode termination conditions
✅ **IMPLEMENTED**: Terminates on cycle limit, all duplicants dead, or critical oxygen shortage

### R2.1.4: Define success/failure criteria for training
✅ **IMPLEMENTED**: Success score based on survival, oxygen, happiness, and time progression

## Next Steps

With Task 2.1 complete, the next tasks in the implementation plan are:

- **Task 2.2**: Objective System - Implement specific objectives and scoring
- **Task 2.3**: Action Space Design - Refine action parameterization  
- **Task 2.4**: State Representation - Add attention mechanisms

The Mini-ONI Environment provides a solid foundation for these subsequent tasks and RL agent training.