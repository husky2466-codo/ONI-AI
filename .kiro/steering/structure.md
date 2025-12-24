# Project Structure

## Directory Organization

### Source Code Layout
```
src/
├── data/                    # Data processing and parsing
│   ├── parsers/            # ONI save file parsers
│   ├── preprocessors/      # State tensor generation
│   └── datasets/           # Dataset builders and loaders
├── environments/           # Game environments and wrappers
│   ├── mini_oni/          # Simplified ONI environment
│   └── gym_wrappers/      # OpenAI Gym compatibility
├── models/                 # Neural network architectures
│   ├── cnn/               # Convolutional networks for spatial reasoning
│   ├── rl/                # RL agent implementations
│   └── hierarchical/      # Multi-level planning models
├── agents/                 # Agent implementations
│   ├── heuristic/         # Rule-based baseline agents
│   ├── supervised/        # Imitation learning agents
│   └── rl/                # Reinforcement learning agents
├── training/              # Training loops and utilities
│   ├── supervised/        # Supervised learning training
│   ├── rl/                # RL training pipelines
│   └── evaluation/        # Model evaluation and benchmarking
└── utils/                 # Shared utilities and helpers
    ├── config/            # Configuration management
    ├── logging/           # Experiment tracking
    └── visualization/     # Plotting and rendering
```

### Configuration and Data
```
configs/                   # Training and model configurations
├── environments/          # Environment-specific configs
├── models/               # Model architecture configs
└── training/             # Training hyperparameters

data/                     # Training and evaluation data
├── raw/                  # Original ONI save files
├── processed/            # Parsed game states
├── ml_ready/             # ML-formatted datasets
└── benchmarks/           # Standardized test scenarios

checkpoints/              # Model checkpoints and saved weights
├── supervised/           # Baseline model checkpoints
├── rl/                   # RL agent checkpoints
└── best_models/          # Production-ready models
```

### Testing Structure
```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_parsers.py   # Save file parsing tests
│   ├── test_models.py    # Neural network tests
│   └── test_agents.py    # Agent behavior tests
├── integration/          # End-to-end integration tests
│   ├── test_training.py  # Training pipeline tests
│   └── test_evaluation.py # Evaluation workflow tests
├── properties/           # Property-based tests with Hypothesis
│   ├── test_data_properties.py      # Data processing properties
│   ├── test_environment_properties.py # Environment behavior properties
│   └── test_agent_properties.py     # Agent decision properties
└── fixtures/             # Test data and mock objects
    ├── sample_saves/     # Example ONI save files
    └── mock_states/      # Synthetic game states
```

## Code Organization Principles

### Modular Design
- **Separation of Concerns**: Clear boundaries between data processing, environment simulation, model training, and evaluation
- **Plugin Architecture**: Easy to swap different parsers, models, or training algorithms
- **Configuration-Driven**: All hyperparameters and settings externalized to YAML files

### Phase-Based Structure
- **Phase 1 (Data)**: `src/data/` contains all save parsing and preprocessing logic
- **Phase 2 (Environment)**: `src/environments/` implements game simulation and action spaces
- **Phase 3 (Baselines)**: `src/agents/heuristic/` and `src/agents/supervised/` for baseline models
- **Phase 4 (RL)**: `src/agents/rl/` and `src/training/rl/` for reinforcement learning
- **Phase 5 (Advanced)**: `src/models/hierarchical/` for multi-level planning

### Interface Standards
- **Consistent APIs**: All components follow standard interfaces (e.g., `step()`, `reset()`, `predict()`)
- **Type Hints**: Full type annotations for better IDE support and error catching
- **Dataclasses**: Structured data objects for game states, actions, and configurations
- **Error Handling**: Graceful degradation and informative error messages

## File Naming Conventions

### Python Modules
- **Snake case**: `parse_save_files.py`, `train_rl_agent.py`
- **Descriptive names**: Clear indication of module purpose
- **Consistent prefixes**: `test_` for tests, `config_` for configurations

### Configuration Files
- **Environment configs**: `mini_oni_64x64.yaml`, `full_oni_standard.yaml`
- **Model configs**: `resnet18_cnn.yaml`, `ppo_agent.yaml`
- **Training configs**: `supervised_baseline.yaml`, `rl_training.yaml`

### Data Files
- **Raw saves**: `colony_cycle_100.sav`, `failed_oxygen_crisis.sav`
- **Processed data**: `states_batch_001.npz`, `trajectories_heuristic.parquet`
- **Checkpoints**: `model_epoch_50.pth`, `best_ppo_agent.zip`

## Import Organization

### Standard Import Order
1. Standard library imports
2. Third-party library imports (torch, numpy, etc.)
3. Local project imports (src modules)
4. Relative imports within same package

### Example Import Structure
```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np
import torch
from stable_baselines3 import PPO

# Project imports
from src.data.parsers import ONISaveParser
from src.environments.mini_oni import MiniONIEnv
from src.utils.config import load_config

# Relative imports
from .base_agent import BaseAgent
```

## Documentation Standards

### Docstring Format
- **Google style**: Consistent docstring format across all modules
- **Type information**: Parameter and return types clearly documented
- **Examples**: Usage examples for complex functions
- **Property tests**: Reference to related property-based tests

### Code Comments
- **Algorithm explanations**: Complex ML/RL algorithms well-commented
- **Design decisions**: Rationale for architectural choices
- **TODOs**: Clear marking of future improvements
- **Performance notes**: Memory and computational complexity notes