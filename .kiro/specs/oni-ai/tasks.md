# Implementation Tasks: ONI AI Agent

## Phase 1: Data Extraction Pipeline

### Task 1.1: ONI Save Parser Integration
- [x] Research and evaluate the [oni-save-parser](https://github.com/RoboPhred/oni-save-parser) JavaScript library
- [x] Create Python wrapper using subprocess or Node.js bridge
- [x] Implement `parse_save(file_path: str) -> GameState` interface
- [x] Extract core data: grid tiles, elements, temperatures, buildings, duplicants, cycle info
- [x] Handle corrupted save files with graceful error handling
- [x] Implement batch processing with `parse_save_batch()` method
- [x] Add comprehensive error categorization and statistics
- [x] Create enhanced mock data generation for testing
- [x] Write unit tests with sample save files
- [x] Refine detailed grid-level tile extraction (completed with enhanced data extraction)
- [x] **Estimated Time**: 1-2 weeks → **Status**: 100% complete

### Task 1.2: Data Preprocessor Implementation
- [x] Design `GameState` dataclass with numpy arrays
- [x] Implement `preprocess_state(game_state: GameState) -> StateTensor`
- [x] Create multi-channel tensor format (Height × Width × Channels):
  - [x] Material state channels (solid/liquid/gas) - 3 channels
  - [x] Element ID channel with one-hot encoding - 1 channel
  - [x] Temperature channel with normalization - 1 channel
  - [x] Building type channel with categorical encoding - 1 channel
  - [x] Duplicant positions channel - 1 channel
- [x] Implement normalization for temperature ranges and IDs
- [x] Add data validation and consistency checks
- [x] **Estimated Time**: 1 week → **Status**: 100% complete

### Task 1.3: Dataset Builder
- [x] Implement `build_dataset(save_files: List[str]) -> Dataset`
- [x] Design storage formats:
  - [x] JSON for metadata
  - [x] NPZ for tensor data
  - [x] Parquet for tabular features
- [x] Create labeling system for colony outcomes
- [x] Implement data augmentation (rotation, cropping)
- [x] Add batch processing with memory management
- [x] Create data loading utilities for ML frameworks
- [x] **Estimated Time**: 1 week

## Phase 2: Environment Design

### Task 2.1: Mini-ONI Environment Core
- [x] Define environment scope (64x64 tile maximum)
- [x] Implement rectangular starter base constraints
- [x] Create essential building type restrictions (10-15 types)
- [x] Set time horizon limits (100 cycles maximum)
- [x] Design environment reset and initialization
- [x] **Estimated Time**: 2 weeks

### Task 2.2: Objective System
- [x] Implement primary objective: oxygen maintenance (>500g/tile)
- [x] Add secondary objective: polluted water routing
- [x] Create tertiary objective: duplicant happiness (>50%)
- [x] Design objective evaluation and scoring
- [x] Add objective progress tracking
- [x] **Estimated Time**: 1 week

### Task 2.3: Action Space Design
- [x] Implement high-level action types:
  - [x] `PlaceBuildingAction(building_type, region)`
  - [x] `DigAction(region, material_type)`
  - [x] `PriorityAction(task_type, priority_level)`
  - [x] `DuplicantAction(duplicant_id, skill_assignment)`
- [x] Create action validation and masking
- [x] Limit action space to ~200 discrete actions
- [x] Implement action parameter encoding
- [x] **Estimated Time**: 1-2 weeks

### Task 2.4: State Representation
- [x] Design downscaled 32x32 multi-channel tensor format
- [x] Implement global features vector (64 dimensions)
- [x] Create attention mechanism for critical areas
- [x] Add state normalization and preprocessing
- [x] Implement state validation checks
- [ ] **Estimated Time**: 1 week

## Phase 3: Baseline Models

### Task 3.1: Supervised CNN Model
- [x] Implement ResNet-18 backbone with custom head
- [x] Design input processing for 32x32x8 state tensor
- [x] Create output layer for building placement probabilities
- [x] Implement cross-entropy loss for human placement decisions
- [x] Add spatial data augmentation pipeline
- [x] Create training loop with validation
- [ ] **Estimated Time**: 1-2 weeks

### Task 3.2: Heuristic Bot Implementation
- [x] Design rule engine with priority-based decisions
- [x] Implement problem detection systems:
  - [x] Oxygen level monitoring
  - [x] Temperature threshold detection
  - [x] Facility requirement checking
  - [x] Food shortage detection
- [x] Create template system for room layouts
- [x] Implement hierarchical decision tree
- [x] Add conflict resolution mechanisms
- [x] **Estimated Time**: 2-3 weeks

### Task 3.3: Imitation Learning Pipeline
- [x] Implement behavioral cloning model
- [x] Create state-action pair collection from heuristic bot
- [x] Design DAgger algorithm for iterative improvement
- [x] Implement loss functions (cross-entropy, MSE)
- [x] Create training and evaluation loops
- [x] Add performance comparison metrics
- [ ] **Estimated Time**: 1-2 weeks

## Phase 4: RL Training Environment

### Task 4.1: Gym Environment Wrapper
- [x] Implement standard OpenAI Gym API:
  - [x] `step(action) -> (observation, reward, done, info)`
  - [x] `reset() -> observation`
  - [x] `render(mode='human') -> None`
- [x] Choose implementation approach:
  - [x] Option A: Custom ONI simulator (preferred)
  - [ ] Option B: C# mod with file communication
- [x] Create parallel environment support (16-32 environments)
- [x] Add environment monitoring and logging
- [x] **Estimated Time**: 3-4 weeks

### Task 4.2: PPO Agent Implementation
- [x] Set up Stable Baselines3 with PyTorch backend
- [x] Design network architecture:
  - [x] CNN feature extractor
  - [x] Policy network with MLP head
  - [x] Value network with shared features
- [x] Configure hyperparameters:
  - [x] Learning rate: 3e-4 with cosine annealing
  - [x] Batch size: 256
  - [x] PPO clip ratio: 0.2
  - [x] GAE lambda: 0.95
  - [x] Entropy coefficient: 0.01
- [x] Implement training loop with checkpointing
- [x] **Estimated Time**: 2 weeks

### Task 4.3: Reward System Design
- [x] Implement dense rewards (per step):
  - [x] +0.1 per tile with breathable oxygen
  - [x] +0.05 per happy duplicant
  - [x] +0.02 per unit of stored resources
  - [x] -0.1 per dangerous temperature tile
- [x] Add sparse rewards (episodic):
  - [x] +100 for 100-cycle survival
  - [x] -50 for duplicant death
  - [x] +20 for infrastructure milestones
- [x] Implement potential-based reward shaping
- [x] **Estimated Time**: 1 week → **Status**: Complete

## Phase 5: Advanced Representations

### Task 5.1: Hierarchical Planning Architecture
- [x] Design high-level planner for abstract goals
- [x] Implement mid-level controller for goal decomposition
- [x] Create low-level executor for precise actions
- [x] Design goal embedding communication system
- [x] Add hierarchical coordination mechanisms
- [x] **Estimated Time**: 3-4 weeks → **Status**: Complete

### Task 5.2: LLM Integration
- [ ] Set up local LLM deployment (Code Llama 7B)
- [ ] Design natural language state summarization
- [ ] Implement structured build plan generation (JSON)
- [ ] Create feasibility checker for LLM suggestions
- [ ] Add LLM-agent communication interface
- [ ] **Estimated Time**: 2-3 weeks

## Infrastructure and Testing

### Task 6.1: Property-Based Testing Setup
- [ ] Set up Hypothesis testing framework
- [ ] Implement custom strategies for game states and actions
- [ ] Create property tests for all 24 defined properties
- [ ] Configure minimum 100 iterations per property test
- [ ] Integrate with pytest for test discovery
- [ ] **Estimated Time**: 2-3 weeks

### Task 6.2: Unit Testing Implementation
- [ ] Create unit tests for save file parsing
- [ ] Add tests for state tensor generation
- [ ] Implement reward function testing
- [ ] Create heuristic bot response tests
- [ ] Add environment step function tests
- [ ] Test model loading/saving functionality
- [ ] **Estimated Time**: 2 weeks

### Task 6.3: Data Management System
- [ ] Design database schema for training data
- [ ] Implement data storage (states, actions, episodes, models)
- [ ] Create data loading and batching utilities
- [ ] Add experiment logging and tracking
- [ ] Implement checkpoint and recovery systems
- [ ] **Estimated Time**: 1-2 weeks

### Task 6.4: Evaluation and Benchmarking
- [ ] Create standardized benchmark scenarios
- [ ] Implement performance metrics collection
- [ ] Design agent comparison framework
- [ ] Add failure mode analysis tools
- [ ] Create visualization and reporting tools
- [ ] **Estimated Time**: 1-2 weeks

## Deployment and Optimization

### Task 7.1: High-Performance Computing Setup
- [ ] Configure DGX Spark environment
- [ ] Set up distributed training capabilities
- [ ] Implement GPU memory optimization
- [ ] Add multi-node training support
- [ ] Create resource monitoring and allocation
- [ ] **Estimated Time**: 1-2 weeks

### Task 7.2: Model Optimization
- [ ] Implement model quantization and pruning
- [ ] Add inference optimization techniques
- [ ] Create model serving infrastructure
- [ ] Implement batch inference capabilities
- [ ] Add performance profiling tools
- [ ] **Estimated Time**: 1-2 weeks

### Task 7.3: Documentation and Deployment
- [ ] Create comprehensive API documentation
- [ ] Write user guides and tutorials
- [ ] Implement deployment scripts and containers
- [ ] Add monitoring and alerting systems
- [ ] Create maintenance and update procedures
- [ ] **Estimated Time**: 1-2 weeks

## Timeline Summary

**Phase 1 (Data Pipeline)**: 3-4 weeks - *✅ **100% COMPLETE** - All tasks (1.1, 1.2, 1.3) complete*
**Phase 2 (Environment)**: 5-6 weeks  
**Phase 3 (Baselines)**: 4-7 weeks
**Phase 4 (RL Training)**: 6-7 weeks
**Phase 5 (Advanced)**: 5-7 weeks
**Infrastructure**: 6-9 weeks
**Deployment**: 3-6 weeks

**Total Estimated Timeline**: 32-46 weeks (8-12 months)

## Dependencies and Prerequisites

- Python 3.8+ with PyTorch, Stable Baselines3, Hypothesis
- Node.js for ONI save parser integration
- Access to ONI save files for training data
- High-performance computing resources (DGX Spark)
- ONI game knowledge for heuristic rule design
- ML/RL expertise for model development and tuning

## Risk Mitigation

- **Save Parser Issues**: Have fallback manual parsing implementation
- **Environment Complexity**: Start with simplified version, gradually add features
- **Training Instability**: Implement robust checkpointing and recovery
- **Performance Issues**: Profile early and optimize bottlenecks
- **Data Quality**: Implement comprehensive validation and cleaning