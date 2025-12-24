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
- [-] **Estimated Time**: 2 weeks

### Task 2.2: Objective System
- [x] Implement primary objective: oxygen maintenance (>500g/tile)
- [x] Add secondary objective: polluted water routing
- [x] Create tertiary objective: duplicant happiness (>50%)
- [x] Design objective evaluation and scoring
- [x] Add objective progress tracking
- [x] **Estimated Time**: 1 week

### Task 2.3: Action Space Design
- [ ] Implement high-level action types:
  - [ ] `PlaceBuildingAction(building_type, region)`
  - [ ] `DigAction(region, material_type)`
  - [ ] `PriorityAction(task_type, priority_level)`
  - [ ] `DuplicantAction(duplicant_id, skill_assignment)`
- [ ] Create action validation and masking
- [ ] Limit action space to ~200 discrete actions
- [ ] Implement action parameter encoding
- [ ] **Estimated Time**: 1-2 weeks

### Task 2.4: State Representation
- [ ] Design downscaled 32x32 multi-channel tensor format
- [ ] Implement global features vector (64 dimensions)
- [ ] Create attention mechanism for critical areas
- [ ] Add state normalization and preprocessing
- [ ] Implement state validation checks
- [ ] **Estimated Time**: 1 week

## Phase 3: Baseline Models

### Task 3.1: Supervised CNN Model
- [ ] Implement ResNet-18 backbone with custom head
- [ ] Design input processing for 32x32x8 state tensor
- [ ] Create output layer for building placement probabilities
- [ ] Implement cross-entropy loss for human placement decisions
- [ ] Add spatial data augmentation pipeline
- [ ] Create training loop with validation
- [ ] **Estimated Time**: 1-2 weeks

### Task 3.2: Heuristic Bot Implementation
- [ ] Design rule engine with priority-based decisions
- [ ] Implement problem detection systems:
  - [ ] Oxygen level monitoring
  - [ ] Temperature threshold detection
  - [ ] Facility requirement checking
  - [ ] Food shortage detection
- [ ] Create template system for room layouts
- [ ] Implement hierarchical decision tree
- [ ] Add conflict resolution mechanisms
- [ ] **Estimated Time**: 2-3 weeks

### Task 3.3: Imitation Learning Pipeline
- [ ] Implement behavioral cloning model
- [ ] Create state-action pair collection from heuristic bot
- [ ] Design DAgger algorithm for iterative improvement
- [ ] Implement loss functions (cross-entropy, MSE)
- [ ] Create training and evaluation loops
- [ ] Add performance comparison metrics
- [ ] **Estimated Time**: 1-2 weeks

## Phase 4: RL Training Environment

### Task 4.1: Gym Environment Wrapper
- [ ] Implement standard OpenAI Gym API:
  - [ ] `step(action) -> (observation, reward, done, info)`
  - [ ] `reset() -> observation`
  - [ ] `render(mode='human') -> None`
- [ ] Choose implementation approach:
  - [ ] Option A: Custom ONI simulator (preferred)
  - [ ] Option B: C# mod with file communication
- [ ] Create parallel environment support (16-32 environments)
- [ ] Add environment monitoring and logging
- [ ] **Estimated Time**: 3-4 weeks

### Task 4.2: PPO Agent Implementation
- [ ] Set up Stable Baselines3 with PyTorch backend
- [ ] Design network architecture:
  - [ ] CNN feature extractor
  - [ ] Policy network with MLP head
  - [ ] Value network with shared features
- [ ] Configure hyperparameters:
  - [ ] Learning rate: 3e-4 with cosine annealing
  - [ ] Batch size: 256
  - [ ] PPO clip ratio: 0.2
  - [ ] GAE lambda: 0.95
  - [ ] Entropy coefficient: 0.01
- [ ] Implement training loop with checkpointing
- [ ] **Estimated Time**: 2 weeks

### Task 4.3: Reward System Design
- [ ] Implement dense rewards (per step):
  - [ ] +0.1 per tile with breathable oxygen
  - [ ] +0.05 per happy duplicant
  - [ ] +0.02 per unit of stored resources
  - [ ] -0.1 per dangerous temperature tile
- [ ] Add sparse rewards (episodic):
  - [ ] +100 for 100-cycle survival
  - [ ] -50 for duplicant death
  - [ ] +20 for infrastructure milestones
- [ ] Implement potential-based reward shaping
- [ ] **Estimated Time**: 1 week

## Phase 5: Advanced Representations

### Task 5.1: Hierarchical Planning Architecture
- [ ] Design high-level planner for abstract goals
- [ ] Implement mid-level controller for goal decomposition
- [ ] Create low-level executor for precise actions
- [ ] Design goal embedding communication system
- [ ] Add hierarchical coordination mechanisms
- [ ] **Estimated Time**: 3-4 weeks

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