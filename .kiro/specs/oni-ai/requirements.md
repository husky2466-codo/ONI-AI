# Requirements: ONI AI Agent

## Project Overview

This project follows a standard ML progression: start with offline data and simple goals, then move toward an online RL agent once tools and baselines are established. The approach prioritizes practical implementation over theoretical completeness, building incrementally from data extraction to a full reinforcement learning agent.

## Phase 1: Tooling and Data Extraction (3-4 weeks)

### 1.1 ONI Save Parser Integration
**Requirement**: Set up tools to read ONI save files into structured data
- **R1.1.1**: Install or fork an existing ONI save parser
  - Primary option: [oni-save-parser](https://github.com/RoboPhred/oni-save-parser) (JavaScript)
  - Alternative: .NET parser for direct C# integration
- **R1.1.2**: Create Python wrapper to interface with the parser
- **R1.1.3**: Extract core game state data:
  - Grid tiles (material states, positions)
  - Element types and properties
  - Temperature maps
  - Building locations and types
  - Duplicant status and positions
  - Cycle number and game progression
- **R1.1.4**: Handle parsing errors gracefully with fallback mechanisms

### 1.2 Data Export Pipeline
**Requirement**: Convert save files to ML-ready format
- **R1.2.1**: Create compact snapshot exporter that outputs:
  - JSON for metadata and structured data
  - Parquet for tabular features
  - NPZ for tensor data (numpy arrays)
- **R1.2.2**: Implement efficient data serialization for large datasets
- **R1.2.3**: Add data validation and integrity checks
- **R1.2.4**: Support batch processing of multiple save files

### 1.3 Dataset Construction
**Requirement**: Build comprehensive training dataset
- **R1.3.1**: Collect diverse save files from multiple sources:
  - Personal gameplay saves across different scenarios
  - Community saves (with appropriate permissions)
  - Various colony stages and outcomes
- **R1.3.2**: Generate state tensors with format: height × width × channels
  - Material state channels: solid/liquid/gas (3 channels)
  - Element ID channel (1 channel, categorical)
  - Temperature channel (1 channel, normalized)
  - Building type channel (1 channel, categorical)
- **R1.3.3**: Create supervised learning labels:
  - Colony survival status (alive/dead)
  - Resource counts (oxygen, food, water, power)
  - Duplicant stress levels
  - Infrastructure completion metrics

## Phase 2: Simplified Game and Objectives (5-6 weeks)

### 2.1 Problem Scope Definition
**Requirement**: Define manageable "mini-ONI" environment
- **R2.1.1**: Limit map area to rectangular starter base (e.g., 64x64 tiles)
- **R2.1.2**: Restrict available buildings to essential subset (10-15 types)
- **R2.1.3**: Set clear episode termination conditions
- **R2.1.4**: Define success/failure criteria for training

### 2.2 Core Objectives
**Requirement**: Establish clear, measurable goals
- **R2.2.1**: Primary objective: Maintain oxygen above threshold (e.g., 500g/tile) in living areas
- **R2.2.2**: Secondary objective: Route polluted water through water sieve system
- **R2.2.3**: Tertiary objective: Maintain duplicant happiness above baseline (50%)
- **R2.2.4**: Add resource management goals (food, power sustainability)

### 2.3 Action Space Design
**Requirement**: Create manageable action space for RL
- **R2.3.1**: Use coarse, high-level commands instead of cell-level micromanagement:
  - "Place building type B in region R"
  - "Build ladder column at position X"
  - "Set priority level P for task type T"
  - "Assign duplicant D to skill S"
- **R2.3.2**: Limit total action space to ~200 discrete actions
- **R2.3.3**: Implement action masking for invalid moves
- **R2.3.4**: Design action parameterization for spatial commands

### 2.4 State Representation
**Requirement**: Efficient state encoding for ML models
- **R2.4.1**: Downscale grid to manageable resolution (e.g., 32x32)
- **R2.4.2**: Include key global features as numeric vector:
  - Resource counts and rates
  - Duplicant statistics
  - Cycle number and time progression
- **R2.4.3**: Add attention mechanisms for critical areas
- **R2.4.4**: Normalize all features for stable training

## Phase 3: Supervised and Heuristic Baselines (4-7 weeks)

### 3.1 Supervised Learning Models
**Requirement**: Train models on human gameplay data
- **R3.1.1**: Implement CNN architecture for spatial reasoning
- **R3.1.2**: Predict "good" design patterns from human saves
- **R3.1.3**: Train building placement models given local context
- **R3.1.4**: Create tile-wise building type prediction system
- **R3.1.5**: Validate model performance on held-out saves

### 3.2 Heuristic Bot Implementation
**Requirement**: Rule-based baseline agent
- **R3.2.1**: Implement problem detection system:
  - Low oxygen level alerts
  - High temperature warnings
  - Missing essential facilities (toilets, food, power)
  - Resource shortage detection
- **R3.2.2**: Create template-based building patterns
- **R3.2.3**: Design priority-based decision making
- **R3.2.4**: Add conflict resolution for competing objectives

### 3.3 Baseline Evaluation
**Requirement**: Establish performance benchmarks
- **R3.3.1**: Define success metrics for baseline comparison
- **R3.3.2**: Create standardized test scenarios
- **R3.3.3**: Measure baseline agent performance across scenarios
- **R3.3.4**: Generate synthetic trajectories for imitation learning

## Phase 4: Online RL Loop (3-6+ weeks)

### 4.1 Environment Wrapper
**Requirement**: Create RL-compatible game interface
- **R4.1.1**: **Preferred approach**: Build simplified ONI-like prototype with explicit API
  - Implement `step(action) -> (next_state, reward, done, info)`
  - Ensure fast execution for RL training
  - Maintain core ONI mechanics (oxygen, temperature, resources)
- **R4.1.2**: **Alternative approach**: C# mod for real ONI game
  - Create state dump mechanism to file/pipe
  - Implement action file reader for build commands
  - Handle game timing and synchronization

### 4.2 RL Algorithm Configuration
**Requirement**: Set up reinforcement learning training
- **R4.2.1**: Use Stable Baselines3 framework with proven algorithms:
  - PPO (Proximal Policy Optimization) - primary choice
  - A2C (Advantage Actor-Critic) - alternative
  - DQN variants for discrete action spaces
- **R4.2.2**: Leverage DGX Spark capabilities:
  - Utilize unified memory for large replay buffers
  - Run multiple parallel environments
  - Optimize GPU utilization for training

### 4.3 Reward System Design
**Requirement**: Create effective reward signal
- **R4.3.1**: Implement dense reward components:
  - **Positive rewards**:
    - +points for maintaining breathable tiles
    - +points for stable temperatures
    - +points for increasing stored food and power
    - +points for duplicant happiness
  - **Negative rewards**:
    - -points for duplicant death
    - -points for extreme stress levels
    - -points for unsustainable resource drain
- **R4.3.2**: Balance reward components to avoid exploitation
- **R4.3.3**: Add sparse rewards for major milestones
- **R4.3.4**: Implement reward shaping for faster learning

### 4.4 Training Infrastructure
**Requirement**: Efficient training pipeline on DGX Spark
- **R4.4.1**: **For prototype environment**: Spin up many parallel environments
- **R4.4.2**: **For real ONI**: Record long sessions and train off-policy from logged trajectories
- **R4.4.3**: Implement regular evaluation against heuristic bot on fixed scenarios
- **R4.4.4**: Add checkpointing and experiment tracking
- **R4.4.5**: Monitor training stability and convergence

## Phase 5: Iteration and Scaling (Ongoing)

### 5.1 Advanced Representations
**Requirement**: Improve state understanding and planning
- **R5.1.1**: Add higher-level feature extraction:
  - Room detection and classification
  - System identification (power, plumbing, ventilation)
  - Automation network mapping
- **R5.1.2**: Implement multi-channel spatial representations
- **R5.1.3**: Add temporal features for trend analysis
- **R5.1.4**: Create hierarchical state abstractions

### 5.2 Hierarchical Planning
**Requirement**: Multi-level decision making
- **R5.2.1**: Design high-level planner for abstract goals
  - "Build electrolyzer room"
  - "Establish food production"
  - "Create cooling system"
- **R5.2.2**: Implement low-level controller for execution
- **R5.2.3**: Create goal decomposition mechanisms
- **R5.2.4**: Add coordination between planning levels

### 5.3 Language Model Integration
**Requirement**: LLM-assisted planning and reasoning
- **R5.3.1**: Deploy local LLM on DGX Spark (e.g., Code Llama, Llama 2)
- **R5.3.2**: Create ONI state summarization for LLM input
- **R5.3.3**: Generate high-level build plans from LLM suggestions
- **R5.3.4**: Integrate LLM planning with RL controller execution
- **R5.3.5**: Add feasibility checking for LLM proposals

## Technical Requirements

### 6.1 Development Stack
**Requirement**: Define technology choices and setup
- **R6.1.1**: **Primary languages**: Python for ML/RL, C# for ONI integration
- **R6.1.2**: **ML frameworks**: PyTorch, Stable Baselines3, Transformers
- **R6.1.3**: **Data handling**: Pandas, NumPy, Parquet, HDF5
- **R6.1.4**: **Visualization**: Matplotlib, TensorBoard, Weights & Biases
- **R6.1.5**: **Game integration**: Unity (if building prototype), ONI modding tools

### 6.2 Hardware Requirements
**Requirement**: Optimize for DGX Spark capabilities
- **R6.2.1**: Utilize high-bandwidth memory for large state representations
- **R6.2.2**: Leverage GPU acceleration for CNN training and inference
- **R6.2.3**: Implement multi-GPU training for faster convergence
- **R6.2.4**: Optimize memory usage for parallel environments

### 6.3 Data Management
**Requirement**: Efficient data pipeline and storage
- **R6.3.1**: Implement scalable data loading for large datasets
- **R6.3.2**: Add data versioning and experiment reproducibility
- **R6.3.3**: Create efficient batch processing for save file conversion
- **R6.3.4**: Implement data quality monitoring and validation

## Success Criteria

### 7.1 Phase-wise Milestones
- **Phase 1**: Successfully parse and convert 1000+ ONI saves to ML format
- **Phase 2**: Define and validate mini-ONI environment with clear objectives
- **Phase 3**: Achieve baseline performance better than random on test scenarios
- **Phase 4**: Train RL agent that outperforms heuristic bot on core objectives
- **Phase 5**: Demonstrate hierarchical planning and LLM integration benefits

### 7.2 Performance Metrics
- **Survival rate**: Percentage of episodes where colony survives target duration
- **Resource efficiency**: Sustainable resource production and consumption
- **Objective completion**: Success rate on defined mini-ONI goals
- **Learning efficiency**: Sample complexity and training time requirements
- **Scalability**: Performance on larger maps and longer episodes

### 7.3 Quality Assurance
- **Code quality**: Comprehensive testing, documentation, and version control
- **Reproducibility**: Deterministic results with proper seed management
- **Robustness**: Graceful handling of edge cases and failures
- **Maintainability**: Modular design for easy extension and modification

## Risk Mitigation

### 8.1 Technical Risks
- **Save parser compatibility**: Maintain fallback options for different ONI versions
- **Training instability**: Implement robust hyperparameter tuning and monitoring
- **Environment complexity**: Start simple and gradually increase complexity
- **Performance bottlenecks**: Profile early and optimize critical paths

### 8.2 Project Risks
- **Scope creep**: Maintain focus on core objectives and defer advanced features
- **Data availability**: Ensure sufficient diverse training data before model development
- **Resource constraints**: Plan computational requirements and optimize for available hardware
- **Timeline management**: Regular milestone reviews and adaptive planning