# Technology Stack

## Core Technologies

### Programming Languages
- **Python 3.8+**: Primary language for ML/RL development
- **JavaScript/Node.js**: ONI save file parsing integration
- **C#**: Optional for direct ONI game integration via mods

### ML/AI Frameworks
- **PyTorch**: Deep learning framework for neural networks
- **Stable Baselines3**: Reinforcement learning algorithms (PPO, A2C, DQN)
- **Transformers**: For LLM integration (Code Llama, Llama 2)
- **Hypothesis**: Property-based testing framework

### Data Processing
- **NumPy**: Numerical computing and tensor operations
- **Pandas**: Data manipulation and analysis
- **Parquet**: Efficient columnar data storage
- **HDF5/NPZ**: Large array storage formats

### Visualization & Monitoring
- **Matplotlib**: Plotting and visualization
- **TensorBoard**: Training metrics and model monitoring
- **Weights & Biases**: Experiment tracking (optional)

### Development Tools
- **pytest**: Unit testing framework
- **black/autopep8**: Code formatting
- **flake8/pylint**: Linting and code quality
- **mypy**: Type checking
- **isort**: Import organization

## Hardware Requirements

### DGX Spark Optimization
- **GPU Acceleration**: Multi-GPU training for CNN and RL models
- **High-Bandwidth Memory**: Large state representations and replay buffers
- **Parallel Environments**: 16-32 concurrent training environments
- **Memory Management**: Efficient batching for large datasets

## Common Commands

### Environment Setup
```bash
# Install Python dependencies
pip install torch stable-baselines3 hypothesis pandas numpy matplotlib

# Install Node.js for save parser
npm install oni-save-parser

# Set up development tools
pip install pytest black flake8 mypy isort
```

### Development Workflow
```bash
# Run comprehensive tests
pytest tests/ --hypothesis-show-statistics

# Code quality checks
black src/ tests/
flake8 src/ tests/
mypy src/

# Property-based testing
pytest tests/test_properties.py -v --hypothesis-show-statistics
```

### Training Commands
```bash
# Train supervised baseline
python src/train_supervised.py --config configs/cnn_baseline.yaml

# Train RL agent
python src/train_rl.py --env mini-oni --algorithm ppo --parallel-envs 32

# Evaluate models
python src/evaluate.py --model checkpoints/best_model.zip --scenarios benchmark/
```

### Data Processing
```bash
# Parse ONI save files
python src/data/parse_saves.py --input saves/ --output data/processed/

# Build training dataset
python src/data/build_dataset.py --saves data/processed/ --output data/ml_ready/

# Validate data integrity
python src/data/validate.py --dataset data/ml_ready/
```

## Build System

### Project Structure
- Standard Python package with `src/` layout
- Configuration-driven training with YAML files
- Modular components for easy experimentation
- Automated testing and quality checks via hooks

### Dependency Management
- `requirements.txt` for core dependencies
- `requirements-dev.txt` for development tools
- Version pinning for reproducible builds
- Regular dependency updates via automation hooks