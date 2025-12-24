# ONI AI Agent - Experiment Tracking

This directory contains all experiment tracking, logging, and comparison tools for the ONI AI Agent project.

## Directory Structure

```
experiments/
├── experiment_log.md           # Main experiment log (human-readable)
├── results.json               # Detailed results data (machine-readable)
├── comparison_table.csv       # Performance comparison table
├── configs/                   # Saved experiment configurations
│   └── YYYY-MM-DD-HH-MM-SS.yaml
├── scripts/
│   └── log_experiment.py      # Experiment logging utilities
└── config_templates/
    └── experiment_config_template.yaml
```

## Usage

### Starting an Experiment

```python
from experiments.scripts.log_experiment import ExperimentLogger

logger = ExperimentLogger()

# Load your experiment config
with open("experiments/config_templates/experiment_config_template.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Customize config for your experiment
config["experiment"]["name"] = "cnn_baseline_v1"
config["experiment"]["objective"] = "Train CNN for building placement prediction"

# Start logging
experiment_id = logger.start_experiment(config)
```

### During Training

```python
# Log metrics each epoch
for epoch in range(num_epochs):
    # ... training code ...
    
    metrics = {
        "accuracy": accuracy,
        "loss": loss,
        "survival_rate": survival_rate
    }
    logger.log_metrics(experiment_id, metrics, epoch=epoch)
    
    # Save checkpoint if performance improved
    if is_best_model:
        logger.save_checkpoint(experiment_id, model_path, metrics, is_best=True)
```

### Finishing an Experiment

```python
final_metrics = {
    "survival_rate": 0.85,
    "resource_efficiency": 0.72,
    "training_time_hours": 4.5
}

logger.finish_experiment(
    experiment_id, 
    final_metrics, 
    status="completed",
    notes="Successfully trained CNN baseline. Ready for RL phase."
)
```

## Experiment Configuration

Use the template in `config_templates/experiment_config_template.yaml` as a starting point. Key sections:

- **experiment**: Basic metadata and objectives
- **hardware**: DGX Spark configuration
- **model**: Architecture specifications
- **training**: Hyperparameters and training settings
- **data**: Dataset and preprocessing configuration
- **evaluation**: Metrics and benchmarking setup
- **logging**: Output and checkpoint settings

## Metrics Tracked

### Phase 1: Data Extraction
- Parsing accuracy
- Processing speed (saves/hour)
- Dataset size and quality metrics

### Phase 2: Environment Design
- Environment stability
- Action space coverage
- State representation quality

### Phase 3: Baseline Models
- Model accuracy/loss
- Inference speed
- Memory usage

### Phase 4: RL Training
- Colony survival rate
- Resource efficiency
- Episode length
- Reward progression

### Phase 5: Advanced Features
- Hierarchical planning success
- LLM integration quality
- Multi-scale performance

## Best Practices

1. **Always use the experiment logger** for any training runs
2. **Save configurations** before starting experiments
3. **Log metrics regularly** during training
4. **Save checkpoints** for promising models
5. **Document observations** in the notes field
6. **Compare results** using the comparison table
7. **Back up best models** to prevent data loss

## Automated Logging

The project includes Kiro hooks that automatically trigger experiment logging:

- `ml-training-monitor.kiro.hook`: Monitors active training jobs
- `ml-experiment-logger.kiro.hook`: Logs completed experiments
- `update-research-context.kiro.hook`: Updates research progress

## Integration with DGX Spark

The logging system is optimized for DGX Spark workflows:

- Automatic GPU memory tracking
- Multi-GPU training support
- Distributed training coordination
- Resource utilization monitoring

## Viewing Results

- **Human-readable**: Check `experiment_log.md`
- **Data analysis**: Load `results.json` or `comparison_table.csv`
- **Best models**: Check `checkpoints/` directory
- **Configurations**: Review `configs/` for reproducibility