#!/usr/bin/env python3
"""
ONI AI Agent - Experiment Logging Script

This script handles automatic logging of training experiments, including:
- Saving hyperparameters and configuration
- Recording performance metrics and results  
- Updating experiment comparison tables
- Checking for new best results
- Backing up model checkpoints
- Updating project progress tracking
"""

# Standard library
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import pandas as pd
import yaml

class ExperimentLogger:
    """Handles comprehensive experiment logging for ONI AI training.
    
    This class provides a complete experiment tracking system that manages:
    - Configuration storage and versioning
    - Metrics logging and checkpoint management
    - Comparison tables and performance tracking
    - Markdown log generation for human review
    
    Attributes:
        base_path: Base directory for experiment files
        log_file: Markdown log file path
        results_file: JSON results storage path
        comparison_file: CSV comparison table path
        checkpoints_dir: Model checkpoint storage directory
    """
    
    def __init__(self, base_path: str = "experiments"):
        self.base_path = Path(base_path)
        self.log_file = self.base_path / "experiment_log.md"
        self.results_file = self.base_path / "results.json"
        self.comparison_file = self.base_path / "comparison_table.csv"
        self.checkpoints_dir = Path("checkpoints")
        
        # Create directories if they don't exist
        self.base_path.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
    def start_experiment(self, config: Dict[str, Any]) -> str:
        """Start logging a new experiment.
        
        Args:
            config: Experiment configuration dictionary containing model,
                   training, and hardware settings
                   
        Returns:
            Unique experiment ID string in format YYYY-MM-DD-HH-MM-SS
            
        Raises:
            OSError: If unable to create config directory or save file
        """
        experiment_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config["experiment"]["id"] = experiment_id
        
        # Save experiment config
        config_path = self.base_path / f"configs/{experiment_id}.yaml"
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"Started experiment {experiment_id}")
        return experiment_id
        
    def log_metrics(
        self, 
        experiment_id: str, 
        metrics: Dict[str, float], 
        epoch: Optional[int] = None
    ) -> None:
        """Log performance metrics for an experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            metrics: Dictionary of metric names to values
            epoch: Optional training epoch number
            
        Raises:
            FileNotFoundError: If experiment_id doesn't exist
            json.JSONDecodeError: If results file is corrupted
        """
        timestamp = datetime.now().isoformat()
        
        # Load or create results file
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {}
            
        if experiment_id not in results:
            results[experiment_id] = {"metrics": [], "checkpoints": []}
            
        # Add metrics entry
        metric_entry = {
            "timestamp": timestamp,
            "epoch": epoch,
            "metrics": metrics
        }
        results[experiment_id]["metrics"].append(metric_entry)
        
        # Save updated results
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    def save_checkpoint(
        self, 
        experiment_id: str, 
        model_path: str, 
        metrics: Dict[str, float], 
        is_best: bool = False
    ) -> None:
        """Save model checkpoint and update tracking.
        
        Args:
            experiment_id: Unique experiment identifier
            model_path: Path to the model file to checkpoint
            metrics: Performance metrics at checkpoint time
            is_best: Whether this is the best model so far
            
        Raises:
            FileNotFoundError: If model_path doesn't exist
            OSError: If unable to copy checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint directory for this experiment
        exp_checkpoint_dir = self.checkpoints_dir / experiment_id
        exp_checkpoint_dir.mkdir(exist_ok=True)
        
        # Copy model file
        checkpoint_name = f"model_epoch_{metrics.get('epoch', 0)}_{timestamp}.pth"
        if is_best:
            checkpoint_name = f"best_model_{timestamp}.pth"
            
        checkpoint_path = exp_checkpoint_dir / checkpoint_name
        shutil.copy2(model_path, checkpoint_path)
        
        # Update results with checkpoint info
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {}
            
        if experiment_id not in results:
            results[experiment_id] = {"metrics": [], "checkpoints": []}
            
        checkpoint_entry = {
            "timestamp": datetime.now().isoformat(),
            "path": str(checkpoint_path),
            "metrics": metrics,
            "is_best": is_best
        }
        results[experiment_id]["checkpoints"].append(checkpoint_entry)
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved checkpoint: {checkpoint_path}")
        if is_best:
            print("🏆 New best model!")
            
    def finish_experiment(
        self, 
        experiment_id: str, 
        final_metrics: Dict[str, float],
        status: str = "completed", 
        notes: str = ""
    ) -> None:
        """Complete experiment logging and update comparison tables.
        
        Args:
            experiment_id: Unique experiment identifier
            final_metrics: Final performance metrics
            status: Experiment completion status (completed, failed, stopped)
            notes: Optional notes about the experiment
            
        Raises:
            FileNotFoundError: If experiment config doesn't exist
        """
        
        # Load experiment config
        config_path = self.base_path / f"configs/{experiment_id}.yaml"
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        # Update comparison table
        self._update_comparison_table(experiment_id, config, final_metrics, status)
        
        # Update experiment log markdown
        self._update_experiment_log(experiment_id, config, final_metrics, status, notes)
        
        print(f"Experiment {experiment_id} completed with status: {status}")
        
    def _update_comparison_table(
        self, 
        experiment_id: str, 
        config: Dict[str, Any],
        metrics: Dict[str, float], 
        status: str
    ) -> None:
        """Update the CSV comparison table with experiment results."""
        
        # Load existing table or create new one
        if self.comparison_file.exists():
            df = pd.read_csv(self.comparison_file)
        else:
            df = pd.DataFrame(columns=[
                "experiment_id", "model_type", "phase", "survival_rate", 
                "resource_efficiency", "training_time", "status", "date"
            ])
            
        # Create new row
        new_row = {
            "experiment_id": experiment_id,
            "model_type": config["experiment"]["model_type"],
            "phase": config["experiment"]["phase"],
            "survival_rate": metrics.get("survival_rate", 0.0),
            "resource_efficiency": metrics.get("resource_efficiency", 0.0),
            "training_time": metrics.get("training_time_hours", 0.0),
            "status": status,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Add row and save
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.comparison_file, index=False)
        
    def _update_experiment_log(
        self, 
        experiment_id: str, 
        config: Dict[str, Any],
        metrics: Dict[str, float], 
        status: str, 
        notes: str
    ) -> None:
        """Update the markdown experiment log with new experiment entry."""
        
        # Read existing log
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                content = f.read()
        else:
            content = "# ONI AI Agent - Experiment Log\n\n"
            
        # Create experiment entry
        experiment_entry = f"""
### Experiment ID: {experiment_id}
- **Phase**: {config["experiment"]["phase"]} ({self._get_phase_name(config["experiment"]["phase"])})
- **Model Type**: {config["experiment"]["model_type"]}
- **Objective**: {config["experiment"]["objective"]}
- **Status**: {status}

#### Configuration
```yaml
{yaml.dump(config, default_flow_style=False)}
```

#### Results
- **Survival Rate**: {metrics.get("survival_rate", "N/A")}
- **Resource Efficiency**: {metrics.get("resource_efficiency", "N/A")}
- **Training Time**: {metrics.get("training_time_hours", "N/A")} hours
- **Hardware Used**: {config.get("hardware", {}).get("device", "DGX Spark")}

#### Notes
{notes}

---

"""
        
        # Insert before "## Experiment History" section
        if "## Experiment History" in content:
            content = content.replace("## Experiment History", 
                                    f"## Experiment History{experiment_entry}")
        else:
            content += experiment_entry
            
        # Write updated log
        with open(self.log_file, 'w') as f:
            f.write(content)
            
    def _get_phase_name(self, phase: int) -> str:
        """Get human-readable phase name."""
        phase_names = {
            1: "Data Extraction",
            2: "Environment Design", 
            3: "Baseline Models",
            4: "RL Training",
            5: "Advanced Features"
        }
        return phase_names.get(phase, "Unknown")
        
    def get_best_results(
        self, 
        phase: Optional[int] = None, 
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get best results for comparison.
        
        Args:
            phase: Optional phase filter (1-5)
            model_type: Optional model type filter (CNN, PPO, etc.)
            
        Returns:
            Dictionary containing best experiment results, empty if none found
            
        Raises:
            FileNotFoundError: If comparison file doesn't exist
        """
        if not self.comparison_file.exists():
            return {}
            
        df = pd.read_csv(self.comparison_file)
        
        # Filter by phase and model type if specified
        if phase:
            df = df[df["phase"] == phase]
        if model_type:
            df = df[df["model_type"] == model_type]
            
        if df.empty:
            return {}
            
        # Find best by survival rate
        best_idx = df["survival_rate"].idxmax()
        return df.loc[best_idx].to_dict()

# Example usage
if __name__ == "__main__":
    logger = ExperimentLogger()
    
    # Example experiment config
    config = {
        "experiment": {
            "name": "test_experiment",
            "phase": 1,
            "model_type": "CNN",
            "objective": "Test experiment logging system"
        },
        "hardware": {"device": "DGX_Spark"},
        "training": {"epochs": 10}
    }
    
    # Start experiment
    exp_id = logger.start_experiment(config)
    
    # Log some metrics
    logger.log_metrics(exp_id, {"accuracy": 0.85, "loss": 0.3}, epoch=5)
    
    # Finish experiment
    logger.finish_experiment(exp_id, {"survival_rate": 0.75, "training_time_hours": 2.5})
    
    print("Experiment logging test completed!")