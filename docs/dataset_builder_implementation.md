# Dataset Builder Implementation

## Overview

The Dataset Builder module provides comprehensive functionality for converting ONI save files into ML-ready datasets with proper storage formats, labeling systems, and data loading utilities for various ML frameworks.

## Implementation Status

✅ **Task 1.3: Dataset Builder - COMPLETE**

All sub-tasks have been successfully implemented:

- ✅ `build_dataset(save_files: List[str]) -> Dataset` interface
- ✅ Storage formats (JSON for metadata, NPZ for tensors, Parquet for tabular features)
- ✅ Labeling system for colony outcomes
- ✅ Data augmentation (rotation, cropping)
- ✅ Batch processing with memory management
- ✅ Data loading utilities for ML frameworks

## Key Features

### 1. Dataset Building (`DatasetBuilder`)

**Core Functionality:**
- Parses ONI save files using the existing parser
- Preprocesses game states into ML-ready tensors
- Applies data augmentation (rotation, cropping)
- Creates comprehensive labeling system
- Manages memory usage with batch processing
- Stores data in optimized formats

**Memory Management:**
- Automatic batch size calculation based on available memory
- Temporary file storage for large datasets
- Configurable memory limits and compression levels
- Efficient processing of large save file collections

### 2. Storage Formats

**Multi-format Storage:**
- **NPZ files**: Compressed numpy arrays for spatial tensors and global features
- **Parquet files**: Efficient columnar storage for tabular features
- **JSON files**: Human-readable metadata and labels
- **Metadata**: Complete configuration and processing information

### 3. Labeling System

**Colony Outcome Labels:**
- `survival_status`: Boolean colony survival assessment
- `resource_counts`: Dictionary of resource quantities
- `duplicant_stress`: Average duplicant stress levels
- `infrastructure_score`: Infrastructure completion metrics
- `cycle_progression`: Game cycle information
- `colony_health`: Overall colony health score

### 4. Data Augmentation

**Spatial Augmentations:**
- **Rotation**: 90°, 180°, 270° rotations
- **Cropping**: Center crops with resize back to original dimensions
- **Configurable**: Adjustable augmentation factors and parameters

### 5. ML Framework Integration

**PyTorch Support:**
- `ONIPyTorchDataset`: Native PyTorch Dataset wrapper
- `create_pytorch_dataloader()`: DataLoader creation with batching
- Multiple return formats (tensors, dict, tuple)

**TensorFlow Support:**
- `create_tensorflow_dataset()`: tf.data.Dataset creation
- Automatic batching and shuffling
- Compatible tensor formats

**Scikit-learn Support:**
- `create_sklearn_data()`: X, y array format
- Feature type selection (spatial, global, tabular, combined)
- Automatic numeric feature extraction

**Framework-agnostic:**
- `ONIDataset`: Universal dataset wrapper
- `create_data_loader()`: Automatic framework detection
- Data splitting and subset functionality

## Usage Examples

### Basic Dataset Building

```python
from src.data.datasets import build_dataset

# Build dataset from save files
save_files = ["colony1.sav", "colony2.sav", "colony3.sav"]
dataset = build_dataset(
    save_files=save_files,
    output_dir="data/my_dataset"
)

print(f"Dataset contains {dataset.num_samples} samples")
```

### Advanced Configuration

```python
# Configure preprocessing
preprocessor_config = {
    'target_size': (64, 64),
    'temperature_range': (-50.0, 200.0)
}

# Configure augmentation
augmentation_config = {
    'enable_rotation': True,
    'rotation_angles': [90, 180, 270],
    'augmentation_factor': 3
}

# Configure memory management
memory_config = {
    'batch_size': 25,
    'max_memory_gb': 8.0,
    'use_temp_storage': True
}

dataset = build_dataset(
    save_files=save_files,
    preprocessor_config=preprocessor_config,
    augmentation_config=augmentation_config,
    memory_config=memory_config
)
```

### PyTorch Integration

```python
from src.data.datasets import ONIDataset, create_pytorch_dataloader

# Create dataset wrapper
oni_dataset = ONIDataset(dataset)

# Split data
train_ds, val_ds, test_ds = oni_dataset.split(0.7, 0.15, 0.15)

# Create PyTorch DataLoader
train_loader = create_pytorch_dataloader(
    train_ds, 
    batch_size=32, 
    shuffle=True,
    return_format='dict'
)

# Use in training loop
for batch in train_loader:
    spatial = batch['spatial']  # Shape: (batch_size, height, width, channels)
    global_feat = batch['global']  # Shape: (batch_size, 64)
    labels = batch['labels']
    # ... training code
```

### Scikit-learn Integration

```python
from src.data.datasets import create_sklearn_data

# Get data in scikit-learn format
X, y = create_sklearn_data(
    oni_dataset, 
    feature_type='combined',  # Use both global and tabular features
    target_type='survival_status'
)

# Use with scikit-learn models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
```

## File Structure

```
src/data/datasets/
├── __init__.py                 # Module exports
├── dataset_builder.py          # Core dataset building functionality
└── data_loaders.py            # ML framework integration

examples/
└── dataset_builder_demo.py    # Comprehensive usage demo

docs/
└── dataset_builder_implementation.md  # This documentation
```

## Testing and Validation

**Validation Script:** `validate_dataset_builder.py`
- Tests basic dataset building
- Validates data loading functionality
- Checks ML framework integration
- Verifies data splitting and subset operations

**Demo Script:** `examples/dataset_builder_demo.py`
- Comprehensive demonstration of all features
- Framework availability checking
- Performance testing with real save files

## Performance Characteristics

**Memory Efficiency:**
- Automatic batch processing for large datasets
- Configurable memory limits
- Temporary file storage for memory-constrained environments
- Compressed storage formats (NPZ with configurable compression)

**Processing Speed:**
- Parallel-ready architecture
- Efficient numpy operations
- Optimized data loading with caching
- Framework-specific optimizations

**Storage Efficiency:**
- Compressed tensor storage (NPZ)
- Columnar tabular data (Parquet)
- Minimal metadata overhead
- Configurable compression levels

## Integration with Existing Components

**Parser Integration:**
- Uses existing `ONISaveParser` for save file processing
- Handles parser errors gracefully with mock data fallbacks
- Supports batch parsing with error categorization

**Preprocessor Integration:**
- Uses existing `StatePreprocessor` for tensor generation
- Configurable preprocessing parameters
- Maintains preprocessing metadata for reproducibility

**Future Compatibility:**
- Extensible labeling system
- Pluggable augmentation strategies
- Framework-agnostic design
- Version-tracked metadata

## Next Steps

With Task 1.3 complete, Phase 1 (Data Extraction Pipeline) is now **100% complete**. The system is ready to proceed to:

- **Phase 2**: Environment Design (Mini-ONI Environment, Action Space, State Representation)
- **Phase 3**: Baseline Models (Supervised CNN, Heuristic Bot, Imitation Learning)

The dataset builder provides a solid foundation for all future ML training and evaluation tasks in the ONI AI Agent project.