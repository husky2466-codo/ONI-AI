#!/usr/bin/env python3
"""
Dataset Builder Demo for ONI AI Agent.

This script demonstrates the dataset building functionality by creating
a dataset from sample ONI save files and showing various data loading options.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.datasets import build_dataset, ONIDataset, create_data_loader, get_framework_info, demo_data_loaders


def main():
    """Run the dataset builder demo."""
    print("=== ONI Dataset Builder Demo ===")
    
    # Check for sample save files
    save_files_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_saves"
    
    if not save_files_dir.exists():
        print(f"Sample saves directory not found: {save_files_dir}")
        print("Creating mock dataset with empty file list (will use mock data)")
        save_files = []
    else:
        # Find all .sav files
        save_files = list(save_files_dir.glob("*.sav"))
        print(f"Found {len(save_files)} save files in {save_files_dir}")
    
    if not save_files:
        print("No save files found, creating minimal mock dataset...")
        # Create a minimal dataset with mock data
        save_files = ["mock_colony.sav"]  # This will trigger mock data generation
    
    # Convert to string paths
    save_file_paths = [str(f) for f in save_files]
    
    # Configure dataset building
    output_dir = "data/demo_dataset"
    
    # Configuration for smaller dataset (good for demo)
    preprocessor_config = {
        'target_size': (32, 32),  # Smaller for demo
        'temperature_range': (-50.0, 200.0),
        'max_element_id': 255,
        'max_building_types': 20
    }
    
    augmentation_config = {
        'enable_rotation': True,
        'enable_cropping': True,
        'rotation_angles': [90, 180],  # Fewer angles for demo
        'crop_ratios': [0.9],  # Single crop ratio
        'augmentation_factor': 2
    }
    
    memory_config = {
        'batch_size': 10,  # Small batch for demo
        'max_memory_gb': 2.0,  # Conservative memory usage
        'use_temp_storage': True,
        'compression_level': 6
    }
    
    try:
        print("\n--- Building Dataset ---")
        dataset = build_dataset(
            save_files=save_file_paths,
            output_dir=output_dir,
            preprocessor_config=preprocessor_config,
            augmentation_config=augmentation_config,
            memory_config=memory_config
        )
        
        print(f"\n--- Dataset Built Successfully ---")
        print(f"Output directory: {dataset.tensor_data_path}")
        print(f"Number of samples: {dataset.num_samples}")
        print(f"Metadata keys: {list(dataset.metadata.keys())}")
        
        # Test basic dataset access
        print(f"\n--- Testing Dataset Access ---")
        oni_dataset = ONIDataset(dataset)
        
        # Get a sample
        sample_tensor, sample_labels = oni_dataset[0]
        print(f"Sample spatial shape: {sample_tensor.spatial.shape}")
        print(f"Sample global features shape: {sample_tensor.global_features.shape}")
        print(f"Sample labels: {list(sample_labels.keys())}")
        
        # Test data splitting
        train_ds, val_ds, test_ds = oni_dataset.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        print(f"Split sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        
        # Show framework availability
        frameworks = get_framework_info()
        print(f"\n--- Available ML Frameworks ---")
        for framework, available in frameworks.items():
            status = "✓" if available else "✗"
            print(f"{status} {framework}")
        
        # Test framework-specific loaders
        print(f"\n--- Testing Framework Loaders ---")
        
        # Test scikit-learn format (always available)
        try:
            from data.datasets import create_sklearn_data
            X, y = create_sklearn_data(train_ds, feature_type='combined', target_type='survival_status')
            print(f"✓ Scikit-learn format - X: {X.shape}, y: {y.shape}")
        except Exception as e:
            print(f"✗ Scikit-learn format error: {e}")
        
        # Test PyTorch loader (if available)
        if frameworks['pytorch']:
            try:
                pytorch_loader = create_data_loader(dataset, framework='pytorch', batch_size=2)
                print(f"✓ PyTorch DataLoader created with {len(pytorch_loader)} batches")
            except Exception as e:
                print(f"✗ PyTorch loader error: {e}")
        
        # Test TensorFlow dataset (if available)
        if frameworks['tensorflow']:
            try:
                tf_dataset = create_data_loader(dataset, framework='tensorflow', batch_size=2)
                print(f"✓ TensorFlow Dataset created")
            except Exception as e:
                print(f"✗ TensorFlow dataset error: {e}")
        
        # Run comprehensive demo
        print(f"\n--- Running Comprehensive Demo ---")
        demo_data_loaders(dataset)
        
        print(f"\n--- Demo Complete ---")
        print(f"Dataset files saved in: {output_dir}")
        print(f"You can now use this dataset for ML training!")
        
    except Exception as e:
        print(f"Error building dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())