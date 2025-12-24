"""
Data Preprocessing Demo for ONI AI Agent.

This script demonstrates the data preprocessing pipeline that converts
parsed ONI save files into ML-ready tensor formats.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import parse_save, preprocess_state


def main():
    """Demonstrate data preprocessing functionality."""
    print("ONI AI Agent - Data Preprocessing Demo")
    print("=" * 50)
    
    # Use a sample save file for demonstration
    save_files = [
        "tests/fixtures/sample_saves/test_colony_main.sav",
        "tests/fixtures/sample_saves/test_colony_cycle_181.sav",
        "tests/fixtures/sample_saves/test_colony_cycle_190.sav"
    ]
    
    for save_file in save_files:
        if Path(save_file).exists():
            print(f"\nProcessing: {save_file}")
            demonstrate_preprocessing(save_file)
            break
    else:
        print("\nNo sample save files found. Creating mock demonstration...")
        demonstrate_with_mock_data()


def demonstrate_preprocessing(save_file_path: str):
    """Demonstrate preprocessing with a real save file."""
    try:
        # Step 1: Parse the save file
        print("Step 1: Parsing save file...")
        game_state = parse_save(save_file_path)
        
        print(f"  ✓ Parsed successfully")
        print(f"  ✓ World size: {game_state.world_size}")
        print(f"  ✓ Cycle: {game_state.cycle}")
        print(f"  ✓ Duplicants: {len(game_state.duplicants)}")
        print(f"  ✓ Buildings: {len(game_state.buildings)}")
        print(f"  ✓ Grid shape: {game_state.grid.shape}")
        
        # Step 2: Preprocess to ML format
        print("\nStep 2: Preprocessing to ML format...")
        state_tensor = preprocess_state(game_state, target_size=(64, 64))
        
        print(f"  ✓ Spatial tensor shape: {state_tensor.spatial.shape}")
        print(f"  ✓ Global features shape: {state_tensor.global_features.shape}")
        print(f"  ✓ Number of channels: {state_tensor.channels}")
        
        # Step 3: Analyze the preprocessed data
        print("\nStep 3: Analyzing preprocessed data...")
        analyze_spatial_tensor(state_tensor.spatial)
        analyze_global_features(state_tensor.global_features)
        
        # Step 4: Demonstrate different target sizes
        print("\nStep 4: Testing different target sizes...")
        for size in [(32, 32), (128, 128), (16, 16)]:
            tensor = preprocess_state(game_state, target_size=size)
            print(f"  ✓ Target size {size}: {tensor.spatial.shape}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Falling back to mock demonstration...")
        demonstrate_with_mock_data()


def demonstrate_with_mock_data():
    """Demonstrate preprocessing with mock data when no save files available."""
    from data.parsers.oni_save_parser import ONISaveParser
    
    print("Creating mock game state for demonstration...")
    
    # Create parser and generate mock state
    parser = ONISaveParser()
    game_state = parser._create_mock_game_state("demo.sav", reason="demonstration")
    
    print(f"  ✓ Mock state created")
    print(f"  ✓ World size: {game_state.world_size}")
    print(f"  ✓ Cycle: {game_state.cycle}")
    print(f"  ✓ Duplicants: {len(game_state.duplicants)}")
    print(f"  ✓ Buildings: {len(game_state.buildings)}")
    
    # Preprocess the mock data
    print("\nPreprocessing mock data...")
    state_tensor = preprocess_state(game_state, target_size=(64, 64))
    
    print(f"  ✓ Spatial tensor shape: {state_tensor.spatial.shape}")
    print(f"  ✓ Global features shape: {state_tensor.global_features.shape}")
    
    # Analyze the results
    analyze_spatial_tensor(state_tensor.spatial)
    analyze_global_features(state_tensor.global_features)


def analyze_spatial_tensor(spatial_tensor: np.ndarray):
    """Analyze the spatial tensor channels."""
    print("\n  Spatial Tensor Analysis:")
    
    channel_names = [
        "Solid", "Liquid", "Gas", "Element ID", 
        "Temperature", "Buildings", "Duplicants"
    ]
    
    for i, name in enumerate(channel_names):
        channel = spatial_tensor[:, :, i]
        non_zero_count = np.count_nonzero(channel)
        mean_val = np.mean(channel)
        max_val = np.max(channel)
        min_val = np.min(channel)
        
        print(f"    {name:12}: {non_zero_count:5d} non-zero cells, "
              f"range [{min_val:.3f}, {max_val:.3f}], mean {mean_val:.3f}")


def analyze_global_features(global_features: np.ndarray):
    """Analyze the global features vector."""
    print("\n  Global Features Analysis:")
    
    # Resource features (0-9)
    resource_features = global_features[0:10]
    print(f"    Resources (0-9):     mean {np.mean(resource_features):.3f}, "
          f"max {np.max(resource_features):.3f}")
    
    # Duplicant features (10-19)
    duplicant_features = global_features[10:20]
    print(f"    Duplicants (10-19):  mean {np.mean(duplicant_features):.3f}, "
          f"max {np.max(duplicant_features):.3f}")
    
    # Building features (20-30)
    building_features = global_features[20:31]
    print(f"    Buildings (20-30):   mean {np.mean(building_features):.3f}, "
          f"max {np.max(building_features):.3f}")
    
    # Cycle features (31-35)
    cycle_features = global_features[31:36]
    print(f"    Cycle info (31-35):  mean {np.mean(cycle_features):.3f}, "
          f"max {np.max(cycle_features):.3f}")
    
    # Non-zero feature count
    non_zero_count = np.count_nonzero(global_features)
    print(f"    Total non-zero features: {non_zero_count}/64")


def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple save files."""
    print("\n" + "=" * 50)
    print("Batch Processing Demonstration")
    print("=" * 50)
    
    # Find all available save files
    save_dir = Path("tests/fixtures/sample_saves")
    if save_dir.exists():
        save_files = list(save_dir.glob("*.sav"))
        print(f"Found {len(save_files)} save files")
        
        processed_states = []
        
        for save_file in save_files[:3]:  # Process first 3 files
            try:
                print(f"\nProcessing {save_file.name}...")
                game_state = parse_save(str(save_file))
                state_tensor = preprocess_state(game_state, target_size=(32, 32))
                processed_states.append(state_tensor)
                print(f"  ✓ Success: {state_tensor.spatial.shape}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        if processed_states:
            print(f"\nBatch processing complete: {len(processed_states)} states processed")
            
            # Demonstrate stacking for batch training
            spatial_batch = np.stack([s.spatial for s in processed_states])
            global_batch = np.stack([s.global_features for s in processed_states])
            
            print(f"Batch spatial tensor: {spatial_batch.shape}")
            print(f"Batch global features: {global_batch.shape}")
    else:
        print("No sample save directory found - skipping batch demo")


if __name__ == "__main__":
    main()
    demonstrate_batch_processing()
    
    print("\n" + "=" * 50)
    print("Demo complete! The data preprocessing pipeline is ready.")
    print("Key features implemented:")
    print("  ✓ Multi-channel spatial tensor (7 channels)")
    print("  ✓ Global features vector (64 dimensions)")
    print("  ✓ Temperature and element ID normalization")
    print("  ✓ Building and duplicant encoding")
    print("  ✓ Flexible target sizing")
    print("  ✓ Comprehensive data validation")
    print("=" * 50)