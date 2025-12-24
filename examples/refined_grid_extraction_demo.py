#!/usr/bin/env python3
"""
Demonstration of refined grid-level tile extraction in ONI Save Parser.

This script shows the improvements made to grid extraction, including:
1. Detailed tile-level data extraction from save files when available
2. Enhanced placeholder data when detailed data is not available
3. Proper material state classification (solid/liquid/gas)
4. Temperature and element information extraction
5. Metadata about grid extraction quality
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.parsers.oni_save_parser import ONISaveParser
import numpy as np


def demonstrate_refined_grid_extraction():
    """Demonstrate the refined grid extraction capabilities."""
    print("=== ONI Save Parser: Refined Grid-Level Tile Extraction Demo ===\n")
    
    try:
        # Initialize the parser
        parser = ONISaveParser()
        print("✓ ONI Save Parser initialized successfully")
        
        # Look for sample save files
        fixtures_dir = Path(__file__).parent.parent / 'tests' / 'fixtures' / 'sample_saves'
        sample_files = [
            fixtures_dir / 'test_colony_main.sav',
            fixtures_dir / 'test_colony_cycle_181.sav',
            fixtures_dir / 'test_colony_cycle_190.sav'
        ]
        
        # Find an existing sample file
        sample_file = None
        for file_path in sample_files:
            if file_path.exists():
                sample_file = file_path
                break
        
        if sample_file:
            print(f"✓ Found sample save file: {sample_file.name}")
            
            # Parse the save file
            print("\n--- Parsing Save File ---")
            game_state = parser.parse_save(str(sample_file))
            
            # Display basic information
            print(f"Cycle: {game_state.cycle}")
            print(f"World Size: {game_state.world_size}")
            print(f"Duplicants: {len(game_state.duplicants)}")
            print(f"Buildings: {len(game_state.buildings)}")
            print(f"Grid Shape: {game_state.grid.shape}")
            
            # Display grid extraction metadata
            print("\n--- Grid Extraction Quality ---")
            metadata = game_state.metadata
            print(f"Real Parse (not mock): {metadata.get('real_parse', False)}")
            print(f"Detailed Grid Data Available: {metadata.get('grid_data_available', False)}")
            print(f"Grid Cells Extracted: {metadata.get('grid_cells_extracted', 0)}")
            print(f"Grid Elements Available: {metadata.get('grid_elements_available', 0)}")
            
            # Analyze grid content
            print("\n--- Grid Content Analysis ---")
            grid = game_state.grid
            
            # Count material states
            solid_cells = np.sum(grid[:, :, 0] > 0.5)
            liquid_cells = np.sum(grid[:, :, 1] > 0.5)
            gas_cells = np.sum(grid[:, :, 2] > 0.5)
            total_cells = grid.shape[0] * grid.shape[1]
            
            print(f"Solid cells: {solid_cells:,} ({solid_cells/total_cells*100:.1f}%)")
            print(f"Liquid cells: {liquid_cells:,} ({liquid_cells/total_cells*100:.1f}%)")
            print(f"Gas cells: {gas_cells:,} ({gas_cells/total_cells*100:.1f}%)")
            
            # Temperature analysis
            temp_channel = grid[:, :, 4]
            min_temp = np.min(temp_channel)
            max_temp = np.max(temp_channel)
            avg_temp = np.mean(temp_channel)
            
            print(f"\nTemperature Range: {min_temp:.1f}°C to {max_temp:.1f}°C")
            print(f"Average Temperature: {avg_temp:.1f}°C")
            
            # Building and duplicant presence
            building_cells = np.sum(grid[:, :, 5] > 0.5)
            duplicant_cells = np.sum(grid[:, :, 6] > 0.5)
            
            print(f"\nCells with Buildings: {building_cells}")
            print(f"Cells with Duplicants: {duplicant_cells}")
            
            # Sample some specific grid locations
            print("\n--- Sample Grid Locations ---")
            height, width = game_state.world_size
            sample_locations = [
                (width//4, height//4),      # Upper left quadrant
                (3*width//4, height//4),    # Upper right quadrant
                (width//2, 3*height//4),    # Lower center
            ]
            
            for x, y in sample_locations:
                if 0 <= x < width and 0 <= y < height:
                    cell = grid[y, x, :]
                    material_state = "Gas" if cell[2] > 0.5 else ("Liquid" if cell[1] > 0.5 else "Solid")
                    element_id = int(cell[3] * 255)
                    temperature = cell[4]
                    has_building = cell[5] > 0.5
                    has_duplicant = cell[6] > 0.5
                    
                    print(f"  ({x:3d}, {y:3d}): {material_state:6s} | Element {element_id:3d} | {temperature:5.1f}°C | Building: {has_building} | Duplicant: {has_duplicant}")
            
        else:
            print("⚠ No sample save files found. Demonstrating with mock data...")
            
            # Create a mock game state to show the enhanced placeholder functionality
            mock_file_path = "mock_save.sav"
            game_state = parser._create_mock_game_state(mock_file_path, reason="demonstration")
            
            print(f"\n--- Mock Game State (Enhanced Placeholders) ---")
            print(f"World Size: {game_state.world_size}")
            print(f"Duplicants: {len(game_state.duplicants)}")
            print(f"Buildings: {len(game_state.buildings)}")
            print(f"Grid Shape: {game_state.grid.shape}")
            print(f"Mock Reason: {game_state.metadata.get('reason', 'unknown')}")
            
            # Analyze mock grid content
            grid = game_state.grid
            solid_cells = np.sum(grid[:, :, 0] > 0.5)
            liquid_cells = np.sum(grid[:, :, 1] > 0.5)
            gas_cells = np.sum(grid[:, :, 2] > 0.5)
            total_cells = grid.shape[0] * grid.shape[1]
            
            print(f"\nMock Grid Analysis:")
            print(f"Solid cells: {solid_cells:,} ({solid_cells/total_cells*100:.1f}%)")
            print(f"Liquid cells: {liquid_cells:,} ({liquid_cells/total_cells*100:.1f}%)")
            print(f"Gas cells: {gas_cells:,} ({gas_cells/total_cells*100:.1f}%)")
        
        print("\n--- Refinement Summary ---")
        print("✓ Grid extraction now supports detailed tile-level data from save files")
        print("✓ Proper material state classification (solid/liquid/gas)")
        print("✓ Temperature and element information extraction")
        print("✓ Enhanced placeholder data with realistic patterns when detailed data unavailable")
        print("✓ Metadata tracking of grid extraction quality")
        print("✓ Backward compatibility with existing code")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("\nThis may be due to missing dependencies (Node.js, oni-save-parser)")
        print("The parser will still work with mock data for development purposes.")


def demonstrate_grid_extraction_comparison():
    """Compare old placeholder vs new refined extraction."""
    print("\n=== Grid Extraction Comparison ===\n")
    
    try:
        parser = ONISaveParser()
        world_size = (64, 64)
        duplicants = []
        buildings = []
        
        # Test with detailed data
        print("--- With Detailed Grid Data ---")
        detailed_grid_data = {
            'hasDetailedData': True,
            'cells': [
                {'x': 10, 'y': 20, 'element': 1, 'temperature': 300.0, 'mass': 1000.0},
                {'x': 15, 'y': 25, 'element': 2, 'temperature': 273.0, 'mass': 500.0},
                {'x': 30, 'y': 40, 'element': 3, 'temperature': 350.0, 'mass': 2000.0}
            ],
            'elements': {
                1: {'id': 1, 'name': 'Oxygen', 'state': 'Gas'},
                2: {'id': 2, 'name': 'Water', 'state': 'Liquid'},
                3: {'id': 3, 'name': 'Sandstone', 'state': 'Solid'}
            }
        }
        
        detailed_grid = parser._create_enhanced_grid(world_size, duplicants, buildings, detailed_grid_data)
        
        print("Sample cells with detailed data:")
        for cell_data in detailed_grid_data['cells']:
            x, y = cell_data['x'], cell_data['y']
            cell = detailed_grid[y, x, :]
            material_state = "Gas" if cell[2] > 0.5 else ("Liquid" if cell[1] > 0.5 else "Solid")
            element_id = int(cell[3] * 255)
            temperature = cell[4]
            print(f"  ({x:2d}, {y:2d}): {material_state:6s} | Element {element_id:3d} | {temperature:5.1f}°C")
        
        # Test without detailed data (enhanced placeholders)
        print("\n--- With Enhanced Placeholders ---")
        placeholder_grid = parser._create_enhanced_grid(world_size, duplicants, buildings, None)
        
        # Show some sample locations from placeholder grid
        sample_locations = [(32, 10), (32, 50), (16, 55)]  # Top, middle, bottom
        print("Sample cells with enhanced placeholders:")
        for x, y in sample_locations:
            cell = placeholder_grid[y, x, :]
            material_state = "Gas" if cell[2] > 0.5 else ("Liquid" if cell[1] > 0.5 else "Solid")
            element_id = int(cell[3] * 255)
            temperature = cell[4]
            print(f"  ({x:2d}, {y:2d}): {material_state:6s} | Element {element_id:3d} | {temperature:5.1f}°C")
        
        print("\n✓ Refined extraction provides accurate data when available")
        print("✓ Enhanced placeholders provide realistic fallback data")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")


if __name__ == "__main__":
    demonstrate_refined_grid_extraction()
    demonstrate_grid_extraction_comparison()