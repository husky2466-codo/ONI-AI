#!/usr/bin/env python3
"""
Example script demonstrating how to use the ONI save parser.

This script shows how to parse an ONI save file and extract game state data
for machine learning processing.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import parse_save


def main():
    """Demonstrate ONI save parser usage."""
    print("🔧 ONI Save Parser Example")
    print("=" * 50)
    
    try:
        # Check if a save file was provided
        if len(sys.argv) < 2:
            print("\n📝 Usage: python parse_save_example.py <path_to_save_file.sav>")
            print("\n🎯 Parser is ready for use!")
            print("   Example: parse_save('Colony001.sav')")
            return
        
        save_file_path = sys.argv[1]
        
        # Parse the save file using the main interface
        print(f"\n⚙️ Parsing save file: {save_file_path}")
        game_state = parse_save(save_file_path)
        
        # Display extracted information
        print("\n📊 Extracted Game State Information:")
        print(f"   🌍 World Size: {game_state.world_size[0]} x {game_state.world_size[1]}")
        print(f"   🔄 Cycle: {game_state.cycle}")
        print(f"   👥 Duplicants: {len(game_state.duplicants)}")
        print(f"   🏗️ Buildings: {len(game_state.buildings)}")
        print(f"   📈 Grid Shape: {game_state.grid.shape}")
        
        # Show duplicant details
        if game_state.duplicants:
            print("\n👥 Duplicant Details:")
            for i, dup in enumerate(game_state.duplicants[:3]):  # Show first 3
                print(f"   {i+1}. {dup.name}")
                print(f"      Position: {dup.position}")
                print(f"      Health: {dup.health:.1f}")
                print(f"      Stress: {dup.stress_level:.1f}")
                if dup.skills:
                    print(f"      Skills: {list(dup.skills.keys())}")
        
        # Show building details
        if game_state.buildings:
            print("\n🏗️ Building Details:")
            building_types = {}
            for building in game_state.buildings:
                building_types[building.building_type] = building_types.get(building.building_type, 0) + 1
            
            for building_type, count in building_types.items():
                print(f"   {building_type.title()}: {count}")
        
        # Show resource information
        print("\n📦 Resources:")
        for resource, amount in game_state.resources.items():
            print(f"   {resource.replace('_', ' ').title()}: {amount:.1f}")
        
        print("\n🎯 Parsing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())