#!/usr/bin/env python3
"""
Demonstration of enhanced error handling in the ONI Save Parser.

This script shows how the parser gracefully handles various types of corrupted
or problematic save files by creating appropriate mock states.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.parsers.oni_save_parser import ONISaveParser


def create_test_files():
    """Create various test files to demonstrate error handling."""
    temp_dir = tempfile.mkdtemp()
    
    # Create different types of problematic files
    files = {}
    
    # Empty file
    empty_file = os.path.join(temp_dir, 'empty.sav')
    with open(empty_file, 'wb') as f:
        pass  # Create empty file
    files['empty'] = empty_file
    
    # Very small file (too small to be valid)
    small_file = os.path.join(temp_dir, 'too_small.sav')
    with open(small_file, 'wb') as f:
        f.write(b'x')  # Just one byte
    files['small'] = small_file
    
    # Corrupted file with random data
    corrupted_file = os.path.join(temp_dir, 'corrupted.sav')
    with open(corrupted_file, 'wb') as f:
        f.write(b'This is not a valid ONI save file!' * 100)
    files['corrupted'] = corrupted_file
    
    # File with cycle information in name
    cycle_file = os.path.join(temp_dir, 'Colony Cycle 42.sav')
    with open(cycle_file, 'wb') as f:
        f.write(b'fake save data with cycle info')
    files['cycle'] = cycle_file
    
    # Non-existent file
    files['missing'] = os.path.join(temp_dir, 'does_not_exist.sav')
    
    return temp_dir, files


def demonstrate_error_handling():
    """Demonstrate the enhanced error handling capabilities."""
    print("ONI Save Parser - Enhanced Error Handling Demonstration")
    print("=" * 60)
    
    try:
        parser = ONISaveParser()
        print("✅ Parser initialized successfully")
    except RuntimeError as e:
        print(f"⚠️  Parser initialization failed: {e}")
        print("This is expected if Node.js or oni-save-parser is not installed.")
        print("The parser will still demonstrate error handling with mock states.")
        parser = None
    
    temp_dir, files = create_test_files()
    
    print(f"\nCreated test files in: {temp_dir}")
    print("\nTesting different error scenarios:")
    print("-" * 40)
    
    test_cases = [
        ('empty', "Empty save file"),
        ('small', "File too small to be valid"),
        ('corrupted', "Corrupted save file"),
        ('cycle', "File with cycle info in name"),
        ('missing', "Non-existent file")
    ]
    
    if parser:
        for file_type, description in test_cases:
            file_path = files[file_type]
            print(f"\n{description}:")
            print(f"  File: {os.path.basename(file_path)}")
            
            try:
                # First, validate the file
                is_valid, error_msg = parser.validate_save_file(file_path)
                print(f"  Validation: {'✅ Valid' if is_valid else f'❌ Invalid - {error_msg}'}")
                
                # Then try to parse it
                game_state = parser.parse_save(file_path)
                
                if game_state.metadata.get('mock', False):
                    reason = game_state.metadata.get('reason', 'unknown')
                    recovery_mode = game_state.metadata.get('recovery_mode', False)
                    print(f"  Result: 🔧 Mock state created (reason: {reason})")
                    print(f"  Recovery mode: {'Yes' if recovery_mode else 'No'}")
                    print(f"  Duplicants: {len(game_state.duplicants)}")
                    print(f"  Buildings: {len(game_state.buildings)}")
                    print(f"  World size: {game_state.world_size}")
                    print(f"  Cycle: {game_state.cycle}")
                    print(f"  Oxygen: {game_state.resources.get('oxygen', 0):.1f}")
                else:
                    print(f"  Result: ✅ Successfully parsed real save data")
                    
            except FileNotFoundError:
                print(f"  Result: ❌ File not found (as expected)")
            except Exception as e:
                print(f"  Result: ❌ Unexpected error: {e}")
    
    # Demonstrate batch processing
    print(f"\n\nBatch Processing Demonstration:")
    print("-" * 40)
    
    if parser:
        all_files = list(files.values())
        
        # Get statistics without parsing
        stats = parser.get_parsing_statistics(all_files)
        print(f"Statistics for {stats['total_files']} files:")
        print(f"  Valid files: {stats['valid_files']}")
        print(f"  Invalid files: {stats['invalid_files']}")
        print(f"  Missing files: {stats['missing_files']}")
        print(f"  Error types: {dict(stats['error_types'])}")
        
        if stats['file_sizes']:
            print(f"  Average file size: {stats['avg_file_size']:.0f} bytes")
            print(f"  Size range: {stats['min_file_size']}-{stats['max_file_size']} bytes")
        
        # Batch parse with error recovery
        print(f"\nBatch parsing results:")
        results = parser.parse_save_batch(all_files, skip_corrupted=False)
        
        successful = 0
        mock_states = 0
        failed = 0
        
        for file_path, game_state, error_msg in results:
            filename = os.path.basename(file_path)
            if game_state:
                if game_state.metadata.get('mock', False):
                    mock_states += 1
                    print(f"  {filename}: 🔧 Mock state (reason: {game_state.metadata.get('reason')})")
                else:
                    successful += 1
                    print(f"  {filename}: ✅ Parsed successfully")
            else:
                failed += 1
                print(f"  {filename}: ❌ Failed - {error_msg}")
        
        print(f"\nBatch Summary:")
        print(f"  Successfully parsed: {successful}")
        print(f"  Mock states created: {mock_states}")
        print(f"  Failed: {failed}")
        print(f"  Total recovery rate: {((successful + mock_states) / len(results)) * 100:.1f}%")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\nCleaned up temporary files.")
    
    print(f"\n" + "=" * 60)
    print("Demonstration complete!")
    print("\nKey features demonstrated:")
    print("• Graceful handling of corrupted save files")
    print("• Creation of appropriate mock states for different failure types")
    print("• Comprehensive validation before parsing")
    print("• Batch processing with error recovery")
    print("• Detailed error categorization and statistics")
    print("• Recovery modes for different types of corruption")


if __name__ == '__main__':
    demonstrate_error_handling()