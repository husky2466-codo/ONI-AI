# Test Fixtures

This directory contains test fixtures for the ONI AI Agent project.

## Sample Save Files

The `sample_saves/` directory contains ONI save files used for testing:

### Real ONI Save Files
- `test_colony_main.sav` - Main colony save file from the ONI_BOT_LEARN directory
- `test_colony_cycle_181.sav` - Colony save from cycle 181
- `test_colony_cycle_190.sav` - Colony save from cycle 190

### Test Edge Cases
- `empty_file.sav` - Empty file to test empty file handling
- `corrupted_file.sav` - Corrupted text file to test error handling

## Usage

These files are used by the unit tests in `tests/unit/test_parsers.py`, specifically the `TestRealSaveFiles` test class. The tests verify that:

1. Real ONI save files can be parsed successfully (or create appropriate mock states)
2. Empty files are handled gracefully
3. Corrupted files are handled gracefully
4. Batch processing works with mixed file types
5. The main interface function works with both string and Path objects

## Notes

- Tests will skip if sample files are not found
- Tests will skip if Node.js dependencies are not available
- When dependencies are missing, the parser creates mock states for demonstration
- All tests are designed to be robust and handle missing dependencies gracefully