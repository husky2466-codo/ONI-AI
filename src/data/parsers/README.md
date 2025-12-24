# ONI Save Parser

This module provides a Python wrapper for the [oni-save-parser](https://github.com/RoboPhred/oni-save-parser) JavaScript library, enabling extraction of structured game state data from Oxygen Not Included save files for machine learning processing.

## Overview

The ONI Save Parser bridges the gap between the existing JavaScript parsing library and the Python-based ML pipeline. It extracts key game state information including:

- Grid tiles with material states, elements, and temperatures
- Duplicant information (position, health, stress, skills)
- Building locations and operational status
- Resource counts and game progression data
- World dimensions and metadata

## Architecture

```
Python Application
       ↓
ONISaveParser (Python)
       ↓
subprocess call
       ↓
oni_parser_bridge.js (Node.js)
       ↓
oni-save-parser library
       ↓
ONI Save File (.sav)
```

## Installation

### Prerequisites

1. **Node.js**: Required to run the JavaScript parsing library
   ```bash
   # Install Node.js from https://nodejs.org/
   node --version  # Should show v16+ 
   ```

2. **oni-save-parser**: JavaScript library for parsing ONI saves
   ```bash
   npm install oni-save-parser
   ```

3. **Python dependencies**: NumPy for array processing
   ```bash
   pip install numpy
   ```

### Setup

The parser automatically creates the necessary Node.js bridge script on first use. No additional setup is required.

## Usage

### Basic Usage

```python
from src.data.parsers import ONISaveParser

# Initialize the parser
parser = ONISaveParser()

# Parse a save file
game_state = parser.parse_save('path/to/save.sav')

# Access extracted data
print(f"World size: {game_state.world_size}")
print(f"Cycle: {game_state.cycle}")
print(f"Duplicants: {len(game_state.duplicants)}")
print(f"Buildings: {len(game_state.buildings)}")
```

### Validation

```python
# Validate a save file before parsing
is_valid, error_msg = parser.validate_save_file('path/to/save.sav')

if is_valid:
    game_state = parser.parse_save('path/to/save.sav')
else:
    print(f"Invalid save file: {error_msg}")
```

### Error Handling

```python
try:
    game_state = parser.parse_save('path/to/save.sav')
except FileNotFoundError:
    print("Save file not found")
except RuntimeError as e:
    print(f"Parsing failed: {e}")
```

## Data Structures

### GameState

The main data structure containing all extracted game information:

```python
@dataclass
class GameState:
    grid: np.ndarray              # (height, width, channels) - spatial data
    duplicants: List[Duplicant]   # Duplicant information
    buildings: List[Building]     # Building information  
    resources: Dict[str, float]   # Resource counts
    cycle: int                    # Game cycle number
    timestamp: float              # Game timestamp
    world_size: Tuple[int, int]   # World dimensions
    metadata: Dict[str, Any]      # Additional metadata
```

### Duplicant

Information about individual duplicants:

```python
@dataclass
class Duplicant:
    name: str                     # Duplicant name
    position: Tuple[float, float, float]  # 3D position
    stress_level: float           # Stress percentage
    health: float                 # Health percentage
    skills: Dict[str, int]        # Skill levels
    traits: List[str]             # Character traits
```

### Building

Information about buildings and structures:

```python
@dataclass
class Building:
    name: str                     # Building name
    position: Tuple[float, float, float]  # 3D position
    building_type: str            # Classified type (power, plumbing, etc.)
    operational: bool             # Operational status
    temperature: float            # Building temperature
```

## Grid Data Format

The spatial grid uses a multi-channel format optimized for ML processing with **refined tile-level extraction**:

- **Dimensions**: (height, width, channels)
- **Channels**:
  - 0: Solid material presence (0-1)
  - 1: Liquid material presence (0-1) 
  - 2: Gas material presence (0-1)
  - 3: Element ID (categorical, normalized)
  - 4: Temperature (normalized)
  - 5: Building presence (0-1)
  - 6: Duplicant presence (0-1)

### Refined Grid Extraction

The parser now supports **detailed tile-level data extraction** from ONI save files:

- **Detailed Mode**: When available, extracts actual cell data including:
  - Precise material states (solid/liquid/gas) based on element properties
  - Real temperature values from save file (converted from Kelvin to Celsius)
  - Element IDs and properties from the game's element system
  - Mass and insulation information per cell
  
- **Enhanced Placeholder Mode**: When detailed data is unavailable, generates realistic patterns:
  - Solid ground layers at bottom of world
  - Liquid pockets in appropriate locations
  - Temperature variations based on depth and location
  - Varied oxygen concentrations in gas areas

- **Quality Metadata**: GameState metadata includes extraction quality information:
  - `grid_data_available`: Whether detailed tile data was found
  - `grid_cells_extracted`: Number of cells with detailed data
  - `grid_elements_available`: Number of element types identified

## Building Classification

Buildings are automatically classified into categories:

- **power**: Generators, batteries, electrical wiring
- **plumbing**: Pumps, pipes, valves, filters
- **ventilation**: Vents, fans, air scrubbers
- **agriculture**: Farms, planters, hydroponic systems
- **living**: Beds, toilets, recreational facilities
- **research**: Research stations, computers, telescopes
- **storage**: Storage containers, lockers
- **infrastructure**: Doors, ladders, tiles, walls
- **other**: Unclassified buildings

## Performance Considerations

- **Memory Usage**: Large save files may require significant memory for grid processing
- **Processing Time**: Parsing time scales with save file size and complexity
- **Subprocess Overhead**: Each parse operation spawns a Node.js subprocess

## Error Handling

The parser includes comprehensive error handling:

- **Dependency Verification**: Checks for Node.js and oni-save-parser availability
- **File Validation**: Validates save files before parsing
- **Graceful Degradation**: Continues processing even if some objects fail to parse
- **Timeout Protection**: Prevents hanging on corrupted files

## Testing

Run the test suite to verify functionality:

```bash
# Unit tests
python -m pytest tests/unit/test_parsers.py -v

# Integration tests (requires Node.js and oni-save-parser)
python -m pytest tests/integration/test_parser_integration.py -v
```

## Troubleshooting

### Common Issues

1. **"Node.js not found"**
   - Install Node.js from https://nodejs.org/
   - Ensure `node` command is in PATH

2. **"oni-save-parser library not properly installed"**
   - Run `npm install oni-save-parser`
   - Verify installation with `node -e "require('oni-save-parser')"`

3. **"Parsing timed out"**
   - Save file may be corrupted or extremely large
   - Try with a smaller/different save file

4. **"Invalid save file"**
   - Save file may be from unsupported ONI version
   - Verify file is not corrupted

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

parser = ONISaveParser()
game_state = parser.parse_save('save.sav')
```

## Future Enhancements

- ~~Enhanced grid data extraction with actual material/element information~~ ✅ **Completed**
- Support for streaming large save files
- Caching mechanism for repeated parsing
- Direct integration with C# ONI modding tools
- Real-time save file monitoring

## Contributing

When contributing to the parser:

1. Maintain backward compatibility with existing GameState structure
2. Add comprehensive tests for new functionality
3. Update documentation for API changes
4. Follow the existing error handling patterns
5. Consider performance impact of changes