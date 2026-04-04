# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ONI-AI is a reinforcement learning agent for playing Oxygen Not Included (ONI), a complex colony simulation game. The project follows a 5-phase roadmap:

| Phase | Name | Status |
|-------|------|--------|
| 1 | Data Extraction | Complete |
| 2 | Environment Design | Ready to start |
| 3 | Baseline Models | Pending |
| 4 | RL Training | Pending |
| 5 | Hierarchical Planning | In progress (Task 5.1) |

See `.kiro/specs/oni-ai/tasks.md` for the detailed implementation roadmap.

## Commands

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run a single test file
pytest tests/unit/test_mini_oni_environment.py -v

# Run a specific test class or method
pytest tests/unit/test_mini_oni_environment.py::TestMiniONIEnvironment::test_reset -v

# Run with property-based test statistics
pytest tests/ --hypothesis-show-statistics

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/

# Sort imports
isort src/ tests/
```

### Installation

```bash
pip install -r requirements.txt
npm install
```

### Running Examples

```bash
python examples/mini_oni_environment_demo.py
python examples/objective_system_demo.py
python examples/parse_save_example.py
```

## Architecture

### Source Layout

```
src/
├── data/           # Phase 1: Data extraction pipeline
│   ├── parsers/    # ONI save file parsing (Python wrapping JS library)
│   ├── preprocessors/  # GameState → StateTensor conversion
│   └── datasets/   # Dataset building (NPZ + Parquet + JSON)
├── environments/   # Phase 2: RL training environment
│   └── mini_oni/   # Simplified ONI (64x64 max, 3 duplicants, 100 cycles)
└── agents/         # Phase 5: Agent implementations
    └── hierarchical/  # 3-level hierarchical planning agent
```

### Key Data Flow

1. **Parse**: `oni_parser_bridge.js` (Node subprocess) → `ONISaveParser` → raw `GameState`
2. **Preprocess**: `StatePreprocessor` → `StateTensor` (64x64x7 spatial array + 64-dim global features)
3. **Dataset**: `DatasetBuilder` → NPZ/Parquet/JSON files for ML training
4. **Environment**: `MiniONIEnvironment` (gym-like) → observation tensors + rewards
5. **Agent**: `HierarchicalAgent` → 3-level decisions (every 20 cycles / 5 steps / continuous)

### Parser Bridge

The save file parser is a JavaScript library (`oni-save-parser`) called via Node.js subprocess. `oni_parser_bridge.js` is the JS bridge, and `src/data/parsers/oni_save_parser.py` is the Python wrapper. The parser falls back to mock data for corrupted saves.

### Mini-ONI Environment

Constraints enforced by design:
- Map: max 64×64 tiles
- Episode: max 100 cycles
- Duplicants: 3 (starter setup)
- Buildings: 10–15 essential types only

Action space uses `ActionType` enum: `PlaceBuilding`, `Dig`, `Priority`, `DuplicantAssign`, `NoOp` — all position-based `(x, y)`.

Objectives are three-tier:
- Primary: Oxygen > 500g/tile
- Secondary: Polluted water routing
- Tertiary: Duplicant happiness > 50%

### Hierarchical Agent (Phase 5)

Three-level architecture with temporal abstraction:
- **High-level planner**: 5 abstract goals, decides every 20 cycles
- **Mid-level controller**: 15 subgoals, decides every 5 steps
- **Low-level executor**: Primitive actions, continuous

`HierarchicalCoordinator` manages inter-level communication. `HierarchicalIntrinsicRewards` provides bonus rewards (subgoal: +10, goal: +50, progress shaping: 5×progress).

Configuration via `HierarchicalConfig` dataclass in `src/agents/hierarchical/config.py`.

## Code Conventions

- **Type hints** required on all function signatures
- **Dataclasses** for structured data (`GameState`, `StateTensor`, `Dataset`, etc.)
- **Google-style docstrings**
- **Import order**: stdlib → third-party → project → relative
- **Naming**: `snake_case` for files and functions

## Specifications

The `.kiro/` directory contains authoritative project specs:
- `.kiro/steering/product.md` — product vision and success metrics
- `.kiro/steering/tech.md` — technology decisions
- `.kiro/specs/oni-ai/requirements.md` — detailed requirements
- `.kiro/specs/oni-ai/design.md` — technical design
- `.kiro/specs/oni-ai/tasks.md` — implementation roadmap with task IDs

When implementing new tasks, reference the task ID (e.g., Task 5.1) in commit messages.

## Training Hardware

Target training hardware is the DGX Spark at `10.0.0.69` (accessible via `dgx1-ssh` MCP). Multi-GPU training is planned for Phases 3–5.
