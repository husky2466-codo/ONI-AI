# ONI AI Agent

An AI agent system capable of playing Oxygen Not Included (ONI), a complex colony simulation game. This project progresses through five phases: data extraction from save files, environment design, baseline model development, reinforcement learning training, and advanced hierarchical planning with LLM integration.

## Quick Start

```bash
# Install dependencies
pip install torch stable-baselines3 hypothesis pandas numpy matplotlib

# Install Node.js for save parser
npm install oni-save-parser

# Set up development tools
pip install pytest black flake8 mypy isort
```

## Project Structure

- **Phase 1**: Data Extraction Pipeline (3-4 weeks) - *✅ **100% COMPLETE** - ONI save parser, data preprocessor, and dataset builder all complete*
- **Phase 2**: Environment Design (5-6 weeks) - *Ready to begin*
- **Phase 3**: Baseline Models (4-7 weeks)  
- **Phase 4**: RL Training (6-7 weeks)
- **Phase 5**: Advanced Representations (5-7 weeks)

**Total Timeline**: 32-46 weeks (8-12 months)

## Core Objectives

- **Primary Goal**: Create an RL agent that can maintain colony survival by managing oxygen, temperature, resources, and duplicant happiness
- **Research Focus**: Hierarchical planning, multi-scale decision making, and LLM-assisted reasoning for complex strategy games
- **Technical Achievement**: Bridge between traditional game AI and modern deep learning approaches

## Current Status

**Phase 1 Status** ✅ - **100% COMPLETED**: Successfully implemented comprehensive data extraction pipeline including ONI save file parsing with enhanced error handling, multi-channel tensor data preprocessing, and complete dataset builder with ML framework integration. All Phase 1 components are production-ready.

## Implementation Details

### Completed Components
- **ONI Save Parser**: Full Python wrapper around JavaScript library with compatibility patches
- **GameState Data Model**: Complete dataclass structure for game state representation
- **Error Handling**: Comprehensive fallback system for corrupted/incompatible save files
- **Mock Data Generation**: Enhanced placeholder system for testing and development
- **Batch Processing**: Support for processing multiple save files with statistics
- **Data Preprocessor**: Multi-channel tensor format with normalization and validation
- **Dataset Builder**: Complete ML-ready dataset creation with multiple storage formats
- **ML Framework Integration**: PyTorch, TensorFlow, and scikit-learn compatibility

### Next Steps
- **Phase 1**: ✅ 100% Complete - All data extraction components implemented and tested
- **Phase 2**: Ready to begin - Environment design and Mini-ONI implementation
- Begin baseline model development

## Documentation

- [Requirements](.kiro/specs/oni-ai/requirements.md) - Detailed project requirements and success criteria
- [Design](.kiro/specs/oni-ai/design.md) - Technical architecture and component design
- [Tasks](.kiro/specs/oni-ai/tasks.md) - Implementation roadmap and timeline

## Technology Stack

- **Python 3.8+**: Primary language for ML/RL development
- **PyTorch**: Deep learning framework
- **Stable Baselines3**: Reinforcement learning algorithms
- **JavaScript/Node.js**: ONI save file parsing integration
- **DGX Spark**: High-performance computing optimization

## Success Metrics

- Colony survival rate across diverse scenarios
- Resource efficiency and sustainability
- Performance comparison against heuristic baselines
- Scalability to larger maps and longer episodes

## Getting Started

1. Review the [requirements document](/.kiro/specs/oni-ai/requirements.md) for detailed project scope
2. Check the [design document](/.kiro/specs/oni-ai/design.md) for technical architecture
3. Follow the [implementation tasks](/.kiro/specs/oni-ai/tasks.md) for development roadmap
4. See [research findings](research/oni-save-parser-evaluation.md) for ONI save parser evaluation results

## License

[Add license information]

## Contributing

[Add contribution guidelines]