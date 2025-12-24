"""
Dataset building and loading utilities for ONI AI Agent.
"""

from .dataset_builder import build_dataset, Dataset, DatasetBuilder
from .data_loaders import (
    ONIDataset, create_data_loader, create_sklearn_data, 
    get_framework_info, StandardTransform, demo_data_loaders
)

# Conditional imports for framework-specific loaders
try:
    from .data_loaders import create_pytorch_dataloader, ONIPyTorchDataset
    __all__ = ['build_dataset', 'Dataset', 'DatasetBuilder', 'ONIDataset', 
               'create_data_loader', 'create_pytorch_dataloader', 'ONIPyTorchDataset',
               'create_sklearn_data', 'get_framework_info', 'StandardTransform', 'demo_data_loaders']
except ImportError:
    __all__ = ['build_dataset', 'Dataset', 'DatasetBuilder', 'ONIDataset', 
               'create_data_loader', 'create_sklearn_data', 'get_framework_info', 
               'StandardTransform', 'demo_data_loaders']

try:
    from .data_loaders import create_tensorflow_dataset
    __all__.append('create_tensorflow_dataset')
except ImportError:
    pass