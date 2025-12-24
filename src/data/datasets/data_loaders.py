"""
Data loading utilities for ML frameworks.

This module provides utilities for loading ONI datasets into various ML frameworks
including PyTorch, TensorFlow, and scikit-learn compatible formats.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import warnings

import numpy as np
import pandas as pd

from .dataset_builder import Dataset, StateTensor

try:
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    TorchDataLoader = None
    TorchDataset = None

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


class ONIDataset:
    """
    Framework-agnostic dataset wrapper for ONI data.
    
    This class provides a common interface for accessing ONI datasets
    regardless of the underlying ML framework being used.
    """
    
    def __init__(self, dataset: Dataset, transform: Optional[callable] = None):
        """
        Initialize the ONI dataset wrapper.
        
        Args:
            dataset: Built ONI dataset
            transform: Optional transform function to apply to samples
        """
        self.dataset = dataset
        self.transform = transform
        
        # Load metadata once
        self.metadata = dataset.metadata
        self.num_samples = dataset.num_samples
        
        # Cache for loaded data
        self._spatial_tensors = None
        self._global_features = None
        self._tabular_data = None
        self._labels = None
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, index: Union[int, slice, List[int]]) -> Union[Tuple[StateTensor, Dict[str, Any]], List[Tuple[StateTensor, Dict[str, Any]]]]:
        """
        Get sample(s) from the dataset.
        
        Args:
            index: Sample index, slice, or list of indices
            
        Returns:
            Single sample or list of samples as (StateTensor, labels) tuples
        """
        if isinstance(index, int):
            return self._get_single_sample(index)
        elif isinstance(index, slice):
            indices = range(*index.indices(self.num_samples))
            return [self._get_single_sample(i) for i in indices]
        elif isinstance(index, list):
            return [self._get_single_sample(i) for i in index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")
    
    def _get_single_sample(self, index: int) -> Tuple[StateTensor, Dict[str, Any]]:
        """Get a single sample from the dataset."""
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Sample index {index} out of range [0, {self.num_samples})")
        
        # Lazy load data
        if self._spatial_tensors is None:
            self._spatial_tensors = self.dataset.load_tensors()
        if self._global_features is None:
            self._global_features = self.dataset.load_global_features()
        if self._tabular_data is None:
            self._tabular_data = self.dataset.load_tabular_data()
        if self._labels is None:
            self._labels = self.dataset.load_labels()
        
        # Create StateTensor
        sample_metadata = self._tabular_data.iloc[index].to_dict()
        state_tensor = StateTensor(
            spatial=self._spatial_tensors[index],
            global_features=self._global_features[index],
            metadata=sample_metadata
        )
        
        # Extract labels for this sample
        sample_labels = {}
        for key, values in self._labels.items():
            if isinstance(values, list) and len(values) > index:
                sample_labels[key] = values[index]
            else:
                sample_labels[key] = values  # Scalar value
        
        # Apply transform if provided
        if self.transform:
            state_tensor, sample_labels = self.transform(state_tensor, sample_labels)
        
        return state_tensor, sample_labels
    
    def get_spatial_data(self) -> np.ndarray:
        """Get all spatial tensor data as numpy array."""
        if self._spatial_tensors is None:
            self._spatial_tensors = self.dataset.load_tensors()
        return self._spatial_tensors
    
    def get_global_features(self) -> np.ndarray:
        """Get all global features as numpy array."""
        if self._global_features is None:
            self._global_features = self.dataset.load_global_features()
        return self._global_features
    
    def get_tabular_data(self) -> pd.DataFrame:
        """Get tabular features as pandas DataFrame."""
        if self._tabular_data is None:
            self._tabular_data = self.dataset.load_tabular_data()
        return self._tabular_data
    
    def get_labels(self, label_type: Optional[str] = None) -> Union[Dict[str, Any], np.ndarray]:
        """
        Get labels from the dataset.
        
        Args:
            label_type: Specific label type to return, or None for all labels
            
        Returns:
            Dictionary of all labels or numpy array for specific label type
        """
        if self._labels is None:
            self._labels = self.dataset.load_labels()
        
        if label_type is None:
            return self._labels
        elif label_type in self._labels:
            return np.array(self._labels[label_type])
        else:
            raise KeyError(f"Label type '{label_type}' not found. Available: {list(self._labels.keys())}")
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, 
              random_seed: Optional[int] = None) -> Tuple['ONIDataset', 'ONIDataset', 'ONIDataset']:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random indices
        indices = np.random.permutation(self.num_samples)
        
        # Calculate split points
        train_end = int(train_ratio * self.num_samples)
        val_end = train_end + int(val_ratio * self.num_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create subset datasets
        train_dataset = ONISubset(self, train_indices)
        val_dataset = ONISubset(self, val_indices)
        test_dataset = ONISubset(self, test_indices)
        
        return train_dataset, val_dataset, test_dataset


class ONISubset(ONIDataset):
    """Subset of an ONI dataset with specific indices."""
    
    def __init__(self, parent_dataset: ONIDataset, indices: np.ndarray):
        """
        Initialize dataset subset.
        
        Args:
            parent_dataset: Parent ONI dataset
            indices: Indices to include in this subset
        """
        self.parent_dataset = parent_dataset
        self.indices = indices
        self.dataset = parent_dataset.dataset
        self.transform = parent_dataset.transform
        self.metadata = parent_dataset.metadata
        self.num_samples = len(indices)
        
        # Share cached data with parent
        self._spatial_tensors = None
        self._global_features = None
        self._tabular_data = None
        self._labels = None
    
    def _get_single_sample(self, index: int) -> Tuple[StateTensor, Dict[str, Any]]:
        """Get a single sample using parent dataset with mapped index."""
        if index < 0 or index >= len(self.indices):
            raise IndexError(f"Subset index {index} out of range [0, {len(self.indices)})")
        
        parent_index = self.indices[index]
        return self.parent_dataset._get_single_sample(parent_index)
    
    def get_spatial_data(self) -> np.ndarray:
        """Get spatial tensor data for this subset."""
        if self._spatial_tensors is None:
            parent_spatial = self.parent_dataset.get_spatial_data()
            self._spatial_tensors = parent_spatial[self.indices]
        return self._spatial_tensors
    
    def get_global_features(self) -> np.ndarray:
        """Get global features for this subset."""
        if self._global_features is None:
            parent_global = self.parent_dataset.get_global_features()
            self._global_features = parent_global[self.indices]
        return self._global_features
    
    def get_tabular_data(self) -> pd.DataFrame:
        """Get tabular features for this subset."""
        if self._tabular_data is None:
            parent_tabular = self.parent_dataset.get_tabular_data()
            self._tabular_data = parent_tabular.iloc[self.indices].reset_index(drop=True)
        return self._tabular_data
    
    def get_labels(self, label_type: Optional[str] = None) -> Union[Dict[str, Any], np.ndarray]:
        """Get labels for this subset."""
        if self._labels is None:
            parent_labels = self.parent_dataset.get_labels()
            self._labels = {}
            for key, values in parent_labels.items():
                if isinstance(values, list):
                    self._labels[key] = [values[i] for i in self.indices]
                else:
                    self._labels[key] = values  # Scalar values remain the same
        
        if label_type is None:
            return self._labels
        elif label_type in self._labels:
            return np.array(self._labels[label_type])
        else:
            raise KeyError(f"Label type '{label_type}' not found. Available: {list(self._labels.keys())}")


# PyTorch Integration
if TORCH_AVAILABLE:
    class ONIPyTorchDataset(TorchDataset):
        """PyTorch Dataset wrapper for ONI data."""
        
        def __init__(self, oni_dataset: ONIDataset, return_format: str = 'tensors'):
            """
            Initialize PyTorch dataset wrapper.
            
            Args:
                oni_dataset: ONI dataset to wrap
                return_format: Format to return data ('tensors', 'dict', or 'tuple')
            """
            self.oni_dataset = oni_dataset
            self.return_format = return_format
        
        def __len__(self) -> int:
            return len(self.oni_dataset)
        
        def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
            """Get sample in PyTorch tensor format."""
            state_tensor, labels = self.oni_dataset[index]
            
            # Convert to PyTorch tensors
            spatial_tensor = torch.from_numpy(state_tensor.spatial).float()
            global_tensor = torch.from_numpy(state_tensor.global_features).float()
            
            # Convert labels to tensors
            label_tensors = {}
            for key, value in labels.items():
                if isinstance(value, (int, float)):
                    label_tensors[key] = torch.tensor(value).float()
                elif isinstance(value, bool):
                    label_tensors[key] = torch.tensor(value).long()
                elif isinstance(value, dict):
                    # For resource counts, convert to tensor
                    if key == 'resource_counts':
                        resource_values = list(value.values())
                        label_tensors[key] = torch.tensor(resource_values).float()
                    else:
                        label_tensors[key] = value  # Keep as dict
                else:
                    label_tensors[key] = torch.tensor(value).float()
            
            if self.return_format == 'tensors':
                # Return spatial and global tensors separately
                return spatial_tensor, global_tensor
            elif self.return_format == 'dict':
                # Return everything as dictionary
                return {
                    'spatial': spatial_tensor,
                    'global': global_tensor,
                    'labels': label_tensors
                }
            elif self.return_format == 'tuple':
                # Return (inputs, targets) tuple
                inputs = torch.cat([spatial_tensor.flatten(), global_tensor])
                # Use survival status as primary target
                target = label_tensors.get('survival_status', torch.tensor(0.0))
                return inputs, target
            else:
                raise ValueError(f"Invalid return_format: {self.return_format}")
    
    def create_pytorch_dataloader(oni_dataset: ONIDataset, 
                                 batch_size: int = 32,
                                 shuffle: bool = True,
                                 num_workers: int = 0,
                                 return_format: str = 'tensors',
                                 **kwargs) -> TorchDataLoader:
        """
        Create PyTorch DataLoader for ONI dataset.
        
        Args:
            oni_dataset: ONI dataset to wrap
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            return_format: Format to return data ('tensors', 'dict', or 'tuple')
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            PyTorch DataLoader
        """
        pytorch_dataset = ONIPyTorchDataset(oni_dataset, return_format=return_format)
        return TorchDataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )

else:
    def create_pytorch_dataloader(*args, **kwargs):
        raise ImportError("PyTorch is not available. Install with: pip install torch")


# TensorFlow Integration
if TF_AVAILABLE:
    def create_tensorflow_dataset(oni_dataset: ONIDataset,
                                 batch_size: int = 32,
                                 shuffle: bool = True,
                                 buffer_size: int = 1000,
                                 return_format: str = 'tuple') -> tf.data.Dataset:
        """
        Create TensorFlow Dataset for ONI data.
        
        Args:
            oni_dataset: ONI dataset to wrap
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle the data
            buffer_size: Buffer size for shuffling
            return_format: Format to return data ('tuple' or 'dict')
            
        Returns:
            TensorFlow Dataset
        """
        # Get all data
        spatial_data = oni_dataset.get_spatial_data()
        global_data = oni_dataset.get_global_features()
        labels = oni_dataset.get_labels()
        
        if return_format == 'tuple':
            # Return (inputs, targets) format
            inputs = {
                'spatial': spatial_data,
                'global': global_data
            }
            targets = labels.get('survival_status', np.zeros(len(spatial_data)))
            
            dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        elif return_format == 'dict':
            # Return everything as dictionary
            data_dict = {
                'spatial': spatial_data,
                'global': global_data,
                **{f'label_{k}': np.array(v) for k, v in labels.items() if isinstance(v, list)}
            }
            dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        else:
            raise ValueError(f"Invalid return_format: {return_format}")
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        
        dataset = dataset.batch(batch_size)
        return dataset

else:
    def create_tensorflow_dataset(*args, **kwargs):
        raise ImportError("TensorFlow is not available. Install with: pip install tensorflow")


# Scikit-learn Integration
def create_sklearn_data(oni_dataset: ONIDataset, 
                       feature_type: str = 'combined',
                       target_type: str = 'survival_status') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create scikit-learn compatible data arrays.
    
    Args:
        oni_dataset: ONI dataset to convert
        feature_type: Type of features to use ('spatial', 'global', 'tabular', or 'combined')
        target_type: Type of target labels to use
        
    Returns:
        Tuple of (X, y) arrays for scikit-learn
    """
    # Get features based on type
    if feature_type == 'spatial':
        spatial_data = oni_dataset.get_spatial_data()
        X = spatial_data.reshape(spatial_data.shape[0], -1)  # Flatten spatial dimensions
    elif feature_type == 'global':
        X = oni_dataset.get_global_features()
    elif feature_type == 'tabular':
        tabular_data = oni_dataset.get_tabular_data()
        # Select only numeric columns
        numeric_columns = tabular_data.select_dtypes(include=[np.number]).columns
        X = tabular_data[numeric_columns].values
    elif feature_type == 'combined':
        # Combine global features with tabular features
        global_data = oni_dataset.get_global_features()
        tabular_data = oni_dataset.get_tabular_data()
        numeric_columns = tabular_data.select_dtypes(include=[np.number]).columns
        tabular_array = tabular_data[numeric_columns].values
        X = np.concatenate([global_data, tabular_array], axis=1)
    else:
        raise ValueError(f"Invalid feature_type: {feature_type}")
    
    # Get target labels
    y = oni_dataset.get_labels(target_type)
    
    return X, y


# Utility functions
def create_data_loader(dataset: Dataset,
                      framework: str = 'pytorch',
                      batch_size: int = 32,
                      shuffle: bool = True,
                      transform: Optional[callable] = None,
                      **kwargs) -> Union[TorchDataLoader, tf.data.Dataset, Tuple[np.ndarray, np.ndarray]]:
    """
    Create data loader for specified ML framework.
    
    Args:
        dataset: Built ONI dataset
        framework: Target framework ('pytorch', 'tensorflow', or 'sklearn')
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the data
        transform: Optional transform function
        **kwargs: Additional framework-specific arguments
        
    Returns:
        Framework-specific data loader
    """
    oni_dataset = ONIDataset(dataset, transform=transform)
    
    if framework.lower() == 'pytorch':
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        return create_pytorch_dataloader(oni_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif framework.lower() == 'tensorflow':
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        return create_tensorflow_dataset(oni_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif framework.lower() == 'sklearn':
        return create_sklearn_data(oni_dataset, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def get_framework_info() -> Dict[str, bool]:
    """Get information about available ML frameworks."""
    return {
        'pytorch': TORCH_AVAILABLE,
        'tensorflow': TF_AVAILABLE,
        'numpy': True,  # Always available
        'pandas': True  # Always available (required dependency)
    }


# Data transforms
class StandardTransform:
    """Standard data transforms for ONI datasets."""
    
    @staticmethod
    def normalize_spatial(state_tensor: StateTensor, labels: Dict[str, Any]) -> Tuple[StateTensor, Dict[str, Any]]:
        """Normalize spatial tensor to [0, 1] range."""
        normalized_spatial = state_tensor.spatial.copy()
        
        # Normalize each channel separately
        for c in range(normalized_spatial.shape[2]):
            channel_data = normalized_spatial[:, :, c]
            min_val, max_val = channel_data.min(), channel_data.max()
            if max_val > min_val:
                normalized_spatial[:, :, c] = (channel_data - min_val) / (max_val - min_val)
        
        return StateTensor(
            spatial=normalized_spatial,
            global_features=state_tensor.global_features,
            metadata=state_tensor.metadata
        ), labels
    
    @staticmethod
    def standardize_global(state_tensor: StateTensor, labels: Dict[str, Any]) -> Tuple[StateTensor, Dict[str, Any]]:
        """Standardize global features to zero mean and unit variance."""
        global_features = state_tensor.global_features.copy()
        
        # Simple standardization (would be better with dataset statistics)
        mean = global_features.mean()
        std = global_features.std()
        if std > 0:
            global_features = (global_features - mean) / std
        
        return StateTensor(
            spatial=state_tensor.spatial,
            global_features=global_features,
            metadata=state_tensor.metadata
        ), labels
    
    @staticmethod
    def add_noise(noise_level: float = 0.01):
        """Add Gaussian noise to spatial data for regularization."""
        def transform(state_tensor: StateTensor, labels: Dict[str, Any]) -> Tuple[StateTensor, Dict[str, Any]]:
            noisy_spatial = state_tensor.spatial + np.random.normal(0, noise_level, state_tensor.spatial.shape)
            noisy_spatial = np.clip(noisy_spatial, 0, 1)  # Keep in valid range
            
            return StateTensor(
                spatial=noisy_spatial.astype(np.float32),
                global_features=state_tensor.global_features,
                metadata=state_tensor.metadata
            ), labels
        
        return transform


# Example usage and testing
def demo_data_loaders(dataset: Dataset):
    """Demonstrate data loader usage with different frameworks."""
    print("=== ONI Dataset Data Loader Demo ===")
    
    # Create ONI dataset wrapper
    oni_dataset = ONIDataset(dataset)
    print(f"Dataset size: {len(oni_dataset)} samples")
    
    # Framework availability
    frameworks = get_framework_info()
    print(f"Available frameworks: {frameworks}")
    
    # Test basic access
    sample_tensor, sample_labels = oni_dataset[0]
    print(f"Sample spatial shape: {sample_tensor.spatial.shape}")
    print(f"Sample global features shape: {sample_tensor.global_features.shape}")
    print(f"Sample labels: {list(sample_labels.keys())}")
    
    # Test data splitting
    train_ds, val_ds, test_ds = oni_dataset.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"Split sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Test PyTorch loader
    if frameworks['pytorch']:
        try:
            pytorch_loader = create_pytorch_dataloader(train_ds, batch_size=4, shuffle=True)
            batch = next(iter(pytorch_loader))
            print(f"PyTorch batch shapes: {[t.shape for t in batch]}")
        except Exception as e:
            print(f"PyTorch loader error: {e}")
    
    # Test TensorFlow dataset
    if frameworks['tensorflow']:
        try:
            tf_dataset = create_tensorflow_dataset(train_ds, batch_size=4)
            for batch in tf_dataset.take(1):
                print(f"TensorFlow batch keys: {list(batch[0].keys()) if isinstance(batch[0], dict) else 'tuple format'}")
        except Exception as e:
            print(f"TensorFlow dataset error: {e}")
    
    # Test scikit-learn format
    try:
        X, y = create_sklearn_data(train_ds, feature_type='combined')
        print(f"Scikit-learn format - X: {X.shape}, y: {y.shape}")
    except Exception as e:
        print(f"Scikit-learn format error: {e}")
    
    print("=== Demo Complete ===")