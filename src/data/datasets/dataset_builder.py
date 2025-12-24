"""
Dataset Builder for ONI AI Agent.

This module implements the dataset building functionality to convert processed
game states into ML-ready datasets with proper storage formats and labeling.
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from ..parsers.oni_save_parser import ONISaveParser, GameState
from ..preprocessors.state_preprocessor import StatePreprocessor, StateTensor


@dataclass
class Dataset:
    """
    ML-ready dataset containing processed ONI game states.
    
    This dataclass represents a complete dataset with metadata, tensor data,
    and tabular features ready for machine learning training and evaluation.
    """
    metadata: Dict[str, Any]  # Dataset metadata and configuration
    tensor_data_path: str     # Path to NPZ file containing tensor data
    tabular_data_path: str    # Path to Parquet file containing tabular features
    labels_path: str          # Path to JSON file containing labels
    num_samples: int          # Total number of samples in dataset
    
    def load_tensors(self) -> np.ndarray:
        """Load spatial tensor data from NPZ file."""
        if not os.path.exists(self.tensor_data_path):
            raise FileNotFoundError(f"Tensor data file not found: {self.tensor_data_path}")
        
        data = np.load(self.tensor_data_path)
        return data['spatial_tensors']
    
    def load_global_features(self) -> np.ndarray:
        """Load global features from NPZ file."""
        if not os.path.exists(self.tensor_data_path):
            raise FileNotFoundError(f"Tensor data file not found: {self.tensor_data_path}")
        
        data = np.load(self.tensor_data_path)
        return data['global_features']
    
    def load_tabular_data(self) -> pd.DataFrame:
        """Load tabular features from Parquet file."""
        if not os.path.exists(self.tabular_data_path):
            raise FileNotFoundError(f"Tabular data file not found: {self.tabular_data_path}")
        
        return pd.read_parquet(self.tabular_data_path)
    
    def load_labels(self) -> Dict[str, Any]:
        """Load labels from JSON file."""
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        
        with open(self.labels_path, 'r') as f:
            return json.load(f)
    
    def get_sample(self, index: int) -> Tuple[StateTensor, Dict[str, Any]]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (StateTensor, labels_dict)
        """
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Sample index {index} out of range [0, {self.num_samples})")
        
        # Load tensor data
        spatial_tensors = self.load_tensors()
        global_features = self.load_global_features()
        
        # Load tabular data for metadata
        tabular_data = self.load_tabular_data()
        sample_metadata = tabular_data.iloc[index].to_dict()
        
        # Create StateTensor
        state_tensor = StateTensor(
            spatial=spatial_tensors[index],
            global_features=global_features[index],
            metadata=sample_metadata
        )
        
        # Load labels
        labels = self.load_labels()
        sample_labels = {key: values[index] for key, values in labels.items() if isinstance(values, list)}
        
        return state_tensor, sample_labels


class DatasetBuilder:
    """
    Builder for creating ML-ready datasets from ONI save files.
    
    This class handles the complete pipeline from save files to structured
    datasets with proper storage formats and labeling systems.
    """
    
    def __init__(self, 
                 output_dir: str,
                 preprocessor_config: Optional[Dict[str, Any]] = None,
                 augmentation_config: Optional[Dict[str, Any]] = None,
                 memory_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dataset builder.
        
        Args:
            output_dir: Directory to store the built dataset
            preprocessor_config: Configuration for state preprocessing
            augmentation_config: Configuration for data augmentation
            memory_config: Configuration for memory management
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = ONISaveParser()
        
        # Configure preprocessor
        preprocessor_config = preprocessor_config or {}
        self.preprocessor = StatePreprocessor(
            target_size=preprocessor_config.get('target_size', (64, 64)),
            temperature_range=preprocessor_config.get('temperature_range', (-50.0, 200.0)),
            max_element_id=preprocessor_config.get('max_element_id', 255),
            max_building_types=preprocessor_config.get('max_building_types', 20)
        )
        
        # Configure augmentation
        self.augmentation_config = augmentation_config or {
            'enable_rotation': True,
            'enable_cropping': True,
            'rotation_angles': [0, 90, 180, 270],
            'crop_ratios': [0.8, 0.9, 1.0],
            'augmentation_factor': 2  # Number of augmented samples per original
        }
        
        # Configure memory management
        self.memory_config = memory_config or {
            'batch_size': 50,           # Process files in batches
            'max_memory_gb': 8.0,       # Maximum memory usage in GB
            'use_temp_storage': True,   # Use temporary files for large datasets
            'compression_level': 6      # NPZ compression level (0-9)
        }
        
        # Storage configuration
        self.storage_config = {
            'metadata_file': 'metadata.json',
            'tensor_file': 'tensors.npz',
            'tabular_file': 'features.parquet',
            'labels_file': 'labels.json'
        }
    
    def build_dataset(self, save_files: List[str]) -> Dataset:
        """
        Build a complete dataset from ONI save files with memory management.
        
        Args:
            save_files: List of paths to .sav files
            
        Returns:
            Dataset object with all processed data
            
        Raises:
            ValueError: If no valid save files are provided
            RuntimeError: If dataset building fails
        """
        if not save_files:
            raise ValueError("No save files provided")
        
        print(f"Building dataset from {len(save_files)} save files...")
        
        # Estimate memory requirements and determine batch processing strategy
        batch_size = self._calculate_optimal_batch_size(save_files)
        use_batched_processing = len(save_files) > batch_size
        
        if use_batched_processing:
            print(f"Using batched processing with batch size: {batch_size}")
            return self._build_dataset_batched(save_files, batch_size)
        else:
            print("Using in-memory processing")
            return self._build_dataset_in_memory(save_files)
    
    def _calculate_optimal_batch_size(self, save_files: List[str]) -> int:
        """Calculate optimal batch size based on memory constraints."""
        max_memory_gb = self.memory_config.get('max_memory_gb', 8.0)
        default_batch_size = self.memory_config.get('batch_size', 50)
        
        # Estimate memory per sample (rough calculation)
        target_height, target_width = self.preprocessor.target_size
        channels = self.preprocessor.num_channels
        
        # Memory per sample: spatial tensor + global features + metadata
        spatial_memory = target_height * target_width * channels * 4  # float32
        global_memory = 64 * 4  # 64 float32 features
        metadata_memory = 1024  # Rough estimate for metadata
        
        memory_per_sample = spatial_memory + global_memory + metadata_memory
        
        # Account for augmentation
        augmentation_factor = self.augmentation_config.get('augmentation_factor', 1)
        memory_per_sample *= augmentation_factor
        
        # Calculate batch size to stay within memory limit
        max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        calculated_batch_size = int(max_memory_bytes * 0.7 / memory_per_sample)  # Use 70% of available memory
        
        # Use the smaller of calculated and default batch size
        optimal_batch_size = min(max(calculated_batch_size, 10), default_batch_size)
        
        print(f"Memory estimation: {memory_per_sample / 1024 / 1024:.2f} MB per sample")
        print(f"Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def _build_dataset_in_memory(self, save_files: List[str]) -> Dataset:
        """Build dataset entirely in memory (for smaller datasets)."""
        # Step 1: Parse and validate save files
        parsed_states = self._parse_save_files(save_files)
        
        if not parsed_states:
            raise RuntimeError("No valid save files could be parsed")
        
        print(f"Successfully parsed {len(parsed_states)} save files")
        
        # Step 2: Preprocess game states to tensors
        state_tensors = self._preprocess_states(parsed_states)
        print(f"Preprocessed {len(state_tensors)} game states")
        
        # Step 3: Apply data augmentation
        if self.augmentation_config.get('augmentation_factor', 0) > 1:
            augmented_tensors, augmented_metadata = self._apply_augmentation(state_tensors, parsed_states)
            state_tensors.extend(augmented_tensors)
            parsed_states.extend(augmented_metadata)
            print(f"Applied augmentation, total samples: {len(state_tensors)}")
        
        # Step 4: Create labeling system
        labels = self._create_labels(parsed_states)
        print(f"Created labels for {len(labels['survival_status'])} samples")
        
        # Step 5: Extract tabular features
        tabular_features = self._extract_tabular_features(parsed_states, state_tensors)
        print(f"Extracted tabular features: {tabular_features.shape}")
        
        # Step 6: Store data in appropriate formats
        dataset = self._store_dataset(state_tensors, tabular_features, labels)
        print(f"Dataset stored in: {self.output_dir}")
        
        return dataset
    
    def _build_dataset_batched(self, save_files: List[str], batch_size: int) -> Dataset:
        """Build dataset using batched processing for memory efficiency."""
        # Create temporary directory for batch processing
        temp_dir = None
        if self.memory_config.get('use_temp_storage', True):
            temp_dir = tempfile.mkdtemp(prefix='oni_dataset_')
            print(f"Using temporary storage: {temp_dir}")
        
        try:
            # Process files in batches
            all_state_tensors = []
            all_parsed_states = []
            batch_count = 0
            
            for i in range(0, len(save_files), batch_size):
                batch_files = save_files[i:i + batch_size]
                batch_count += 1
                
                print(f"Processing batch {batch_count}/{(len(save_files) + batch_size - 1) // batch_size}")
                
                # Process current batch
                batch_parsed_states = self._parse_save_files(batch_files)
                batch_state_tensors = self._preprocess_states(batch_parsed_states)
                
                # Apply augmentation to batch
                if self.augmentation_config.get('augmentation_factor', 0) > 1:
                    augmented_tensors, augmented_metadata = self._apply_augmentation(batch_state_tensors, batch_parsed_states)
                    batch_state_tensors.extend(augmented_tensors)
                    batch_parsed_states.extend(augmented_metadata)
                
                # Store batch temporarily if using temp storage
                if temp_dir:
                    batch_file = os.path.join(temp_dir, f'batch_{batch_count}.npz')
                    spatial_tensors = np.array([st.spatial for st in batch_state_tensors])
                    global_features = np.array([st.global_features for st in batch_state_tensors])
                    
                    np.savez_compressed(
                        batch_file,
                        spatial_tensors=spatial_tensors,
                        global_features=global_features,
                        metadata=[st.metadata for st in batch_state_tensors]
                    )
                    
                    # Keep only metadata for final processing
                    all_parsed_states.extend(batch_parsed_states)
                    
                    # Clear memory
                    del batch_state_tensors, spatial_tensors, global_features
                else:
                    # Keep in memory
                    all_state_tensors.extend(batch_state_tensors)
                    all_parsed_states.extend(batch_parsed_states)
                
                print(f"Batch {batch_count} processed: {len(batch_parsed_states)} samples")
            
            # Create labels and tabular features
            labels = self._create_labels(all_parsed_states)
            
            # If using temp storage, reconstruct state tensors for tabular features
            if temp_dir:
                # Load first batch to get structure for tabular features
                first_batch = np.load(os.path.join(temp_dir, 'batch_1.npz'))
                sample_tensors = []
                for i in range(min(10, len(first_batch['spatial_tensors']))):  # Sample first 10 for feature extraction
                    sample_tensors.append(StateTensor(
                        spatial=first_batch['spatial_tensors'][i],
                        global_features=first_batch['global_features'][i],
                        metadata=first_batch['metadata'][i]
                    ))
                
                # Extract tabular features from sample
                sample_parsed_states = all_parsed_states[:len(sample_tensors)]
                tabular_features = self._extract_tabular_features(sample_parsed_states, sample_tensors)
                
                # Extend tabular features for all samples
                tabular_features = self._extend_tabular_features(tabular_features, all_parsed_states)
                
                # Combine all batch files into final dataset
                dataset = self._combine_batched_data(temp_dir, batch_count, tabular_features, labels)
            else:
                tabular_features = self._extract_tabular_features(all_parsed_states, all_state_tensors)
                dataset = self._store_dataset(all_state_tensors, tabular_features, labels)
            
            print(f"Batched dataset stored in: {self.output_dir}")
            return dataset
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print("Cleaned up temporary storage")
    
    def _extend_tabular_features(self, sample_features: pd.DataFrame, all_parsed_states: List[Tuple[GameState, str]]) -> pd.DataFrame:
        """Extend tabular features to cover all samples when using batched processing."""
        # Create features for all samples based on the sample structure
        all_features = []
        
        for i, (game_state, file_path) in enumerate(all_parsed_states):
            # Create feature row similar to sample but with current data
            feature_row = {
                'sample_id': i,
                'source_file': os.path.basename(file_path),
                'cycle': game_state.cycle,
                'timestamp': game_state.timestamp,
                'world_height': game_state.world_size[0],
                'world_width': game_state.world_size[1],
                'world_area': game_state.world_size[0] * game_state.world_size[1],
                'num_duplicants': len(game_state.duplicants),
                'avg_duplicant_health': sum(d.health for d in game_state.duplicants) / len(game_state.duplicants) if game_state.duplicants else 0.0,
                'avg_duplicant_stress': sum(d.stress_level for d in game_state.duplicants) / len(game_state.duplicants) if game_state.duplicants else 100.0,
                'total_skills': sum(sum(d.skills.values()) for d in game_state.duplicants),
                'num_buildings': len(game_state.buildings),
                'operational_buildings': sum(1 for b in game_state.buildings if b.operational),
                'operational_ratio': sum(1 for b in game_state.buildings if b.operational) / len(game_state.buildings) if game_state.buildings else 0.0,
                'power_buildings': sum(1 for b in game_state.buildings if b.building_type == 'power'),
                'ventilation_buildings': sum(1 for b in game_state.buildings if b.building_type == 'ventilation'),
                'living_buildings': sum(1 for b in game_state.buildings if b.building_type == 'living'),
                'agriculture_buildings': sum(1 for b in game_state.buildings if b.building_type == 'agriculture'),
                'research_buildings': sum(1 for b in game_state.buildings if b.building_type == 'research'),
                'storage_buildings': sum(1 for b in game_state.buildings if b.building_type == 'storage'),
                'infrastructure_buildings': sum(1 for b in game_state.buildings if b.building_type == 'infrastructure'),
            }
            
            # Add resource features
            for name, value in game_state.resources.items():
                feature_row[f'resource_{name}'] = value
            
            # Add default tensor properties (will be updated when loading)
            feature_row.update({
                'tensor_height': self.preprocessor.target_size[0],
                'tensor_width': self.preprocessor.target_size[1],
                'tensor_channels': self.preprocessor.num_channels,
                'global_features_dim': 64,
                'is_mock': False,
                'preprocessing_version': '1.0',
                'is_augmented': False,
                'augmentation_type': 'none',
                'original_index': i
            })
            
            all_features.append(feature_row)
        
        return pd.DataFrame(all_features)
    
    def _combine_batched_data(self, temp_dir: str, batch_count: int, tabular_features: pd.DataFrame, labels: Dict[str, List[Any]]) -> Dataset:
        """Combine batched data files into final dataset."""
        # Prepare file paths
        tensor_path = self.output_dir / self.storage_config['tensor_file']
        
        # Combine all batch files
        all_spatial_tensors = []
        all_global_features = []
        all_metadata = []
        
        for batch_num in range(1, batch_count + 1):
            batch_file = os.path.join(temp_dir, f'batch_{batch_num}.npz')
            if os.path.exists(batch_file):
                batch_data = np.load(batch_file)
                all_spatial_tensors.append(batch_data['spatial_tensors'])
                all_global_features.append(batch_data['global_features'])
                all_metadata.extend(batch_data['metadata'])
        
        # Concatenate all batches
        combined_spatial = np.concatenate(all_spatial_tensors, axis=0)
        combined_global = np.concatenate(all_global_features, axis=0)
        
        # Store combined tensor data
        compression_level = self.memory_config.get('compression_level', 6)
        np.savez_compressed(
            tensor_path,
            spatial_tensors=combined_spatial,
            global_features=combined_global,
            tensor_metadata=all_metadata,
            compress=compression_level
        )
        
        # Store other data using existing method
        tabular_path = self.output_dir / self.storage_config['tabular_file']
        labels_path = self.output_dir / self.storage_config['labels_file']
        metadata_path = self.output_dir / self.storage_config['metadata_file']
        
        tabular_features.to_parquet(tabular_path, index=False)
        
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2, default=str)
        
        # Create dataset metadata
        dataset_metadata = {
            'created_at': pd.Timestamp.now().isoformat(),
            'num_samples': len(combined_spatial),
            'spatial_shape': list(combined_spatial.shape[1:]),
            'global_features_dim': combined_global.shape[1],
            'preprocessor_config': {
                'target_size': self.preprocessor.target_size,
                'temperature_range': self.preprocessor.temperature_range,
                'max_element_id': self.preprocessor.max_element_id,
                'max_building_types': self.preprocessor.max_building_types
            },
            'augmentation_config': self.augmentation_config,
            'memory_config': self.memory_config,
            'storage_config': self.storage_config,
            'label_types': list(labels.keys()),
            'tabular_features': list(tabular_features.columns),
            'dataset_version': '1.0',
            'processing_method': 'batched'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2, default=str)
        
        return Dataset(
            metadata=dataset_metadata,
            tensor_data_path=str(tensor_path),
            tabular_data_path=str(tabular_path),
            labels_path=str(labels_path),
            num_samples=len(combined_spatial)
        )
    
    def _parse_save_files(self, save_files: List[str]) -> List[Tuple[GameState, str]]:
        """Parse save files and return valid game states with metadata."""
        parsed_states = []
        
        # Get parsing statistics first
        stats = self.parser.get_parsing_statistics(save_files)
        print(f"Parsing statistics: {stats['valid_files']} valid, {stats['invalid_files']} invalid, {stats['missing_files']} missing")
        
        # Parse files in batch with error handling
        results = self.parser.parse_save_batch(save_files, skip_corrupted=False)
        
        for file_path, game_state, error_msg in results:
            if game_state is not None:
                parsed_states.append((game_state, file_path))
            else:
                warnings.warn(f"Failed to parse {file_path}: {error_msg}")
        
        return parsed_states
    
    def _preprocess_states(self, parsed_states: List[Tuple[GameState, str]]) -> List[StateTensor]:
        """Preprocess game states into ML-ready tensors."""
        state_tensors = []
        
        for game_state, file_path in parsed_states:
            try:
                state_tensor = self.preprocessor.preprocess_state(game_state)
                # Add source file information to metadata
                state_tensor.metadata['source_file'] = file_path
                state_tensors.append(state_tensor)
            except Exception as e:
                warnings.warn(f"Failed to preprocess {file_path}: {e}")
                continue
        
        return state_tensors
    
    def _apply_augmentation(self, state_tensors: List[StateTensor], parsed_states: List[Tuple[GameState, str]]) -> Tuple[List[StateTensor], List[Tuple[GameState, str]]]:
        """Apply data augmentation to increase dataset size."""
        augmented_tensors = []
        augmented_metadata = []
        
        augmentation_factor = self.augmentation_config.get('augmentation_factor', 2)
        enable_rotation = self.augmentation_config.get('enable_rotation', True)
        enable_cropping = self.augmentation_config.get('enable_cropping', True)
        rotation_angles = self.augmentation_config.get('rotation_angles', [90, 180, 270])
        crop_ratios = self.augmentation_config.get('crop_ratios', [0.8, 0.9])
        
        for i, (state_tensor, (game_state, file_path)) in enumerate(zip(state_tensors, parsed_states)):
            augmentations_created = 0
            
            # Rotation augmentation
            if enable_rotation and augmentations_created < augmentation_factor - 1:
                for angle in rotation_angles:
                    if augmentations_created >= augmentation_factor - 1:
                        break
                    
                    rotated_tensor = self._rotate_tensor(state_tensor, angle)
                    rotated_tensor.metadata['augmentation'] = f'rotation_{angle}'
                    rotated_tensor.metadata['original_index'] = i
                    augmented_tensors.append(rotated_tensor)
                    
                    # Create corresponding metadata
                    augmented_metadata.append((game_state, f"{file_path}_rot{angle}"))
                    augmentations_created += 1
            
            # Cropping augmentation
            if enable_cropping and augmentations_created < augmentation_factor - 1:
                for crop_ratio in crop_ratios:
                    if augmentations_created >= augmentation_factor - 1:
                        break
                    
                    cropped_tensor = self._crop_tensor(state_tensor, crop_ratio)
                    cropped_tensor.metadata['augmentation'] = f'crop_{crop_ratio}'
                    cropped_tensor.metadata['original_index'] = i
                    augmented_tensors.append(cropped_tensor)
                    
                    # Create corresponding metadata
                    augmented_metadata.append((game_state, f"{file_path}_crop{crop_ratio}"))
                    augmentations_created += 1
        
        return augmented_tensors, augmented_metadata
    
    def _rotate_tensor(self, state_tensor: StateTensor, angle: int) -> StateTensor:
        """Rotate spatial tensor by specified angle."""
        # Number of 90-degree rotations
        k = angle // 90
        
        # Rotate spatial tensor
        rotated_spatial = np.rot90(state_tensor.spatial, k=k, axes=(0, 1))
        
        # Global features remain the same
        rotated_global = state_tensor.global_features.copy()
        
        # Update metadata
        rotated_metadata = state_tensor.metadata.copy()
        rotated_metadata['rotation_angle'] = angle
        
        return StateTensor(
            spatial=rotated_spatial,
            global_features=rotated_global,
            metadata=rotated_metadata
        )
    
    def _crop_tensor(self, state_tensor: StateTensor, crop_ratio: float) -> StateTensor:
        """Crop spatial tensor to specified ratio and resize back."""
        height, width, channels = state_tensor.spatial.shape
        
        # Calculate crop dimensions
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)
        
        # Calculate crop offsets (center crop)
        y_offset = (height - crop_height) // 2
        x_offset = (width - crop_width) // 2
        
        # Crop the tensor
        cropped = state_tensor.spatial[y_offset:y_offset+crop_height, x_offset:x_offset+crop_width, :]
        
        # Resize back to original dimensions using simple interpolation
        resized_spatial = self._resize_tensor(cropped, (height, width))
        
        # Global features remain the same
        cropped_global = state_tensor.global_features.copy()
        
        # Update metadata
        cropped_metadata = state_tensor.metadata.copy()
        cropped_metadata['crop_ratio'] = crop_ratio
        
        return StateTensor(
            spatial=resized_spatial,
            global_features=cropped_global,
            metadata=cropped_metadata
        )
    
    def _resize_tensor(self, tensor: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize tensor to target dimensions using nearest neighbor interpolation."""
        current_height, current_width, channels = tensor.shape
        target_height, target_width = target_size
        
        # Create output tensor
        resized = np.zeros((target_height, target_width, channels), dtype=tensor.dtype)
        
        # Calculate scaling factors
        height_scale = current_height / target_height
        width_scale = current_width / target_width
        
        # Nearest neighbor interpolation
        for y in range(target_height):
            for x in range(target_width):
                src_y = int(y * height_scale)
                src_x = int(x * width_scale)
                
                # Ensure bounds
                src_y = min(src_y, current_height - 1)
                src_x = min(src_x, current_width - 1)
                
                resized[y, x, :] = tensor[src_y, src_x, :]
        
        return resized
    
    def _create_labels(self, parsed_states: List[Tuple[GameState, str]]) -> Dict[str, List[Any]]:
        """Create labeling system for colony outcomes."""
        labels = {
            'survival_status': [],      # Boolean: colony alive/dead
            'resource_counts': [],      # Dict: resource quantities
            'duplicant_stress': [],     # Float: average duplicant stress
            'infrastructure_score': [], # Float: infrastructure completion metric
            'cycle_progression': [],    # Int: cycle number
            'colony_health': []         # Float: overall colony health score
        }
        
        for game_state, file_path in parsed_states:
            # Survival status (based on duplicant health and resources)
            survival = self._assess_survival_status(game_state)
            labels['survival_status'].append(survival)
            
            # Resource counts
            labels['resource_counts'].append(game_state.resources.copy())
            
            # Duplicant stress (average)
            if game_state.duplicants:
                avg_stress = sum(d.stress_level for d in game_state.duplicants) / len(game_state.duplicants)
            else:
                avg_stress = 100.0  # Maximum stress if no duplicants
            labels['duplicant_stress'].append(avg_stress)
            
            # Infrastructure score
            infra_score = self._calculate_infrastructure_score(game_state)
            labels['infrastructure_score'].append(infra_score)
            
            # Cycle progression
            labels['cycle_progression'].append(game_state.cycle)
            
            # Overall colony health
            health_score = self._calculate_colony_health(game_state)
            labels['colony_health'].append(health_score)
        
        return labels
    
    def _assess_survival_status(self, game_state: GameState) -> bool:
        """Assess whether the colony is in a surviving state."""
        # Check if any duplicants are alive and healthy
        if not game_state.duplicants:
            return False
        
        # Check if duplicants have reasonable health
        healthy_duplicants = sum(1 for d in game_state.duplicants if d.health > 20.0)
        if healthy_duplicants == 0:
            return False
        
        # Check critical resources
        oxygen = game_state.resources.get('oxygen', 0.0)
        water = game_state.resources.get('water', 0.0)
        food = game_state.resources.get('food', 0.0)
        
        # Basic survival thresholds
        if oxygen < 100.0 or water < 50.0 or food < 20.0:
            return False
        
        return True
    
    def _calculate_infrastructure_score(self, game_state: GameState) -> float:
        """Calculate infrastructure completion score."""
        if not game_state.buildings:
            return 0.0
        
        # Count essential building types
        building_types = {}
        operational_count = 0
        
        for building in game_state.buildings:
            building_types[building.building_type] = building_types.get(building.building_type, 0) + 1
            if building.operational:
                operational_count += 1
        
        # Essential building categories
        essential_types = ['power', 'ventilation', 'living', 'agriculture']
        essential_score = sum(min(building_types.get(bt, 0), 3) for bt in essential_types) / (len(essential_types) * 3)
        
        # Operational ratio
        operational_ratio = operational_count / len(game_state.buildings) if game_state.buildings else 0.0
        
        # Diversity bonus
        diversity_bonus = min(len(building_types), 8) / 8.0
        
        return (essential_score * 0.5 + operational_ratio * 0.3 + diversity_bonus * 0.2)
    
    def _calculate_colony_health(self, game_state: GameState) -> float:
        """Calculate overall colony health score."""
        if not game_state.duplicants:
            return 0.0
        
        # Duplicant health component
        avg_health = sum(d.health for d in game_state.duplicants) / len(game_state.duplicants)
        avg_stress = sum(d.stress_level for d in game_state.duplicants) / len(game_state.duplicants)
        duplicant_score = (avg_health / 100.0) * (1.0 - avg_stress / 100.0)
        
        # Resource adequacy component
        oxygen = game_state.resources.get('oxygen', 0.0)
        water = game_state.resources.get('water', 0.0)
        food = game_state.resources.get('food', 0.0)
        
        resource_score = min(1.0, (
            min(oxygen / 1000.0, 1.0) * 0.4 +
            min(water / 500.0, 1.0) * 0.3 +
            min(food / 200.0, 1.0) * 0.3
        ))
        
        # Infrastructure component
        infra_score = self._calculate_infrastructure_score(game_state)
        
        # Combined score
        return duplicant_score * 0.4 + resource_score * 0.4 + infra_score * 0.2
    
    def _extract_tabular_features(self, parsed_states: List[Tuple[GameState, str]], state_tensors: List[StateTensor]) -> pd.DataFrame:
        """Extract tabular features for analysis and ML frameworks."""
        features = []
        
        for i, ((game_state, file_path), state_tensor) in enumerate(zip(parsed_states, state_tensors)):
            feature_row = {
                # Basic identifiers
                'sample_id': i,
                'source_file': os.path.basename(file_path),
                'cycle': game_state.cycle,
                'timestamp': game_state.timestamp,
                
                # World properties
                'world_height': game_state.world_size[0],
                'world_width': game_state.world_size[1],
                'world_area': game_state.world_size[0] * game_state.world_size[1],
                
                # Duplicant statistics
                'num_duplicants': len(game_state.duplicants),
                'avg_duplicant_health': sum(d.health for d in game_state.duplicants) / len(game_state.duplicants) if game_state.duplicants else 0.0,
                'avg_duplicant_stress': sum(d.stress_level for d in game_state.duplicants) / len(game_state.duplicants) if game_state.duplicants else 100.0,
                'total_skills': sum(sum(d.skills.values()) for d in game_state.duplicants),
                
                # Building statistics
                'num_buildings': len(game_state.buildings),
                'operational_buildings': sum(1 for b in game_state.buildings if b.operational),
                'operational_ratio': sum(1 for b in game_state.buildings if b.operational) / len(game_state.buildings) if game_state.buildings else 0.0,
                
                # Building type counts
                'power_buildings': sum(1 for b in game_state.buildings if b.building_type == 'power'),
                'ventilation_buildings': sum(1 for b in game_state.buildings if b.building_type == 'ventilation'),
                'living_buildings': sum(1 for b in game_state.buildings if b.building_type == 'living'),
                'agriculture_buildings': sum(1 for b in game_state.buildings if b.building_type == 'agriculture'),
                'research_buildings': sum(1 for b in game_state.buildings if b.building_type == 'research'),
                'storage_buildings': sum(1 for b in game_state.buildings if b.building_type == 'storage'),
                'infrastructure_buildings': sum(1 for b in game_state.buildings if b.building_type == 'infrastructure'),
                
                # Resource features
                **{f'resource_{name}': value for name, value in game_state.resources.items()},
                
                # Tensor properties
                'tensor_height': state_tensor.spatial.shape[0],
                'tensor_width': state_tensor.spatial.shape[1],
                'tensor_channels': state_tensor.spatial.shape[2],
                'global_features_dim': len(state_tensor.global_features),
                
                # Preprocessing metadata
                'is_mock': state_tensor.metadata.get('mock', False),
                'preprocessing_version': state_tensor.metadata.get('preprocessing_version', 'unknown'),
                
                # Augmentation metadata
                'is_augmented': 'augmentation' in state_tensor.metadata,
                'augmentation_type': state_tensor.metadata.get('augmentation', 'none'),
                'original_index': state_tensor.metadata.get('original_index', i)
            }
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _store_dataset(self, state_tensors: List[StateTensor], tabular_features: pd.DataFrame, labels: Dict[str, List[Any]]) -> Dataset:
        """Store dataset in appropriate formats."""
        # Prepare file paths
        metadata_path = self.output_dir / self.storage_config['metadata_file']
        tensor_path = self.output_dir / self.storage_config['tensor_file']
        tabular_path = self.output_dir / self.storage_config['tabular_file']
        labels_path = self.output_dir / self.storage_config['labels_file']
        
        # Prepare tensor data
        spatial_tensors = np.array([st.spatial for st in state_tensors])
        global_features = np.array([st.global_features for st in state_tensors])
        
        # Store tensor data in NPZ format
        np.savez_compressed(
            tensor_path,
            spatial_tensors=spatial_tensors,
            global_features=global_features,
            tensor_metadata=[st.metadata for st in state_tensors]
        )
        
        # Store tabular features in Parquet format
        tabular_features.to_parquet(tabular_path, index=False)
        
        # Store labels in JSON format
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2, default=str)
        
        # Create dataset metadata
        dataset_metadata = {
            'created_at': pd.Timestamp.now().isoformat(),
            'num_samples': len(state_tensors),
            'spatial_shape': list(spatial_tensors.shape[1:]),  # (height, width, channels)
            'global_features_dim': global_features.shape[1],
            'preprocessor_config': {
                'target_size': self.preprocessor.target_size,
                'temperature_range': self.preprocessor.temperature_range,
                'max_element_id': self.preprocessor.max_element_id,
                'max_building_types': self.preprocessor.max_building_types
            },
            'augmentation_config': self.augmentation_config,
            'storage_config': self.storage_config,
            'label_types': list(labels.keys()),
            'tabular_features': list(tabular_features.columns),
            'dataset_version': '1.0'
        }
        
        # Store metadata in JSON format
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2, default=str)
        
        return Dataset(
            metadata=dataset_metadata,
            tensor_data_path=str(tensor_path),
            tabular_data_path=str(tabular_path),
            labels_path=str(labels_path),
            num_samples=len(state_tensors)
        )


# Main interface function
def build_dataset(save_files: List[str], 
                 output_dir: str = "data/ml_ready",
                 preprocessor_config: Optional[Dict[str, Any]] = None,
                 augmentation_config: Optional[Dict[str, Any]] = None,
                 memory_config: Optional[Dict[str, Any]] = None) -> Dataset:
    """
    Build a complete ML-ready dataset from ONI save files with memory management.
    
    This is the main interface function that implements the required
    `build_dataset(save_files: List[str]) -> Dataset` interface.
    
    Args:
        save_files: List of paths to .sav files
        output_dir: Directory to store the built dataset
        preprocessor_config: Configuration for state preprocessing
        augmentation_config: Configuration for data augmentation
        memory_config: Configuration for memory management and batch processing
        
    Returns:
        Dataset object containing all processed data
        
    Example:
        >>> save_files = ["Colony001.sav", "Colony002.sav"]
        >>> dataset = build_dataset(save_files, "data/my_dataset")
        >>> print(f"Dataset contains {dataset.num_samples} samples")
        >>> tensors = dataset.load_tensors()
        >>> labels = dataset.load_labels()
        
        # For large datasets with memory constraints
        >>> memory_config = {'batch_size': 25, 'max_memory_gb': 4.0}
        >>> dataset = build_dataset(save_files, memory_config=memory_config)
    """
    builder = DatasetBuilder(
        output_dir=output_dir,
        preprocessor_config=preprocessor_config,
        augmentation_config=augmentation_config,
        memory_config=memory_config
    )
    return builder.build_dataset(save_files)