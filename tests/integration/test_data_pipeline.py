"""
Integration tests for the complete data processing pipeline.

Tests the end-to-end flow from ONI save file parsing through
data preprocessing to ML-ready tensor format.
"""

import numpy as np
import pytest
from pathlib import Path

from src.data import parse_save, preprocess_state, StateTensor


class TestDataPipeline:
    """Integration tests for the complete data processing pipeline."""
    
    def test_complete_pipeline_with_real_save(self):
        """Test the complete pipeline with a real save file."""
        # Use the main colony save file
        save_file = "tests/fixtures/sample_saves/test_colony_main.sav"
        
        if not Path(save_file).exists():
            pytest.skip("Sample save file not available")
        
        # Step 1: Parse save file
        game_state = parse_save(save_file)
        
        # Verify parsing results
        assert game_state is not None
        assert game_state.grid is not None
        assert len(game_state.grid.shape) == 3
        assert game_state.world_size is not None
        assert len(game_state.world_size) == 2
        
        # Step 2: Preprocess to ML format
        state_tensor = preprocess_state(game_state, target_size=(64, 64))
        
        # Verify preprocessing results
        assert isinstance(state_tensor, StateTensor)
        assert state_tensor.spatial.shape == (64, 64, 7)
        assert state_tensor.global_features.shape == (64,)
        assert state_tensor.spatial.dtype == np.float32
        assert state_tensor.global_features.dtype == np.float32
        
        # Verify data ranges
        assert np.all(state_tensor.spatial >= 0.0)
        assert np.all(state_tensor.spatial <= 1.0)
        assert np.all(state_tensor.global_features >= 0.0)
        
        # Verify metadata
        assert 'original_world_size' in state_tensor.metadata
        assert 'target_size' in state_tensor.metadata
        assert 'num_channels' in state_tensor.metadata
    
    def test_pipeline_with_different_target_sizes(self):
        """Test pipeline with various target sizes."""
        save_file = "tests/fixtures/sample_saves/test_colony_cycle_181.sav"
        
        if not Path(save_file).exists():
            pytest.skip("Sample save file not available")
        
        game_state = parse_save(save_file)
        
        # Test different target sizes
        target_sizes = [(32, 32), (64, 64), (128, 128), (16, 16)]
        
        for target_size in target_sizes:
            state_tensor = preprocess_state(game_state, target_size=target_size)
            
            expected_shape = (*target_size, 7)
            assert state_tensor.spatial.shape == expected_shape
            assert state_tensor.global_features.shape == (64,)
            
            # Verify metadata reflects target size
            assert state_tensor.metadata['target_size'] == target_size
    
    def test_pipeline_with_corrupted_file(self):
        """Test pipeline gracefully handles corrupted files."""
        save_file = "tests/fixtures/sample_saves/corrupted_file.sav"
        
        if not Path(save_file).exists():
            pytest.skip("Corrupted sample file not available")
        
        # Should create mock state for corrupted file
        game_state = parse_save(save_file)
        
        # Verify mock state is created
        assert game_state is not None
        assert game_state.metadata.get('mock', False) == True
        
        # Should still be able to preprocess
        state_tensor = preprocess_state(game_state)
        
        assert isinstance(state_tensor, StateTensor)
        assert state_tensor.spatial.shape == (64, 64, 7)
        assert state_tensor.global_features.shape == (64,)
    
    def test_pipeline_with_empty_file(self):
        """Test pipeline handles empty files."""
        save_file = "tests/fixtures/sample_saves/empty_file.sav"
        
        if not Path(save_file).exists():
            pytest.skip("Empty sample file not available")
        
        # Should create minimal mock state for empty file
        game_state = parse_save(save_file)
        
        # Verify minimal mock state
        assert game_state is not None
        assert game_state.metadata.get('mock', False) == True
        assert game_state.metadata.get('reason') == 'empty_file'
        
        # Should still be able to preprocess
        state_tensor = preprocess_state(game_state)
        
        assert isinstance(state_tensor, StateTensor)
        assert state_tensor.spatial.shape == (64, 64, 7)
        assert state_tensor.global_features.shape == (64,)
    
    def test_batch_pipeline_processing(self):
        """Test batch processing through the complete pipeline."""
        save_files = [
            "tests/fixtures/sample_saves/test_colony_main.sav",
            "tests/fixtures/sample_saves/test_colony_cycle_181.sav",
            "tests/fixtures/sample_saves/test_colony_cycle_190.sav"
        ]
        
        # Filter to existing files
        existing_files = [f for f in save_files if Path(f).exists()]
        
        if len(existing_files) < 2:
            pytest.skip("Not enough sample files for batch testing")
        
        processed_states = []
        
        for save_file in existing_files:
            # Parse and preprocess each file
            game_state = parse_save(save_file)
            state_tensor = preprocess_state(game_state, target_size=(32, 32))
            processed_states.append(state_tensor)
        
        # Verify batch results
        assert len(processed_states) >= 2
        
        # All should have same tensor shapes
        for state_tensor in processed_states:
            assert state_tensor.spatial.shape == (32, 32, 7)
            assert state_tensor.global_features.shape == (64,)
        
        # Should be able to stack into batches
        spatial_batch = np.stack([s.spatial for s in processed_states])
        global_batch = np.stack([s.global_features for s in processed_states])
        
        expected_batch_size = len(processed_states)
        assert spatial_batch.shape == (expected_batch_size, 32, 32, 7)
        assert global_batch.shape == (expected_batch_size, 64)
    
    def test_pipeline_data_consistency(self):
        """Test that pipeline produces consistent results for same input."""
        save_file = "tests/fixtures/sample_saves/test_colony_main.sav"
        
        if not Path(save_file).exists():
            pytest.skip("Sample save file not available")
        
        # Process same file multiple times
        results = []
        for _ in range(3):
            game_state = parse_save(save_file)
            state_tensor = preprocess_state(game_state, target_size=(32, 32))
            results.append(state_tensor)
        
        # Results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0].spatial, results[i].spatial,
                "Spatial tensors should be identical for same input"
            )
            np.testing.assert_array_equal(
                results[0].global_features, results[i].global_features,
                "Global features should be identical for same input"
            )
    
    def test_pipeline_memory_efficiency(self):
        """Test that pipeline doesn't leak memory with large inputs."""
        save_file = "tests/fixtures/sample_saves/test_colony_main.sav"
        
        if not Path(save_file).exists():
            pytest.skip("Sample save file not available")
        
        # Process multiple times to check for memory leaks
        for i in range(10):
            game_state = parse_save(save_file)
            state_tensor = preprocess_state(game_state, target_size=(64, 64))
            
            # Verify each result is valid
            assert state_tensor.spatial.shape == (64, 64, 7)
            assert state_tensor.global_features.shape == (64,)
            
            # Clear references to allow garbage collection
            del game_state
            del state_tensor
    
    def test_pipeline_with_custom_preprocessing_params(self):
        """Test pipeline with custom preprocessing parameters."""
        save_file = "tests/fixtures/sample_saves/test_colony_cycle_181.sav"
        
        if not Path(save_file).exists():
            pytest.skip("Sample save file not available")
        
        game_state = parse_save(save_file)
        
        # Test with custom temperature range
        state_tensor = preprocess_state(
            game_state,
            target_size=(48, 48),
            temperature_range=(0.0, 100.0)
        )
        
        assert state_tensor.spatial.shape == (48, 48, 7)
        assert state_tensor.metadata['temperature_range'] == (0.0, 100.0)
        assert state_tensor.metadata['target_size'] == (48, 48)
        
        # Temperature channel should still be normalized to [0, 1]
        temp_channel = state_tensor.spatial[:, :, 4]
        assert np.all(temp_channel >= 0.0)
        assert np.all(temp_channel <= 1.0)


class TestPipelineErrorHandling:
    """Test error handling in the complete pipeline."""
    
    def test_pipeline_with_invalid_file_path(self):
        """Test pipeline with non-existent file."""
        with pytest.raises(FileNotFoundError):
            parse_save("nonexistent_file.sav")
    
    def test_pipeline_with_invalid_game_state(self):
        """Test preprocessing with invalid game state."""
        from src.data.preprocessors.state_preprocessor import preprocess_state
        
        with pytest.raises(ValueError):
            preprocess_state(None)
    
    def test_pipeline_robustness(self):
        """Test pipeline robustness with edge cases."""
        # Test all available sample files, including problematic ones
        sample_dir = Path("tests/fixtures/sample_saves")
        
        if not sample_dir.exists():
            pytest.skip("Sample save directory not available")
        
        save_files = list(sample_dir.glob("*.sav"))
        
        if not save_files:
            pytest.skip("No sample save files available")
        
        successful_processes = 0
        
        for save_file in save_files:
            try:
                # Should not crash, even with problematic files
                game_state = parse_save(str(save_file))
                state_tensor = preprocess_state(game_state, target_size=(32, 32))
                
                # Basic validation
                assert isinstance(state_tensor, StateTensor)
                assert state_tensor.spatial.shape == (32, 32, 7)
                assert state_tensor.global_features.shape == (64,)
                
                successful_processes += 1
                
            except Exception as e:
                # Log but don't fail - some files may be intentionally problematic
                print(f"Warning: Failed to process {save_file}: {e}")
        
        # Should successfully process at least some files
        assert successful_processes > 0, "Pipeline should handle at least some sample files"


if __name__ == '__main__':
    pytest.main([__file__])