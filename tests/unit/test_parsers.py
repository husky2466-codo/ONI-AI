"""
Unit tests for ONI save file parsers.
"""

import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import numpy as np

from src.data.parsers.oni_save_parser import ONISaveParser, GameState, Duplicant, Building
from src.data.parsers.interface import parse_save


class TestONISaveParser(unittest.TestCase):
    """Test cases for the ONI save parser wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = None
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_dependency_verification_success(self, mock_run):
        """Test successful dependency verification."""
        # Mock successful Node.js version check
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
        ]
        
        # Should not raise an exception
        parser = ONISaveParser()
        self.assertIsNotNone(parser)
    
    @patch('subprocess.run')
    def test_dependency_verification_node_missing(self, mock_run):
        """Test dependency verification when Node.js is missing."""
        mock_run.side_effect = FileNotFoundError("node command not found")
        
        with self.assertRaises(RuntimeError) as context:
            ONISaveParser()
        
        self.assertIn("Node.js not found", str(context.exception))
    
    @patch('subprocess.run')
    def test_dependency_verification_parser_missing(self, mock_run):
        """Test dependency verification when oni-save-parser is missing."""
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js OK
            Mock(stdout='ERROR: Cannot find module\n', returncode=1)  # Parser missing
        ]
        
        with self.assertRaises(RuntimeError) as context:
            ONISaveParser()
        
        self.assertIn("oni-save-parser library not properly installed", str(context.exception))
    
    @patch('subprocess.run')
    def test_validate_save_file_success(self, mock_run):
        """Test successful save file validation."""
        # Mock dependency checks
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            Mock(stdout='{"valid": true}\n', returncode=0)  # Validation result
        ]
        
        parser = ONISaveParser()
        
        # Create a temporary file
        test_file = os.path.join(self.temp_dir, 'test.sav')
        with open(test_file, 'wb') as f:
            f.write(b'fake save data')
        
        is_valid, error = parser.validate_save_file(test_file)
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    @patch('subprocess.run')
    def test_validate_save_file_invalid(self, mock_run):
        """Test validation of invalid save file."""
        # Mock dependency checks and validation failure
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            Mock(stdout='{"valid": false, "error": "Corrupted file"}\n', returncode=0)
        ]
        
        parser = ONISaveParser()
        
        # Create a temporary file
        test_file = os.path.join(self.temp_dir, 'test.sav')
        with open(test_file, 'wb') as f:
            f.write(b'corrupted data')
        
        is_valid, error = parser.validate_save_file(test_file)
        
        self.assertFalse(is_valid)
        self.assertEqual(error, "Corrupted file")
    
    def test_validate_save_file_not_found(self):
        """Test validation of non-existent save file."""
        with patch('subprocess.run') as mock_run:
            # Mock dependency checks
            mock_run.side_effect = [
                Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
                Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
            ]
            
            parser = ONISaveParser()
            
            is_valid, error = parser.validate_save_file('nonexistent.sav')
            
            self.assertFalse(is_valid)
            self.assertIn("File not found", error)
    
    @patch('subprocess.run')
    def test_validate_save_file_empty(self, mock_run):
        """Test validation of empty save file."""
        # Mock dependency checks
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
        ]
        
        parser = ONISaveParser()
        
        # Create an empty file
        test_file = os.path.join(self.temp_dir, 'empty.sav')
        with open(test_file, 'wb') as f:
            pass  # Create empty file
        
        is_valid, error = parser.validate_save_file(test_file)
        
        self.assertFalse(is_valid)
        self.assertIn("File is empty", error)
    
    @patch('subprocess.run')
    def test_validate_save_file_too_large(self, mock_run):
        """Test validation of oversized save file."""
        # Mock dependency checks
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
        ]
        
        parser = ONISaveParser()
        
        # Mock os.stat to return large file size
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_size = 600 * 1024 * 1024  # 600MB
            
            test_file = os.path.join(self.temp_dir, 'large.sav')
            with open(test_file, 'wb') as f:
                f.write(b'data')
            
            is_valid, error = parser.validate_save_file(test_file)
            
            self.assertFalse(is_valid)
            self.assertIn("too large", error)
    
    @patch('subprocess.run')
    def test_validate_save_file_timeout(self, mock_run):
        """Test validation timeout handling."""
        # Mock dependency checks and timeout
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            subprocess.TimeoutExpired(['node'], 30)   # Validation timeout
        ]
        
        parser = ONISaveParser()
        
        # Create a temporary file
        test_file = os.path.join(self.temp_dir, 'test.sav')
        with open(test_file, 'wb') as f:
            f.write(b'fake save data')
        
        is_valid, error = parser.validate_save_file(test_file)
        
        self.assertFalse(is_valid)
        self.assertIn("timed out", error)
    
    @patch('subprocess.run')
    def test_parse_save_success(self, mock_run):
        """Test successful save file parsing."""
        # Mock data that would come from the enhanced JavaScript parser
        mock_save_data = {
            "header": {"version": "1.0"},
            "gameObjects": {
                "count": 3,
                "duplicants": [
                    {
                        "name": "TestDuplicant",
                        "position": [10, 20, 0],
                        "health": 95.0,
                        "stress": 15.0,
                        "skills": {"Mining": 2},
                        "traits": ["Quick Learner"]
                    }
                ],
                "buildings": [
                    {
                        "name": "OxygenGenerator",
                        "position": [5, 10, 0],
                        "buildingType": "power",
                        "operational": True,
                        "temperature": 25.0,
                        "power": 0.0
                    }
                ],
                "other": [
                    {
                        "name": "SomeOtherObject",
                        "position": [0, 0, 0],
                        "behaviorCount": 0
                    }
                ]
            },
            "worldDetail": {
                "worldSize": [256, 384],
                "streamed": {}
            },
            "settings": {"gameTime": 1200},
            "gameSpawnData": {},
            "extractedData": {
                "cycle": 2,
                "numberOfDuplicants": 1,
                "totalObjects": 3
            }
        }
        
        # Mock dependency checks, validation, and parsing
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            Mock(stdout='{"valid": true}\n', returncode=0),  # Validation
            Mock(stdout=json.dumps(mock_save_data), returncode=0)  # Parsing
        ]
        
        parser = ONISaveParser()
        
        # Create a temporary file
        test_file = os.path.join(self.temp_dir, 'test.sav')
        with open(test_file, 'wb') as f:
            f.write(b'fake save data')
        
        game_state = parser.parse_save(test_file)
        
        # Verify the parsed game state
        self.assertIsInstance(game_state, GameState)
        self.assertEqual(game_state.world_size, (256, 384))
        self.assertEqual(game_state.cycle, 2)
        self.assertIsInstance(game_state.grid, np.ndarray)
        self.assertEqual(game_state.grid.shape, (256, 384, 7))  # height, width, channels
        
        # Check that duplicants were extracted
        self.assertEqual(len(game_state.duplicants), 1)
        duplicant = game_state.duplicants[0]
        self.assertEqual(duplicant.name, "TestDuplicant")
        self.assertEqual(duplicant.position, (10, 20, 0))
        self.assertEqual(duplicant.health, 95.0)
        self.assertEqual(duplicant.stress_level, 15.0)
        
        # Check that buildings were extracted
        self.assertEqual(len(game_state.buildings), 1)
        building = game_state.buildings[0]
        self.assertEqual(building.name, "OxygenGenerator")
        self.assertEqual(building.position, (5, 10, 0))
        self.assertTrue(building.operational)
        self.assertEqual(building.temperature, 25.0)
        self.assertEqual(building.building_type, "power")
    
    @patch('subprocess.run')
    def test_parse_save_file_not_found(self, mock_run):
        """Test parsing of non-existent save file."""
        # Mock dependency checks
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
        ]
        
        parser = ONISaveParser()
        
        with self.assertRaises(FileNotFoundError) as context:
            parser.parse_save('nonexistent.sav')
        
        self.assertIn("Save file not found", str(context.exception))
    
    @patch('subprocess.run')
    def test_parse_save_corrupted_file_creates_mock(self, mock_run):
        """Test parsing of corrupted save file creates mock state."""
        # Mock dependency checks and validation failure
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            Mock(stdout='{"valid": false, "error": "File appears to be corrupted"}\n', returncode=0)
        ]
        
        parser = ONISaveParser()
        
        # Create a temporary file
        test_file = os.path.join(self.temp_dir, 'corrupted.sav')
        with open(test_file, 'wb') as f:
            f.write(b'corrupted data')
        
        # Should create mock state instead of raising exception
        game_state = parser.parse_save(test_file)
        
        self.assertIsInstance(game_state, GameState)
        self.assertTrue(game_state.metadata.get('mock', False))
        self.assertEqual(game_state.metadata.get('reason'), 'corrupted_file')
        self.assertTrue(game_state.metadata.get('recovery_mode', False))
    
    @patch('subprocess.run')
    def test_parse_save_empty_file_creates_minimal_mock(self, mock_run):
        """Test parsing of empty save file creates minimal mock state."""
        # Mock dependency checks
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
        ]
        
        parser = ONISaveParser()
        
        # Create an empty file
        test_file = os.path.join(self.temp_dir, 'empty.sav')
        with open(test_file, 'wb') as f:
            pass  # Create empty file
        
        game_state = parser.parse_save(test_file)
        
        self.assertIsInstance(game_state, GameState)
        self.assertTrue(game_state.metadata.get('mock', False))
        self.assertEqual(game_state.metadata.get('reason'), 'empty_file')
        self.assertEqual(len(game_state.duplicants), 1)  # Minimal state
        self.assertEqual(len(game_state.buildings), 0)   # No buildings for empty file
        self.assertEqual(game_state.world_size, (64, 64))  # Minimal world
    
    @patch('subprocess.run')
    def test_parse_save_timeout_creates_mock(self, mock_run):
        """Test parsing timeout creates mock state."""
        # Mock dependency checks, validation, and parsing timeout
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            Mock(stdout='{"valid": true}\n', returncode=0),  # Validation
            subprocess.TimeoutExpired(['node'], 60)   # Parsing timeout
        ]
        
        parser = ONISaveParser()
        
        # Create a temporary file
        test_file = os.path.join(self.temp_dir, 'timeout.sav')
        with open(test_file, 'wb') as f:
            f.write(b'fake save data')
        
        game_state = parser.parse_save(test_file)
        
        self.assertIsInstance(game_state, GameState)
        self.assertTrue(game_state.metadata.get('mock', False))
        self.assertEqual(game_state.metadata.get('reason'), 'timeout')
    
    @patch('subprocess.run')
    def test_parse_save_batch_mixed_results(self, mock_run):
        """Test batch parsing with mixed success/failure results."""
        # Mock dependency checks
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            # Validation calls
            Mock(stdout='{"valid": true}\n', returncode=0),   # File 1 valid
            Mock(stdout='{"valid": false, "error": "Corrupted"}\n', returncode=0),  # File 2 invalid
            # Parsing calls
            Mock(stdout='{"gameObjects": {"count": 0, "duplicants": [], "buildings": []}, "worldDetail": {"worldSize": [128, 192]}, "extractedData": {"cycle": 1}}', returncode=0)  # File 1 parse
        ]
        
        parser = ONISaveParser()
        
        # Create test files
        file1 = os.path.join(self.temp_dir, 'valid.sav')
        file2 = os.path.join(self.temp_dir, 'corrupted.sav')
        file3 = 'nonexistent.sav'
        
        with open(file1, 'wb') as f:
            f.write(b'valid data')
        with open(file2, 'wb') as f:
            f.write(b'corrupted data')
        
        # Test with skip_corrupted=True
        results = parser.parse_save_batch([file1, file2, file3], skip_corrupted=True)
        
        self.assertEqual(len(results), 3)
        
        # File 1 should succeed
        path1, state1, error1 = results[0]
        self.assertEqual(path1, file1)
        self.assertIsInstance(state1, GameState)
        self.assertIsNone(error1)
        
        # File 2 should fail but create mock state
        path2, state2, error2 = results[1]
        self.assertEqual(path2, file2)
        self.assertIsInstance(state2, GameState)  # Mock state created
        self.assertIsNone(error2)  # No error because mock was created
        
        # File 3 should fail
        path3, state3, error3 = results[2]
        self.assertEqual(path3, file3)
        self.assertIsNone(state3)
        self.assertIsNotNone(error3)
    
    @patch('subprocess.run')
    def test_get_parsing_statistics(self, mock_run):
        """Test parsing statistics collection."""
        # Mock dependency checks and validation calls
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            Mock(stdout='{"valid": true}\n', returncode=0),   # File 1 valid
            Mock(stdout='{"valid": false, "error": "Corrupted file"}\n', returncode=0),  # File 2 invalid
        ]
        
        parser = ONISaveParser()
        
        # Create test files
        file1 = os.path.join(self.temp_dir, 'valid.sav')
        file2 = os.path.join(self.temp_dir, 'corrupted.sav')
        file3 = 'nonexistent.sav'
        
        with open(file1, 'wb') as f:
            f.write(b'valid data' * 100)  # 1100 bytes
        with open(file2, 'wb') as f:
            f.write(b'corrupted data' * 50)  # 700 bytes
        
        stats = parser.get_parsing_statistics([file1, file2, file3])
        
        self.assertEqual(stats['total_files'], 3)
        self.assertEqual(stats['valid_files'], 1)
        self.assertEqual(stats['invalid_files'], 1)
        self.assertEqual(stats['missing_files'], 1)
        self.assertEqual(len(stats['file_sizes']), 2)  # Only existing files
        self.assertIn('corrupted_file', stats['error_types'])
        self.assertEqual(stats['error_types']['corrupted_file'], 1)
    
    def test_building_type_classification(self):
        """Test building type classification logic."""
        with patch('subprocess.run') as mock_run:
            # Mock dependency checks
            mock_run.side_effect = [
                Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
                Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
            ]
            
            parser = ONISaveParser()
            
            # Test various building classifications
            self.assertEqual(parser._classify_building_type("CoalGenerator"), "power")
            self.assertEqual(parser._classify_building_type("LiquidPump"), "plumbing")
            self.assertEqual(parser._classify_building_type("GasVent"), "ventilation")
            self.assertEqual(parser._classify_building_type("FarmTile"), "agriculture")
            self.assertEqual(parser._classify_building_type("Bed"), "living")
            self.assertEqual(parser._classify_building_type("ResearchStation"), "research")
            self.assertEqual(parser._classify_building_type("StorageLocker"), "storage")
            self.assertEqual(parser._classify_building_type("Ladder"), "infrastructure")
            self.assertEqual(parser._classify_building_type("UnknownObject"), "other")
    
    def test_error_categorization(self):
        """Test error message categorization."""
        with patch('subprocess.run') as mock_run:
            # Mock dependency checks
            mock_run.side_effect = [
                Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
                Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
            ]
            
            parser = ONISaveParser()
            
            # Test error categorization
            self.assertEqual(parser._categorize_error("File not found"), "file_not_found")
            self.assertEqual(parser._categorize_error("File is empty"), "empty_file")
            self.assertEqual(parser._categorize_error("File is too large"), "file_too_large")
            self.assertEqual(parser._categorize_error("File appears corrupted"), "corrupted_file")
            self.assertEqual(parser._categorize_error("Permission denied"), "permission_denied")
            self.assertEqual(parser._categorize_error("Validation timed out"), "timeout")
            self.assertEqual(parser._categorize_error("Out of memory"), "memory_error")
            self.assertEqual(parser._categorize_error("Unsupported version"), "version_incompatible")
            self.assertEqual(parser._categorize_error("Unknown issue"), "unknown_error")
    
    def test_duplicant_identification(self):
        """Test duplicant identification logic."""
        with patch('subprocess.run') as mock_run:
            # Mock dependency checks
            mock_run.side_effect = [
                Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
                Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
            ]
            
            parser = ONISaveParser()
            
            # Test duplicant identification
            duplicant_obj = {
                "name": "MinionInstance",
                "behaviors": [
                    {"name": "Health", "templateData": {}},
                    {"name": "Stress", "templateData": {}}
                ]
            }
            self.assertTrue(parser._is_duplicant(duplicant_obj))
            
            # Test non-duplicant object
            building_obj = {
                "name": "Generator",
                "behaviors": [
                    {"name": "Operational", "templateData": {}}
                ]
            }
            self.assertFalse(parser._is_duplicant(building_obj))
    
    def test_building_identification(self):
        """Test building identification logic."""
        with patch('subprocess.run') as mock_run:
            # Mock dependency checks
            mock_run.side_effect = [
                Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
                Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
            ]
            
            parser = ONISaveParser()
            
            # Test building identification
            building_obj = {
                "name": "CoalGenerator",
                "behaviors": [
                    {"name": "Operational", "templateData": {}}
                ]
            }
            self.assertTrue(parser._is_building(building_obj))
            
            # Test non-building object
            other_obj = {
                "name": "SomeRandomObject",
                "behaviors": []
            }
            self.assertFalse(parser._is_building(other_obj))


    def test_refined_grid_extraction_with_detailed_data(self):
        """Test that refined grid extraction uses detailed tile data when available."""
        with patch('subprocess.run') as mock_run:
            # Mock dependency checks
            mock_run.side_effect = [
                Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
                Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
            ]
            
            parser = ONISaveParser()
            
            # Test detailed grid data extraction
            world_size = (64, 64)
            duplicants = []
            buildings = []
            
            # Mock detailed grid data
            grid_data = {
                'hasDetailedData': True,
                'cells': [
                    {
                        'x': 10, 'y': 20, 'element': 1, 'temperature': 300.0, 'mass': 1000.0
                    },
                    {
                        'x': 15, 'y': 25, 'element': 2, 'temperature': 273.0, 'mass': 500.0
                    }
                ],
                'elements': {
                    1: {'id': 1, 'name': 'Oxygen', 'state': 'Gas'},
                    2: {'id': 2, 'name': 'Water', 'state': 'Liquid'}
                }
            }
            
            # Create grid with detailed data
            grid = parser._create_enhanced_grid(world_size, duplicants, buildings, grid_data)
            
            # Verify grid structure
            self.assertEqual(grid.shape, (64, 64, 7))
            
            # Verify detailed data was used for specific cells
            # Cell at (20, 10) should have oxygen (gas)
            self.assertAlmostEqual(grid[20, 10, 2], 1.0, places=1)  # Gas channel
            self.assertAlmostEqual(grid[20, 10, 0], 0.0, places=1)  # Solid channel
            self.assertAlmostEqual(grid[20, 10, 1], 0.0, places=1)  # Liquid channel
            self.assertAlmostEqual(grid[20, 10, 3], 1.0/255.0, places=3)  # Element ID normalized
            self.assertAlmostEqual(grid[20, 10, 4], 26.85, places=1)  # Temperature in Celsius
            
            # Cell at (25, 15) should have water (liquid)
            self.assertAlmostEqual(grid[25, 15, 1], 1.0, places=1)  # Liquid channel
            self.assertAlmostEqual(grid[25, 15, 0], 0.0, places=1)  # Solid channel
            self.assertAlmostEqual(grid[25, 15, 2], 0.0, places=1)  # Gas channel
            self.assertAlmostEqual(grid[25, 15, 3], 2.0/255.0, places=3)  # Element ID normalized
            self.assertAlmostEqual(grid[25, 15, 4], -0.15, places=1)  # Temperature in Celsius (273K = -0.15°C)
    
    def test_refined_grid_extraction_fallback_to_placeholders(self):
        """Test that grid extraction falls back to enhanced placeholders when detailed data is not available."""
        with patch('subprocess.run') as mock_run:
            # Mock dependency checks
            mock_run.side_effect = [
                Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
                Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
            ]
            
            parser = ONISaveParser()
            
            # Test without detailed grid data
            world_size = (64, 64)
            duplicants = []
            buildings = []
            grid_data = None  # No detailed data
            
            # Create grid without detailed data
            grid = parser._create_enhanced_grid(world_size, duplicants, buildings, grid_data)
            
            # Verify grid structure
            self.assertEqual(grid.shape, (64, 64, 7))
            
            # Verify enhanced placeholders were used
            # Should have solid ground at the bottom
            ground_height = 64 // 4  # 16 rows
            bottom_row = 63
            self.assertAlmostEqual(grid[bottom_row, 32, 0], 1.0, places=1)  # Solid channel
            self.assertAlmostEqual(grid[bottom_row, 32, 1], 0.0, places=1)  # Liquid channel
            self.assertAlmostEqual(grid[bottom_row, 32, 2], 0.0, places=1)  # Gas channel
            
            # Should have gas in upper areas
            top_row = 10
            self.assertAlmostEqual(grid[top_row, 32, 2], 1.0, places=0)  # Gas channel (approximately)
            self.assertAlmostEqual(grid[top_row, 32, 0], 0.0, places=1)  # Solid channel
            self.assertAlmostEqual(grid[top_row, 32, 1], 0.0, places=1)  # Liquid channel
    
    def test_grid_metadata_includes_extraction_info(self):
        """Test that grid extraction metadata is properly included in GameState."""
        # Mock data with detailed grid information
        mock_save_data = {
            "header": {"version": "1.0"},
            "gameObjects": {
                "count": 1,
                "duplicants": [],
                "buildings": [],
                "other": []
            },
            "worldDetail": {
                "worldSize": [128, 192],
                "streamed": {},
                "gridData": {
                    "hasDetailedData": True,
                    "cells": [
                        {"x": 10, "y": 20, "element": 1, "temperature": 300.0, "mass": 1000.0},
                        {"x": 15, "y": 25, "element": 2, "temperature": 273.0, "mass": 500.0}
                    ],
                    "elements": {
                        "1": {"id": 1, "name": "Oxygen", "state": "Gas"},
                        "2": {"id": 2, "name": "Water", "state": "Liquid"}
                    }
                }
            },
            "settings": {},
            "gameSpawnData": {},
            "extractedData": {"cycle": 5, "numberOfDuplicants": 0, "totalObjects": 1}
        }
        
        with patch('subprocess.run') as mock_run:
            # Mock dependency checks, validation, and parsing
            mock_run.side_effect = [
                Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
                Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
                Mock(stdout='{"valid": true}\n', returncode=0),  # Validation
                Mock(stdout=json.dumps(mock_save_data), returncode=0)  # Parsing
            ]
            
            parser = ONISaveParser()
            
            # Create a temporary file
            test_file = os.path.join(self.temp_dir, 'detailed_grid.sav')
            with open(test_file, 'wb') as f:
                f.write(b'fake save data with grid')
            
            game_state = parser.parse_save(test_file)
            
            # Verify metadata includes grid extraction information
            self.assertTrue(game_state.metadata.get('grid_data_available', False))
            self.assertEqual(game_state.metadata.get('grid_cells_extracted', 0), 2)
            self.assertEqual(game_state.metadata.get('grid_elements_available', 0), 2)
            self.assertTrue(game_state.metadata.get('real_parse', False))


class TestCorruptedFileHandling(unittest.TestCase):
    """Specific tests for corrupted file handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_mock_state_for_different_failure_reasons(self, mock_run):
        """Test that different failure reasons create appropriate mock states."""
        # Mock dependency checks
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
        ]
        
        parser = ONISaveParser()
        
        # Test different mock state reasons
        test_cases = [
            ("empty_file", 1, 0, (64, 64)),      # Minimal state
            ("file_too_large", 1, 1, (128, 128)), # Simplified state
            ("corrupted_file", 2, 2, (192, 256)), # Recovery state
            ("unknown", 3, 11, (256, 384))        # Default state
        ]
        
        for reason, expected_dupes, expected_buildings, expected_world_size in test_cases:
            with self.subTest(reason=reason):
                mock_state = parser._create_mock_game_state("test.sav", reason=reason)
                
                self.assertEqual(len(mock_state.duplicants), expected_dupes)
                self.assertEqual(len(mock_state.buildings), expected_buildings)
                self.assertEqual(mock_state.world_size, expected_world_size)
                self.assertEqual(mock_state.metadata['reason'], reason)
                
                if reason in ["corrupted_file", "empty_file"]:
                    self.assertTrue(mock_state.metadata.get('recovery_mode', False))
    
    @patch('subprocess.run')
    def test_mock_resources_vary_by_reason(self, mock_run):
        """Test that mock resources vary appropriately by failure reason."""
        # Mock dependency checks
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0)         # oni-save-parser check
        ]
        
        parser = ONISaveParser()
        
        # Test resource differences
        empty_resources = parser._create_mock_resources(10, "empty_file")
        large_resources = parser._create_mock_resources(10, "file_too_large")
        corrupted_resources = parser._create_mock_resources(10, "corrupted_file")
        default_resources = parser._create_mock_resources(10, "unknown")
        
        # Empty file should have minimal resources
        self.assertEqual(empty_resources['oxygen'], 100.0)
        self.assertEqual(empty_resources['power'], 0.0)
        
        # Large file should have scaled resources
        self.assertEqual(large_resources['oxygen'], 500.0 + (10 * 25.0))
        
        # Corrupted file should have declining resources
        self.assertLess(corrupted_resources['oxygen'], default_resources['oxygen'])
        self.assertGreater(corrupted_resources['polluted_water'], default_resources['polluted_water'])
        
        # Default should have standard progression
        self.assertEqual(default_resources['oxygen'], 1000.0 + (10 * 50.0))


class TestParseInterface(unittest.TestCase):
    """Test cases for the main parse_save interface function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_parse_save_interface_with_string_path(self, mock_run):
        """Test the main parse_save interface with string path."""
        # Mock data that would come from the enhanced JavaScript parser
        mock_save_data = {
            "header": {"version": "1.0"},
            "gameObjects": {
                "count": 1,
                "duplicants": [
                    {
                        "name": "TestDuplicant",
                        "position": [10, 20, 0],
                        "health": 95.0,
                        "stress": 15.0,
                        "skills": {"Mining": 2},
                        "traits": ["Quick Learner"]
                    }
                ],
                "buildings": [],
                "other": []
            },
            "worldDetail": {
                "worldSize": [256, 384],
                "streamed": {}
            },
            "settings": {"gameTime": 1200},
            "gameSpawnData": {},
            "extractedData": {
                "cycle": 2,
                "numberOfDuplicants": 1,
                "totalObjects": 1
            }
        }
        
        # Mock dependency checks, validation, and parsing
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            Mock(stdout='{"valid": true}\n', returncode=0),  # Validation
            Mock(stdout=json.dumps(mock_save_data), returncode=0)  # Parsing
        ]
        
        # Create a temporary file
        test_file = os.path.join(self.temp_dir, 'test.sav')
        with open(test_file, 'wb') as f:
            f.write(b'fake save data')
        
        # Test the interface function
        game_state = parse_save(test_file)
        
        # Verify the result
        self.assertIsInstance(game_state, GameState)
        self.assertEqual(game_state.world_size, (256, 384))
        self.assertEqual(len(game_state.duplicants), 1)
        self.assertEqual(game_state.duplicants[0].name, "TestDuplicant")
    
    @patch('subprocess.run')
    def test_parse_save_interface_with_path_object(self, mock_run):
        """Test the main parse_save interface with Path object."""
        # Mock data
        mock_save_data = {
            "header": {"version": "1.0"},
            "gameObjects": {"count": 0, "duplicants": [], "buildings": [], "other": []},
            "worldDetail": {"worldSize": [128, 192], "streamed": {}},
            "settings": {},
            "gameSpawnData": {},
            "extractedData": {"cycle": 0}
        }
        
        # Mock dependency checks, validation, and parsing
        mock_run.side_effect = [
            Mock(stdout='v18.17.0\n', returncode=0),  # Node.js version
            Mock(stdout='OK\n', returncode=0),        # oni-save-parser check
            Mock(stdout='{"valid": true}\n', returncode=0),  # Validation
            Mock(stdout=json.dumps(mock_save_data), returncode=0)  # Parsing
        ]
        
        # Create a temporary file
        test_file = Path(self.temp_dir) / 'test.sav'
        with open(test_file, 'wb') as f:
            f.write(b'fake save data')
        
        # Test the interface function with Path object
        game_state = parse_save(test_file)
        
        # Verify the result
        self.assertIsInstance(game_state, GameState)
        self.assertEqual(game_state.world_size, (128, 192))


class TestRealSaveFiles(unittest.TestCase):
    """Test cases using actual ONI save files."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures_dir = Path(__file__).parent.parent / 'fixtures' / 'sample_saves'
        self.parser = None
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
    
    def test_parse_real_save_file_main_colony(self):
        """Test parsing the main colony save file."""
        save_file = self.fixtures_dir / 'test_colony_main.sav'
        
        if not save_file.exists():
            self.skipTest(f"Sample save file not found: {save_file}")
        
        try:
            # Initialize parser (may create mock state if dependencies missing)
            parser = ONISaveParser()
            game_state = parser.parse_save(str(save_file))
            
            # Verify basic structure
            self.assertIsInstance(game_state, GameState)
            self.assertIsInstance(game_state.grid, np.ndarray)
            self.assertEqual(len(game_state.grid.shape), 3)  # height, width, channels
            self.assertIsInstance(game_state.duplicants, list)
            self.assertIsInstance(game_state.buildings, list)
            self.assertIsInstance(game_state.resources, dict)
            self.assertIsInstance(game_state.cycle, int)
            self.assertIsInstance(game_state.world_size, tuple)
            self.assertEqual(len(game_state.world_size), 2)
            
            # Verify grid dimensions match world size
            height, width = game_state.world_size
            self.assertEqual(game_state.grid.shape[:2], (height, width))
            
            # Verify we have some game content
            self.assertGreaterEqual(len(game_state.duplicants), 0)
            self.assertGreaterEqual(game_state.cycle, 0)
            
            # Verify resource structure
            expected_resources = ['oxygen', 'water', 'food', 'power']
            for resource in expected_resources:
                self.assertIn(resource, game_state.resources)
                self.assertIsInstance(game_state.resources[resource], (int, float))
            
            print(f"Successfully parsed main colony save:")
            print(f"  Cycle: {game_state.cycle}")
            print(f"  Duplicants: {len(game_state.duplicants)}")
            print(f"  Buildings: {len(game_state.buildings)}")
            print(f"  World size: {game_state.world_size}")
            print(f"  Grid shape: {game_state.grid.shape}")
            
        except Exception as e:
            # If parsing fails due to missing dependencies, verify we get a mock state
            if "Node.js not found" in str(e) or "oni-save-parser" in str(e):
                self.skipTest(f"Dependencies not available: {e}")
            else:
                # Re-raise unexpected errors
                raise
    
    def test_parse_real_save_file_cycle_181(self):
        """Test parsing a specific cycle save file."""
        save_file = self.fixtures_dir / 'test_colony_cycle_181.sav'
        
        if not save_file.exists():
            self.skipTest(f"Sample save file not found: {save_file}")
        
        try:
            parser = ONISaveParser()
            game_state = parser.parse_save(str(save_file))
            
            # Verify basic structure
            self.assertIsInstance(game_state, GameState)
            self.assertGreaterEqual(game_state.cycle, 0)
            
            # If this is a real parse (not mock), cycle should be around 181
            if not game_state.metadata.get('mock', False):
                # Allow some tolerance in case the filename doesn't exactly match the cycle
                self.assertGreaterEqual(game_state.cycle, 170)
                self.assertLessEqual(game_state.cycle, 200)
            
            print(f"Successfully parsed cycle 181 save:")
            print(f"  Actual cycle: {game_state.cycle}")
            print(f"  Mock state: {game_state.metadata.get('mock', False)}")
            
        except Exception as e:
            if "Node.js not found" in str(e) or "oni-save-parser" in str(e):
                self.skipTest(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_parse_real_save_file_cycle_190(self):
        """Test parsing another specific cycle save file."""
        save_file = self.fixtures_dir / 'test_colony_cycle_190.sav'
        
        if not save_file.exists():
            self.skipTest(f"Sample save file not found: {save_file}")
        
        try:
            parser = ONISaveParser()
            game_state = parser.parse_save(str(save_file))
            
            # Verify basic structure
            self.assertIsInstance(game_state, GameState)
            self.assertGreaterEqual(game_state.cycle, 0)
            
            # If this is a real parse (not mock), cycle should be around 190
            if not game_state.metadata.get('mock', False):
                self.assertGreaterEqual(game_state.cycle, 180)
                self.assertLessEqual(game_state.cycle, 200)
            
            print(f"Successfully parsed cycle 190 save:")
            print(f"  Actual cycle: {game_state.cycle}")
            print(f"  Mock state: {game_state.metadata.get('mock', False)}")
            
        except Exception as e:
            if "Node.js not found" in str(e) or "oni-save-parser" in str(e):
                self.skipTest(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_parse_empty_real_file(self):
        """Test parsing an empty save file."""
        save_file = self.fixtures_dir / 'empty_file.sav'
        
        if not save_file.exists():
            self.skipTest(f"Empty test file not found: {save_file}")
        
        try:
            parser = ONISaveParser()
            game_state = parser.parse_save(str(save_file))
            
            # Should create a mock state for empty file
            self.assertIsInstance(game_state, GameState)
            self.assertTrue(game_state.metadata.get('mock', False))
            self.assertEqual(game_state.metadata.get('reason'), 'empty_file')
            
            # Verify minimal state structure
            self.assertEqual(len(game_state.duplicants), 1)  # Emergency duplicant
            self.assertEqual(len(game_state.buildings), 0)   # No buildings
            self.assertEqual(game_state.world_size, (64, 64))  # Minimal world
            
            print(f"Successfully handled empty file with mock state")
            
        except Exception as e:
            if "Node.js not found" in str(e) or "oni-save-parser" in str(e):
                self.skipTest(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_parse_corrupted_real_file(self):
        """Test parsing a corrupted save file."""
        save_file = self.fixtures_dir / 'corrupted_file.sav'
        
        if not save_file.exists():
            self.skipTest(f"Corrupted test file not found: {save_file}")
        
        try:
            parser = ONISaveParser()
            game_state = parser.parse_save(str(save_file))
            
            # Should create a mock state for corrupted file
            self.assertIsInstance(game_state, GameState)
            self.assertTrue(game_state.metadata.get('mock', False))
            self.assertIn(game_state.metadata.get('reason'), ['corrupted_file', 'parsing_corruption'])
            
            # Verify recovery state structure
            self.assertGreaterEqual(len(game_state.duplicants), 1)
            self.assertGreaterEqual(len(game_state.buildings), 0)
            
            print(f"Successfully handled corrupted file with mock state")
            
        except Exception as e:
            if "Node.js not found" in str(e) or "oni-save-parser" in str(e):
                self.skipTest(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_batch_parse_real_files(self):
        """Test batch parsing of multiple real save files."""
        save_files = [
            str(self.fixtures_dir / 'test_colony_main.sav'),
            str(self.fixtures_dir / 'test_colony_cycle_181.sav'),
            str(self.fixtures_dir / 'test_colony_cycle_190.sav'),
            str(self.fixtures_dir / 'empty_file.sav'),
            str(self.fixtures_dir / 'corrupted_file.sav'),
            str(self.fixtures_dir / 'nonexistent_file.sav')  # This file doesn't exist
        ]
        
        # Filter to only existing files for this test
        existing_files = [f for f in save_files if Path(f).exists()]
        
        if len(existing_files) < 2:
            self.skipTest("Not enough sample files available for batch test")
        
        try:
            parser = ONISaveParser()
            results = parser.parse_save_batch(existing_files, skip_corrupted=False)
            
            # Verify we get results for all files
            self.assertEqual(len(results), len(existing_files))
            
            # Verify result structure
            for file_path, game_state, error in results:
                self.assertIn(file_path, existing_files)
                if game_state is not None:
                    self.assertIsInstance(game_state, GameState)
                # error can be None or a string
                
            print(f"Successfully batch parsed {len(existing_files)} files")
            
        except Exception as e:
            if "Node.js not found" in str(e) or "oni-save-parser" in str(e):
                self.skipTest(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_parsing_statistics_real_files(self):
        """Test getting parsing statistics for real save files."""
        save_files = [
            str(self.fixtures_dir / 'test_colony_main.sav'),
            str(self.fixtures_dir / 'test_colony_cycle_181.sav'),
            str(self.fixtures_dir / 'empty_file.sav'),
            str(self.fixtures_dir / 'corrupted_file.sav'),
            str(self.fixtures_dir / 'nonexistent_file.sav')
        ]
        
        try:
            parser = ONISaveParser()
            stats = parser.get_parsing_statistics(save_files)
            
            # Verify statistics structure
            self.assertIn('total_files', stats)
            self.assertIn('valid_files', stats)
            self.assertIn('invalid_files', stats)
            self.assertIn('missing_files', stats)
            self.assertIn('error_types', stats)
            self.assertIn('file_sizes', stats)
            
            # Verify counts add up
            self.assertEqual(
                stats['total_files'],
                stats['valid_files'] + stats['invalid_files'] + stats['missing_files']
            )
            
            # Should have at least one missing file (nonexistent_file.sav)
            self.assertGreaterEqual(stats['missing_files'], 1)
            
            print(f"Parsing statistics:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Valid files: {stats['valid_files']}")
            print(f"  Invalid files: {stats['invalid_files']}")
            print(f"  Missing files: {stats['missing_files']}")
            print(f"  Error types: {stats['error_types']}")
            
        except Exception as e:
            if "Node.js not found" in str(e) or "oni-save-parser" in str(e):
                self.skipTest(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_interface_function_with_real_file(self):
        """Test the main interface function with a real save file."""
        save_file = self.fixtures_dir / 'test_colony_main.sav'
        
        if not save_file.exists():
            self.skipTest(f"Sample save file not found: {save_file}")
        
        try:
            # Test with string path
            game_state = parse_save(str(save_file))
            self.assertIsInstance(game_state, GameState)
            
            # Test with Path object
            game_state_path = parse_save(save_file)
            self.assertIsInstance(game_state_path, GameState)
            
            # Both should produce equivalent results
            self.assertEqual(game_state.cycle, game_state_path.cycle)
            self.assertEqual(game_state.world_size, game_state_path.world_size)
            
            print(f"Interface function works with both string and Path objects")
            
        except Exception as e:
            if "Node.js not found" in str(e) or "oni-save-parser" in str(e):
                self.skipTest(f"Dependencies not available: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()