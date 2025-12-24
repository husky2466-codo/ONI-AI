"""
Integration tests for the ONI save parser Python-Node.js bridge.
"""

import os
import tempfile
import unittest
from pathlib import Path

from src.data.parsers import ONISaveParser


class TestParserIntegration(unittest.TestCase):
    """Integration tests for the complete parsing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parser_initialization(self):
        """Test that the parser can be initialized with real dependencies."""
        try:
            parser = ONISaveParser()
            self.assertIsNotNone(parser)
            self.assertTrue(os.path.exists(parser.node_script_path))
            print("✅ Parser initialization successful")
        except RuntimeError as e:
            self.skipTest(f"Dependencies not available: {e}")
    
    def test_bridge_script_creation(self):
        """Test that the Node.js bridge script is created correctly."""
        try:
            parser = ONISaveParser()
            
            # Verify the bridge script exists and has the expected content
            self.assertTrue(os.path.exists(parser.node_script_path))
            
            with open(parser.node_script_path, 'r') as f:
                content = f.read()
            
            # Check for key components
            self.assertIn("parseSaveGame", content)
            self.assertIn("oni-save-parser", content)
            self.assertIn("JSON.stringify", content)
            
            print("✅ Bridge script creation successful")
        except RuntimeError as e:
            self.skipTest(f"Dependencies not available: {e}")
    
    def test_dependency_verification(self):
        """Test that dependency verification works with real environment."""
        try:
            parser = ONISaveParser()
            # If we get here without exception, dependencies are available
            print("✅ All dependencies verified successfully")
            print(f"   Node.js bridge script: {parser.node_script_path}")
        except RuntimeError as e:
            self.skipTest(f"Dependencies not available: {e}")


if __name__ == '__main__':
    unittest.main()