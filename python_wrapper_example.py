#!/usr/bin/env python3
"""
Example Python wrapper for the oni-save-parser JavaScript library.
This demonstrates how to integrate the JS library with Python for the ONI AI project.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ONISaveParser:
    """Python wrapper for the oni-save-parser JavaScript library."""
    
    def __init__(self, node_script_path: Optional[str] = None):
        """
        Initialize the ONI save parser wrapper.
        
        Args:
            node_script_path: Path to the Node.js parsing script. If None, uses default.
        """
        self.node_script_path = node_script_path or self._create_default_script()
        self._verify_dependencies()
    
    def _verify_dependencies(self) -> None:
        """Verify that Node.js and oni-save-parser are available."""
        try:
            # Check Node.js
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Node.js version: {result.stdout.strip()}")
            
            # Check oni-save-parser
            test_script = '''
            try {
                const parser = require('oni-save-parser');
                console.log('oni-save-parser available');
            } catch (e) {
                console.error('oni-save-parser not found:', e.message);
                process.exit(1);
            }
            '''
            
            result = subprocess.run(['node', '-e', test_script], 
                                  capture_output=True, text=True, check=True)
            print("✅ oni-save-parser library available")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Dependency check failed: {e}")
    
    def _create_default_script(self) -> str:
        """Create a default Node.js script for parsing saves."""
        script_content = '''
const fs = require('fs');
const { parseSaveGame, writeSaveGame } = require('oni-save-parser');

const command = process.argv[2];
const filePath = process.argv[3];

if (command === 'parse') {
    try {
        const fileData = fs.readFileSync(filePath);
        const saveData = parseSaveGame(fileData.buffer);
        
        // Extract key information for Python
        const extracted = {
            header: saveData.header,
            gameObjects: saveData.gameObjects ? {
                count: saveData.gameObjects.length,
                // Sample first few objects for structure analysis
                sample: saveData.gameObjects.slice(0, 3).map(obj => ({
                    name: obj.name,
                    position: obj.position,
                    rotation: obj.rotation,
                    scale: obj.scale,
                    behaviors: obj.behaviors ? obj.behaviors.length : 0
                }))
            } : null,
            worldDetail: saveData.worldDetail ? {
                worldSize: saveData.worldDetail.worldSize,
                streamed: saveData.worldDetail.streamed
            } : null
        };
        
        console.log(JSON.stringify(extracted, null, 2));
    } catch (error) {
        console.error('Parse error:', error.message);
        process.exit(1);
    }
} else {
    console.error('Unknown command:', command);
    process.exit(1);
}
        '''
        
        script_path = 'oni_parser_bridge.js'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def parse_save(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an ONI save file and return structured data.
        
        Args:
            file_path: Path to the .sav file
            
        Returns:
            Dictionary containing parsed save data
            
        Raises:
            FileNotFoundError: If save file doesn't exist
            RuntimeError: If parsing fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Save file not found: {file_path}")
        
        try:
            result = subprocess.run([
                'node', self.node_script_path, 'parse', file_path
            ], capture_output=True, text=True, check=True)
            
            return json.loads(result.stdout)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to parse save file: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode parser output: {e}")


def main():
    """Example usage of the ONI save parser wrapper."""
    print("🔧 ONI Save Parser Python Wrapper Test")
    print("=" * 50)
    
    try:
        parser = ONISaveParser()
        print("✅ Parser initialized successfully")
        
        # Note: This would require an actual ONI save file to test
        print("\n📝 Parser ready for use. Example usage:")
        print("   parser.parse_save('path/to/save.sav')")
        print("\n🎯 Integration test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()