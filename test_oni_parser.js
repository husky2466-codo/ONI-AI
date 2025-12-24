const fs = require('fs');
const path = require('path');

// Test if the oni-save-parser module can be imported
try {
    const parser = require('oni-save-parser');
    console.log('✅ oni-save-parser module imported successfully');
    
    // Check available functions
    console.log('📋 Available functions:');
    console.log('  - parseSaveGame:', typeof parser.parseSaveGame);
    console.log('  - writeSaveGame:', typeof parser.writeSaveGame);
    
    // Check for TypeScript definitions
    const packagePath = path.join(__dirname, 'node_modules', 'oni-save-parser', 'package.json');
    if (fs.existsSync(packagePath)) {
        const packageInfo = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
        console.log('📦 Package info:');
        console.log('  - Version:', packageInfo.version);
        console.log('  - TypeScript types:', packageInfo.types || packageInfo.typings || 'Not specified');
        console.log('  - Main entry:', packageInfo.main);
    }
    
    // Check for type definitions file
    const typesPath = path.join(__dirname, 'node_modules', 'oni-save-parser', 'dts', 'index.d.ts');
    if (fs.existsSync(typesPath)) {
        console.log('✅ TypeScript definitions found');
        
        // Read a sample of the type definitions
        const typesContent = fs.readFileSync(typesPath, 'utf8');
        const lines = typesContent.split('\n').slice(0, 10);
        console.log('📝 Sample type definitions:');
        lines.forEach(line => {
            if (line.trim()) console.log('    ' + line.trim());
        });
    } else {
        console.log('❌ TypeScript definitions not found at expected location');
    }
    
    console.log('\n🎯 Library evaluation complete - ready for integration');
    
} catch (error) {
    console.error('❌ Error importing oni-save-parser:', error.message);
    process.exit(1);
}