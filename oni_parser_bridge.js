
const fs = require('fs');
const util = require('util');

// Patch 1: Fix util.isObject compatibility for newer Node.js versions
if (!util.isObject) {
    util.isObject = function(arg) {
        return typeof arg === 'object' && arg !== null && !Array.isArray(arg);
    };
}

// Patch 2: Bypass version validation for newer save formats
const Module = require('module');
const originalRequire = Module.prototype.require;
Module.prototype.require = function(id) {
    const module = originalRequire.apply(this, arguments);
    if (id.includes('version-validator') || (module && module.validateVersion)) {
        module.validateVersion = function(major, minor, strictness = "minor") {
            // Bypass version check - log but don't throw
            return;
        };
    }
    return module;
};

const { parseSaveGame } = require('oni-save-parser');

const command = process.argv[2];
const filePath = process.argv[3];

function isDuplicant(obj) {
    const name = (obj.name || '').toLowerCase();
    if (name.includes('minion') && !name.includes('proxy')) {
        return true;
    }
    if (obj.behaviors) {
        const behaviorNames = obj.behaviors.map(b => (b.name || '').toLowerCase()).join(' ');
        return behaviorNames.includes('health') || behaviorNames.includes('stress') || 
               behaviorNames.includes('needs') || behaviorNames.includes('skills');
    }
    return false;
}

function isBuilding(obj) {
    const name = (obj.name || '').toLowerCase();
    const buildingKeywords = [
        'generator', 'pump', 'vent', 'pipe', 'wire', 'ladder', 'tile',
        'door', 'bed', 'toilet', 'research', 'storage', 'farm', 'ranch',
        'battery', 'switch', 'valve', 'filter', 'heater', 'cooler'
    ];
    return buildingKeywords.some(keyword => name.includes(keyword));
}

function extractDuplicantData(obj) {
    const duplicant = {
        name: obj.name || 'Unknown Duplicant',
        position: obj.position || [0, 0, 0],
        health: 100.0,
        stress: 0.0,
        skills: {},
        traits: [],
        needs: {}
    };
    
    if (obj.behaviors) {
        obj.behaviors.forEach(behavior => {
            const behaviorName = (behavior.name || '').toLowerCase();
            const templateData = behavior.templateData || {};
            
            if (behaviorName.includes('health')) {
                duplicant.health = templateData.health || templateData.hitPoints || 100.0;
            } else if (behaviorName.includes('stress')) {
                duplicant.stress = templateData.stress || templateData.stressLevel || 0.0;
            } else if (behaviorName.includes('skill')) {
                Object.entries(templateData).forEach(([key, value]) => {
                    if (typeof value === 'number') {
                        duplicant.skills[key] = value;
                    }
                });
            } else if (behaviorName.includes('trait')) {
                if (templateData.traits && Array.isArray(templateData.traits)) {
                    duplicant.traits = templateData.traits;
                }
            }
        });
    }
    
    return duplicant;
}

function extractBuildingData(obj) {
    const building = {
        name: obj.name || 'Unknown Building',
        position: obj.position || [0, 0, 0],
        buildingType: classifyBuildingType(obj.name || ''),
        operational: true,
        temperature: 20.0,
        power: 0.0
    };
    
    if (obj.behaviors) {
        obj.behaviors.forEach(behavior => {
            const behaviorName = (behavior.name || '').toLowerCase();
            const templateData = behavior.templateData || {};
            
            if (behaviorName.includes('operational')) {
                building.operational = templateData.operational !== false;
            } else if (behaviorName.includes('temperature')) {
                building.temperature = templateData.temperature || 20.0;
            } else if (behaviorName.includes('power') || behaviorName.includes('energy')) {
                building.power = templateData.power || templateData.energy || 0.0;
            }
        });
    }
    
    return building;
}

function extractGridData(saveData, worldSize) {
    const [width, height] = worldSize;
    const gridData = {
        cells: [],
        elements: {},
        temperatures: {},
        masses: {},
        materials: {},
        hasDetailedData: false
    };
    
    try {
        // Look for grid data in multiple locations within the save data
        let streamedData = null;
        
        // First, try the world.streamed location (most likely location for grid data)
        if (saveData.world && saveData.world.streamed) {
            streamedData = saveData.world.streamed;
        }
        
        // If no streamed data, try worldDetail.streamed location
        if ((!streamedData || Object.keys(streamedData).length === 0) && 
            saveData.worldDetail && saveData.worldDetail.streamed) {
            streamedData = saveData.worldDetail.streamed;
        }
        
        // If still no streamed data, look for grid data in other locations
        if (!streamedData || Object.keys(streamedData).length === 0) {
            // Try to find grid data in the main save structure
            if (saveData.grid) {
                streamedData = saveData.grid;
            } else if (saveData.cells) {
                streamedData = { cells: saveData.cells };
            }
        }
        
        // Look for cell data in various possible formats
        let cellData = null;
        if (streamedData) {
            // Check if streamedData has numeric keys (chunk indices)
            const chunkKeys = Object.keys(streamedData).filter(key => !isNaN(key));
            if (chunkKeys.length > 0) {
                // This is chunked data - combine all chunks
                cellData = [];
                chunkKeys.forEach(chunkKey => {
                    const chunk = streamedData[chunkKey];
                    if (chunk && typeof chunk === 'object') {
                        // Check if chunk has sub-chunks (numeric keys)
                        const subChunkKeys = Object.keys(chunk).filter(key => !isNaN(key));
                        if (subChunkKeys.length > 0) {
                            // Process sub-chunks
                            subChunkKeys.forEach(subChunkKey => {
                                const subChunk = chunk[subChunkKey];
                                if (Array.isArray(subChunk)) {
                                    cellData = cellData.concat(subChunk);
                                } else if (subChunk && subChunk.cells && Array.isArray(subChunk.cells)) {
                                    cellData = cellData.concat(subChunk.cells);
                                }
                            });
                        } else if (chunk.cells && Array.isArray(chunk.cells)) {
                            cellData = cellData.concat(chunk.cells);
                        } else if (Array.isArray(chunk)) {
                            cellData = cellData.concat(chunk);
                        }
                    }
                });
            } else {
                // Regular cell data format
                cellData = streamedData.cells || streamedData.Cell || 
                          streamedData.cellData || streamedData.CellData ||
                          streamedData.worldCells || streamedData.WorldCells;
            }
        }
        
        if (cellData && Array.isArray(cellData)) {
            gridData.hasDetailedData = true;
            cellData.forEach((cell, index) => {
                if (cell && typeof cell === 'object') {
                    const cellInfo = {
                        index: index,
                        x: index % width,
                        y: Math.floor(index / width),
                        element: cell.element || cell.elementIdx || cell.elementId || 0,
                        temperature: cell.temperature || cell.temp || 293.15, // Default to 20°C in Kelvin
                        mass: cell.mass || cell.Mass || 0,
                        insulation: cell.insulation || cell.Insulation || 0,
                        strengthInfo: cell.strengthInfo || cell.StrengthInfo || {},
                        diseaseCount: cell.diseaseCount || cell.DiseaseCount || 0,
                        diseaseIdx: cell.diseaseIdx || cell.DiseaseIdx || -1
                    };
                    gridData.cells.push(cellInfo);
                }
            });
        }
        
        // Look for element information in various locations
        let elementData = null;
        if (streamedData) {
            elementData = streamedData.elements || streamedData.Element || 
                         streamedData.elementData || streamedData.ElementData;
        }
        
        // Also check the main save data for element definitions
        if (!elementData && saveData.elements) {
            elementData = saveData.elements;
        }
        
        if (elementData && Array.isArray(elementData)) {
            elementData.forEach((element, index) => {
                if (element && typeof element === 'object') {
                    gridData.elements[index] = {
                        id: element.id || element.elementId || index,
                        name: element.name || element.Name || `Element_${index}`,
                        state: element.state || element.State || 'Gas', // Gas, Liquid, Solid
                        lowTemp: element.lowTemp || element.LowTemp || 0,
                        highTemp: element.highTemp || element.HighTemp || 1000,
                        specificHeatCapacity: element.specificHeatCapacity || element.SpecificHeatCapacity || 1000,
                        thermalConductivity: element.thermalConductivity || element.ThermalConductivity || 1,
                        solidSurfaceAreaMultiplier: element.solidSurfaceAreaMultiplier || 1,
                        liquidSurfaceAreaMultiplier: element.liquidSurfaceAreaMultiplier || 1,
                        gasSurfaceAreaMultiplier: element.gasSurfaceAreaMultiplier || 1
                    };
                }
            });
        }
        
        // Look for separate temperature, mass, and material arrays
        if (streamedData) {
            // Temperature data
            const tempData = streamedData.temperature || streamedData.Temperature || 
                            streamedData.temperatures || streamedData.Temperatures;
            if (tempData && Array.isArray(tempData)) {
                tempData.forEach((temp, index) => {
                    gridData.temperatures[index] = temp;
                });
            }
            
            // Mass data
            const massData = streamedData.mass || streamedData.Mass || 
                            streamedData.masses || streamedData.Masses;
            if (massData && Array.isArray(massData)) {
                massData.forEach((mass, index) => {
                    gridData.masses[index] = mass;
                });
            }
            
            // Material/element index data
            const materialData = streamedData.elementIdx || streamedData.ElementIdx || 
                               streamedData.elements || streamedData.Elements ||
                               streamedData.materials || streamedData.Materials;
            if (materialData && Array.isArray(materialData)) {
                materialData.forEach((elementIdx, index) => {
                    gridData.materials[index] = elementIdx;
                });
            }
        }
        
        // If we don't have detailed cell data but have separate arrays, combine them
        if (!gridData.hasDetailedData && (Object.keys(gridData.temperatures).length > 0 || 
                                         Object.keys(gridData.masses).length > 0 || 
                                         Object.keys(gridData.materials).length > 0)) {
            gridData.hasDetailedData = true;
            const maxIndex = Math.max(
                Math.max(...Object.keys(gridData.temperatures).map(k => parseInt(k)), -1),
                Math.max(...Object.keys(gridData.masses).map(k => parseInt(k)), -1),
                Math.max(...Object.keys(gridData.materials).map(k => parseInt(k)), -1),
                width * height - 1
            );
            
            for (let i = 0; i <= maxIndex && i < width * height; i++) {
                const cellInfo = {
                    index: i,
                    x: i % width,
                    y: Math.floor(i / width),
                    element: gridData.materials[i] || 0,
                    temperature: gridData.temperatures[i] || 293.15,
                    mass: gridData.masses[i] || 0,
                    insulation: 0,
                    strengthInfo: {},
                    diseaseCount: 0,
                    diseaseIdx: -1
                };
                gridData.cells.push(cellInfo);
            }
        }
        
        // If still no detailed data, try to extract from game objects that represent world tiles
        if (!gridData.hasDetailedData && saveData.gameObjects) {
            const worldTiles = saveData.gameObjects.filter(obj => {
                const name = (obj.name || '').toLowerCase();
                return name.includes('cell') || name.includes('tile') || name.includes('element');
            });
            
            if (worldTiles.length > 0) {
                gridData.hasDetailedData = true;
                worldTiles.forEach((tile, index) => {
                    const position = tile.position || [0, 0, 0];
                    const x = Math.floor(position[0]);
                    const y = Math.floor(position[1]);
                    
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        const cellInfo = {
                            index: y * width + x,
                            x: x,
                            y: y,
                            element: 0, // Default element
                            temperature: 293.15, // Default temperature
                            mass: 1000, // Default mass
                            insulation: 0,
                            strengthInfo: {},
                            diseaseCount: 0,
                            diseaseIdx: -1
                        };
                        
                        // Try to extract more data from behaviors
                        if (tile.behaviors) {
                            tile.behaviors.forEach(behavior => {
                                const templateData = behavior.templateData || {};
                                const behaviorName = (behavior.name || '').toLowerCase();
                                
                                if (behaviorName.includes('temperature')) {
                                    cellInfo.temperature = templateData.temperature || cellInfo.temperature;
                                } else if (behaviorName.includes('element')) {
                                    cellInfo.element = templateData.element || templateData.elementId || cellInfo.element;
                                } else if (behaviorName.includes('mass')) {
                                    cellInfo.mass = templateData.mass || cellInfo.mass;
                                }
                            });
                        }
                        
                        gridData.cells.push(cellInfo);
                    }
                });
            }
        }
        
    } catch (error) {
        console.error('Error extracting grid data:', error.message);
        gridData.hasDetailedData = false;
    }
    
    return gridData;
}

function classifyBuildingType(name) {
    const nameLower = name.toLowerCase();
    if (nameLower.includes('generator') || nameLower.includes('battery') || nameLower.includes('wire')) {
        return 'power';
    } else if (nameLower.includes('pump') || nameLower.includes('pipe') || nameLower.includes('valve')) {
        return 'plumbing';
    } else if (nameLower.includes('vent') || nameLower.includes('fan') || nameLower.includes('scrubber')) {
        return 'ventilation';
    } else if (nameLower.includes('farm') || nameLower.includes('planter') || nameLower.includes('hydroponic')) {
        return 'agriculture';
    } else if (nameLower.includes('bed') || nameLower.includes('toilet') || nameLower.includes('shower')) {
        return 'living';
    } else if (nameLower.includes('research') || nameLower.includes('computer') || nameLower.includes('telescope')) {
        return 'research';
    } else if (nameLower.includes('storage') || nameLower.includes('container') || nameLower.includes('locker')) {
        return 'storage';
    } else if (nameLower.includes('door') || nameLower.includes('ladder') || nameLower.includes('tile')) {
        return 'infrastructure';
    } else {
        return 'other';
    }
}

if (command === 'parse') {
    try {
        const fileData = fs.readFileSync(filePath);
        const saveData = parseSaveGame(fileData.buffer);
        
        // Extract comprehensive game state information with enhanced data extraction
        const duplicants = [];
        const buildings = [];
        const otherObjects = [];
        
        if (saveData.gameObjects) {
            saveData.gameObjects.forEach(obj => {
                if (isDuplicant(obj)) {
                    duplicants.push(extractDuplicantData(obj));
                } else if (isBuilding(obj)) {
                    buildings.push(extractBuildingData(obj));
                } else {
                    otherObjects.push({
                        name: obj.name || '',
                        position: obj.position || [0, 0, 0],
                        behaviorCount: obj.behaviors ? obj.behaviors.length : 0
                    });
                }
            });
        }
        
        // Extract cycle and world information
        let cycle = 0;
        let worldSize = [256, 384]; // Default size
        let numberOfDuplicants = duplicants.length;
        
        if (saveData.header && saveData.header.gameInfo) {
            const gameInfo = typeof saveData.header.gameInfo === 'string' 
                ? JSON.parse(saveData.header.gameInfo) 
                : saveData.header.gameInfo;
            cycle = gameInfo.numberOfCycles || 0;
            numberOfDuplicants = gameInfo.numberOfDuplicants || duplicants.length;
        }
        
        if (saveData.worldDetail && saveData.worldDetail.worldSize) {
            worldSize = saveData.worldDetail.worldSize;
        }
        
        gridData = extractGridData(saveData, worldSize);

        const extracted = {
            header: saveData.header || {},
            gameObjects: {
                count: saveData.gameObjects ? saveData.gameObjects.length : 0,
                duplicants: duplicants,
                buildings: buildings,
                other: otherObjects
            },
            worldDetail: {
                worldSize: worldSize,
                streamed: saveData.worldDetail ? saveData.worldDetail.streamed || {} : {},
                gridData: gridData
            },
            settings: saveData.settings || {},
            gameSpawnData: saveData.gameSpawnData || {},
            extractedData: {
                cycle: cycle,
                numberOfDuplicants: numberOfDuplicants,
                totalObjects: saveData.gameObjects ? saveData.gameObjects.length : 0
            }
        };
        
        console.log(JSON.stringify(extracted, null, 2));
    } catch (error) {
        console.error('Parse error:', error.message);
        process.exit(1);
    }
} else if (command === 'validate') {
    try {
        const fileData = fs.readFileSync(filePath);
        // Just try to parse without extracting data
        parseSaveGame(fileData.buffer);
        console.log('{"valid": true}');
    } catch (error) {
        console.log('{"valid": false, "error": "' + error.message.replace(/"/g, '\\"') + '"}');
    }
} else {
    console.error('Unknown command:', command);
    console.error('Usage: node script.js [parse|validate] <file_path>');
    process.exit(1);
}
        