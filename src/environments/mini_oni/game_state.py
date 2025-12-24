"""
Game state representation for Mini-ONI environment.

Defines the core data structures for representing the game world.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from .building_types import BuildingType


@dataclass
class Tile:
    """Individual tile in the game grid."""
    
    x: int
    y: int
    material_state: str  # "solid", "liquid", "gas"
    element_id: int  # Element type identifier
    temperature: float  # Temperature in Celsius
    mass: float  # Mass of material in kg
    building: Optional['Building'] = None
    
    def is_breathable(self) -> bool:
        """Check if tile contains breathable oxygen."""
        # Oxygen element ID is typically 1, and needs >500g mass
        return (self.material_state == "gas" and 
                self.element_id == 1 and 
                self.mass > 0.5)  # 500g = 0.5kg
    
    def is_dangerous_temperature(self) -> bool:
        """Check if tile temperature is dangerous."""
        return self.temperature < 0 or self.temperature > 40


@dataclass
class Building:
    """Building instance in the game world."""
    
    building_type: BuildingType
    x: int
    y: int
    width: int
    height: int
    is_operational: bool = True
    power_consumption: float = 0.0
    assigned_duplicant: Optional[int] = None
    
    def get_footprint(self) -> List[tuple]:
        """Get list of (x, y) coordinates occupied by building."""
        footprint = []
        for dx in range(self.width):
            for dy in range(self.height):
                footprint.append((self.x + dx, self.y + dy))
        return footprint


@dataclass
class Duplicant:
    """Duplicant (colonist) in the game."""
    
    duplicant_id: int
    name: str
    x: float
    y: float
    is_alive: bool = True
    stress_level: float = 0.0  # 0-100%
    happiness: float = 50.0  # 0-100%
    assigned_skill: Optional[str] = None
    current_task: Optional[str] = None
    
    def is_happy(self) -> bool:
        """Check if duplicant meets happiness threshold."""
        return self.happiness >= 50.0
    
    def is_stressed(self) -> bool:
        """Check if duplicant is overly stressed."""
        return self.stress_level >= 80.0


@dataclass
class Resources:
    """Colony resource tracking."""
    
    oxygen: float = 0.0  # Total oxygen in kg
    food: float = 0.0  # Total food in kcal
    water: float = 0.0  # Clean water in kg
    polluted_water: float = 0.0  # Polluted water in kg
    power: float = 0.0  # Stored power in kJ
    power_generation: float = 0.0  # Power generation rate in kW
    power_consumption: float = 0.0  # Power consumption rate in kW


@dataclass
class GameState:
    """Complete game state for Mini-ONI environment."""
    
    width: int = 64
    height: int = 64
    cycle: int = 0
    max_cycles: int = 100
    grid: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=object))
    buildings: List[Building] = field(default_factory=list)
    duplicants: List[Duplicant] = field(default_factory=list)
    resources: Resources = field(default_factory=Resources)
    task_priorities: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize grid with empty tiles."""
        # Initialize empty grid with Tile objects
        self.grid = np.empty((self.height, self.width), dtype=object)
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y, x] = Tile(
                    x=x, y=y,
                    material_state="solid",
                    element_id=0,  # Default to sandstone
                    temperature=20.0,  # Room temperature
                    mass=1000.0  # 1000kg per tile
                )
    
    def get_tile(self, x: int, y: int) -> Optional[Tile]:
        """Get tile at coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x]
        return None
    
    def set_tile(self, x: int, y: int, tile: Tile) -> bool:
        """Set tile at coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = tile
            return True
        return False
    
    def place_building(self, building: Building) -> bool:
        """Place a building on the grid."""
        # Check if placement is valid
        footprint = building.get_footprint()
        for bx, by in footprint:
            tile = self.get_tile(bx, by)
            if tile is None or tile.building is not None:
                return False
        
        # Place building
        for bx, by in footprint:
            tile = self.get_tile(bx, by)
            tile.building = building
        
        self.buildings.append(building)
        return True
    
    def remove_building(self, building: Building) -> bool:
        """Remove a building from the grid."""
        if building not in self.buildings:
            return False
        
        # Clear building from tiles
        footprint = building.get_footprint()
        for bx, by in footprint:
            tile = self.get_tile(bx, by)
            if tile and tile.building == building:
                tile.building = None
        
        self.buildings.remove(building)
        return True
    
    def dig_tile(self, x: int, y: int) -> bool:
        """Dig out a solid tile."""
        tile = self.get_tile(x, y)
        if tile and tile.material_state == "solid" and tile.building is None:
            tile.material_state = "gas"
            tile.element_id = 1  # Oxygen
            tile.mass = 1.0  # Light gas
            return True
        return False
    
    def get_breathable_tiles(self) -> int:
        """Count tiles with breathable oxygen."""
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y, x]
                if tile.is_breathable():
                    count += 1
        return count
    
    def get_dangerous_temperature_tiles(self) -> int:
        """Count tiles with dangerous temperatures."""
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y, x]
                if tile.is_dangerous_temperature():
                    count += 1
        return count
    
    def get_happy_duplicants(self) -> int:
        """Count happy duplicants."""
        return sum(1 for dup in self.duplicants if dup.is_happy())
    
    def get_living_duplicants(self) -> int:
        """Count living duplicants."""
        return sum(1 for dup in self.duplicants if dup.is_alive)
    
    def update_resources(self):
        """Update resource calculations based on current state."""
        # Calculate oxygen from breathable tiles
        self.resources.oxygen = sum(
            tile.mass for y in range(self.height) for x in range(self.width)
            if (tile := self.grid[y, x]).is_breathable()
        )
        
        # Calculate power generation and consumption
        self.resources.power_generation = 0.0
        self.resources.power_consumption = 0.0
        
        from .building_types import ESSENTIAL_BUILDINGS
        for building in self.buildings:
            if building.is_operational:
                props = ESSENTIAL_BUILDINGS[building.building_type]
                self.resources.power_generation += props.power_generation
                self.resources.power_consumption += props.power_consumption
    
    def is_episode_done(self) -> bool:
        """Check if episode should terminate."""
        # Time limit reached
        if self.cycle >= self.max_cycles:
            return True
        
        # All duplicants dead
        if self.get_living_duplicants() == 0:
            return True
        
        # Critical oxygen shortage (less than 10% of tiles breathable)
        breathable_ratio = self.get_breathable_tiles() / (self.width * self.height)
        if breathable_ratio < 0.1:
            return True
        
        return False
    
    def get_success_score(self) -> float:
        """Calculate success score for the current state."""
        if not self.duplicants:
            return 0.0
        
        # Survival bonus
        survival_score = self.get_living_duplicants() / len(self.duplicants)
        
        # Oxygen score
        breathable_ratio = self.get_breathable_tiles() / (self.width * self.height)
        oxygen_score = min(1.0, breathable_ratio * 2)  # Target 50% breathable
        
        # Happiness score
        happiness_score = self.get_happy_duplicants() / len(self.duplicants)
        
        # Time bonus (surviving longer is better)
        time_score = min(1.0, self.cycle / self.max_cycles)
        
        # Weighted combination
        total_score = (
            survival_score * 0.4 +
            oxygen_score * 0.3 +
            happiness_score * 0.2 +
            time_score * 0.1
        )
        
        return total_score
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        # Create new game state
        new_state = GameState(
            width=self.width,
            height=self.height,
            cycle=self.cycle,
            max_cycles=self.max_cycles
        )
        
        # Copy grid
        new_state.grid = np.empty((self.height, self.width), dtype=object)
        for y in range(self.height):
            for x in range(self.width):
                old_tile = self.grid[y, x]
                new_tile = Tile(
                    x=old_tile.x,
                    y=old_tile.y,
                    material_state=old_tile.material_state,
                    element_id=old_tile.element_id,
                    temperature=old_tile.temperature,
                    mass=old_tile.mass
                )
                new_state.grid[y, x] = new_tile
        
        # Copy buildings
        building_map = {}
        for building in self.buildings:
            new_building = Building(
                building_type=building.building_type,
                x=building.x,
                y=building.y,
                width=building.width,
                height=building.height,
                is_operational=building.is_operational,
                power_consumption=building.power_consumption,
                assigned_duplicant=building.assigned_duplicant
            )
            new_state.buildings.append(new_building)
            building_map[building] = new_building
        
        # Update tile building references
        for y in range(self.height):
            for x in range(self.width):
                old_tile = self.grid[y, x]
                new_tile = new_state.grid[y, x]
                if old_tile.building:
                    new_tile.building = building_map[old_tile.building]
        
        # Copy duplicants
        for dup in self.duplicants:
            new_dup = Duplicant(
                duplicant_id=dup.duplicant_id,
                name=dup.name,
                x=dup.x,
                y=dup.y,
                is_alive=dup.is_alive,
                stress_level=dup.stress_level,
                happiness=dup.happiness,
                assigned_skill=dup.assigned_skill,
                current_task=dup.current_task
            )
            new_state.duplicants.append(new_dup)
        
        # Copy resources
        new_state.resources = Resources(
            oxygen=self.resources.oxygen,
            food=self.resources.food,
            water=self.resources.water,
            polluted_water=self.resources.polluted_water,
            power=self.resources.power,
            power_generation=self.resources.power_generation,
            power_consumption=self.resources.power_consumption
        )
        
        # Copy task priorities
        new_state.task_priorities = self.task_priorities.copy()
        
        return new_state