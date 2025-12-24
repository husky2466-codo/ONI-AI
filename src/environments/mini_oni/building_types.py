"""
Building types and definitions for Mini-ONI environment.

Defines the essential subset of buildings available in the simplified environment.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Set


class BuildingType(Enum):
    """Essential building types for Mini-ONI environment (10-15 types)."""
    
    # Life Support
    OXYGEN_DIFFUSER = "oxygen_diffuser"
    ELECTROLYZER = "electrolyzer"
    
    # Infrastructure
    LADDER = "ladder"
    TILE = "tile"
    INSULATED_TILE = "insulated_tile"
    
    # Utilities
    OUTHOUSE = "outhouse"
    WASH_BASIN = "wash_basin"
    COT = "cot"
    
    # Food Production
    MEALWOOD = "mealwood"
    MICROBE_MUSHER = "microbe_musher"
    
    # Power
    MANUAL_GENERATOR = "manual_generator"
    BATTERY = "battery"
    
    # Water Management
    WATER_SIEVE = "water_sieve"
    LIQUID_PUMP = "liquid_pump"
    
    # Ventilation
    GAS_PUMP = "gas_pump"


@dataclass
class BuildingProperties:
    """Properties and constraints for each building type."""
    
    name: str
    width: int
    height: int
    power_consumption: float  # kW
    power_generation: float  # kW
    oxygen_production: float  # g/s
    requires_duplicant: bool
    buildable_on: Set[str]  # Material types it can be built on
    category: str


# Essential buildings configuration (limited to 15 types)
ESSENTIAL_BUILDINGS: Dict[BuildingType, BuildingProperties] = {
    BuildingType.OXYGEN_DIFFUSER: BuildingProperties(
        name="Oxygen Diffuser",
        width=1, height=1,
        power_consumption=120.0,
        power_generation=0.0,
        oxygen_production=550.0,
        requires_duplicant=False,
        buildable_on={"solid"},
        category="life_support"
    ),
    
    BuildingType.ELECTROLYZER: BuildingProperties(
        name="Electrolyzer", 
        width=2, height=2,
        power_consumption=120.0,
        power_generation=0.0,
        oxygen_production=888.0,
        requires_duplicant=False,
        buildable_on={"solid"},
        category="life_support"
    ),
    
    BuildingType.LADDER: BuildingProperties(
        name="Ladder",
        width=1, height=1,
        power_consumption=0.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=False,
        buildable_on={"solid", "gas"},
        category="infrastructure"
    ),
    
    BuildingType.TILE: BuildingProperties(
        name="Tile",
        width=1, height=1,
        power_consumption=0.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=False,
        buildable_on={"gas"},
        category="infrastructure"
    ),
    
    BuildingType.INSULATED_TILE: BuildingProperties(
        name="Insulated Tile",
        width=1, height=1,
        power_consumption=0.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=False,
        buildable_on={"gas"},
        category="infrastructure"
    ),
    
    BuildingType.OUTHOUSE: BuildingProperties(
        name="Outhouse",
        width=1, height=2,
        power_consumption=0.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=True,
        buildable_on={"solid"},
        category="utilities"
    ),
    
    BuildingType.WASH_BASIN: BuildingProperties(
        name="Wash Basin",
        width=1, height=1,
        power_consumption=0.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=True,
        buildable_on={"solid"},
        category="utilities"
    ),
    
    BuildingType.COT: BuildingProperties(
        name="Cot",
        width=2, height=1,
        power_consumption=0.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=True,
        buildable_on={"solid"},
        category="utilities"
    ),
    
    BuildingType.MEALWOOD: BuildingProperties(
        name="Mealwood",
        width=1, height=1,
        power_consumption=0.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=True,
        buildable_on={"solid"},
        category="food"
    ),
    
    BuildingType.MICROBE_MUSHER: BuildingProperties(
        name="Microbe Musher",
        width=2, height=1,
        power_consumption=75.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=True,
        buildable_on={"solid"},
        category="food"
    ),
    
    BuildingType.MANUAL_GENERATOR: BuildingProperties(
        name="Manual Generator",
        width=2, height=2,
        power_consumption=0.0,
        power_generation=400.0,
        oxygen_production=0.0,
        requires_duplicant=True,
        buildable_on={"solid"},
        category="power"
    ),
    
    BuildingType.BATTERY: BuildingProperties(
        name="Battery",
        width=2, height=1,
        power_consumption=0.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=False,
        buildable_on={"solid"},
        category="power"
    ),
    
    BuildingType.WATER_SIEVE: BuildingProperties(
        name="Water Sieve",
        width=3, height=2,
        power_consumption=120.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=False,
        buildable_on={"solid"},
        category="water"
    ),
    
    BuildingType.LIQUID_PUMP: BuildingProperties(
        name="Liquid Pump",
        width=1, height=1,
        power_consumption=240.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=False,
        buildable_on={"liquid"},
        category="water"
    ),
    
    BuildingType.GAS_PUMP: BuildingProperties(
        name="Gas Pump",
        width=1, height=1,
        power_consumption=240.0,
        power_generation=0.0,
        oxygen_production=0.0,
        requires_duplicant=False,
        buildable_on={"gas"},
        category="ventilation"
    )
}


def get_building_categories() -> Dict[str, Set[BuildingType]]:
    """Get buildings organized by category."""
    categories = {}
    for building_type, props in ESSENTIAL_BUILDINGS.items():
        if props.category not in categories:
            categories[props.category] = set()
        categories[props.category].add(building_type)
    return categories


def get_power_buildings() -> Set[BuildingType]:
    """Get buildings that generate power."""
    return {bt for bt, props in ESSENTIAL_BUILDINGS.items() if props.power_generation > 0}


def get_oxygen_buildings() -> Set[BuildingType]:
    """Get buildings that produce oxygen."""
    return {bt for bt, props in ESSENTIAL_BUILDINGS.items() if props.oxygen_production > 0}