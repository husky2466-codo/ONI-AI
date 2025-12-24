"""
Action definitions for Mini-ONI environment.

Defines the high-level action space for reinforcement learning agents.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Union
from .building_types import BuildingType


class ActionType(Enum):
    """High-level action types for Mini-ONI environment."""
    
    PLACE_BUILDING = "place_building"
    DIG = "dig"
    SET_PRIORITY = "set_priority"
    ASSIGN_DUPLICANT = "assign_duplicant"
    NO_OP = "no_op"


class MaterialType(Enum):
    """Material types for digging actions."""
    
    SANDSTONE = "sandstone"
    DIRT = "dirt"
    CLAY = "clay"
    GRANITE = "granite"
    COPPER_ORE = "copper_ore"
    COAL = "coal"


class TaskType(Enum):
    """Task types for priority actions."""
    
    BUILD = "build"
    DIG = "dig"
    SUPPLY = "supply"
    OPERATE = "operate"
    RESEARCH = "research"


class SkillType(Enum):
    """Skill types for duplicant assignment."""
    
    MINING = "mining"
    BUILDING = "building"
    RESEARCHING = "researching"
    COOKING = "cooking"
    OPERATING = "operating"


@dataclass
class Region:
    """Rectangular region specification for spatial actions."""
    
    x1: int
    y1: int
    x2: int
    y2: int
    
    def __post_init__(self):
        """Ensure region coordinates are valid."""
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
    
    @property
    def width(self) -> int:
        """Get region width."""
        return self.x2 - self.x1 + 1
    
    @property
    def height(self) -> int:
        """Get region height."""
        return self.y2 - self.y1 + 1
    
    @property
    def area(self) -> int:
        """Get region area."""
        return self.width * self.height
    
    def contains(self, x: int, y: int) -> bool:
        """Check if point is within region."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Action:
    """Base action class for Mini-ONI environment."""
    
    action_type: ActionType
    
    def is_valid(self, game_state) -> bool:
        """Check if action is valid in current game state."""
        return True


@dataclass
class PlaceBuildingAction(Action):
    """Action to place a building in a specified region."""
    
    building_type: BuildingType
    region: Region
    
    def __init__(self, building_type: BuildingType, region: Region):
        """Initialize building placement action."""
        super().__init__(ActionType.PLACE_BUILDING)
        self.building_type = building_type
        self.region = region
    
    def is_valid(self, game_state) -> bool:
        """Check if building placement is valid."""
        from .building_types import ESSENTIAL_BUILDINGS
        
        # Check if building type is allowed
        if self.building_type not in ESSENTIAL_BUILDINGS:
            return False
        
        building_props = ESSENTIAL_BUILDINGS[self.building_type]
        
        # Check if region size matches building requirements
        if (self.region.width != building_props.width or 
            self.region.height != building_props.height):
            return False
        
        # Check if region is within map bounds
        if (self.region.x1 < 0 or self.region.y1 < 0 or
            self.region.x2 >= game_state.width or 
            self.region.y2 >= game_state.height):
            return False
        
        # Check if tiles are suitable for building
        for x in range(self.region.x1, self.region.x2 + 1):
            for y in range(self.region.y1, self.region.y2 + 1):
                tile = game_state.get_tile(x, y)
                if tile.material_state not in building_props.buildable_on:
                    return False
                if tile.building is not None:
                    return False
        
        return True


@dataclass
class DigAction(Action):
    """Action to dig/excavate a region."""
    
    region: Region
    material_type: Optional[MaterialType] = None
    
    def __init__(self, region: Region, material_type: Optional[MaterialType] = None):
        """Initialize dig action."""
        super().__init__(ActionType.DIG)
        self.region = region
        self.material_type = material_type
    
    def is_valid(self, game_state) -> bool:
        """Check if digging is valid."""
        # Check if region is within map bounds
        if (self.region.x1 < 0 or self.region.y1 < 0 or
            self.region.x2 >= game_state.width or 
            self.region.y2 >= game_state.height):
            return False
        
        # Check if tiles can be dug
        for x in range(self.region.x1, self.region.x2 + 1):
            for y in range(self.region.y1, self.region.y2 + 1):
                tile = game_state.get_tile(x, y)
                if tile.material_state != "solid":
                    return False
                if tile.building is not None:
                    return False
        
        return True


@dataclass
class PriorityAction(Action):
    """Action to set task priority levels."""
    
    task_type: TaskType
    priority_level: int  # 1-9, where 9 is highest priority
    
    def __init__(self, task_type: TaskType, priority_level: int):
        """Initialize priority action."""
        super().__init__(ActionType.SET_PRIORITY)
        self.task_type = task_type
        self.priority_level = priority_level
    
    def is_valid(self, game_state) -> bool:
        """Check if priority setting is valid."""
        return 1 <= self.priority_level <= 9


@dataclass
class DuplicantAction(Action):
    """Action to assign duplicant skills or jobs."""
    
    duplicant_id: int
    skill_assignment: SkillType
    
    def __init__(self, duplicant_id: int, skill_assignment: SkillType):
        """Initialize duplicant action."""
        super().__init__(ActionType.ASSIGN_DUPLICANT)
        self.duplicant_id = duplicant_id
        self.skill_assignment = skill_assignment
    
    def is_valid(self, game_state) -> bool:
        """Check if duplicant assignment is valid."""
        return (0 <= self.duplicant_id < len(game_state.duplicants) and
                game_state.duplicants[self.duplicant_id].is_alive)


@dataclass
class NoOpAction(Action):
    """No-operation action (do nothing)."""
    
    def __init__(self):
        """Initialize no-op action."""
        super().__init__(ActionType.NO_OP)


# Action space configuration
MAX_ACTIONS = 200  # Limit total action space to ~200 discrete actions

def generate_action_space(map_width: int, map_height: int) -> list:
    """Generate the complete action space for the environment."""
    actions = []
    
    # Add no-op action
    actions.append(NoOpAction())
    
    # Add building placement actions for common sizes
    for building_type in BuildingType:
        from .building_types import ESSENTIAL_BUILDINGS
        props = ESSENTIAL_BUILDINGS[building_type]
        
        # Generate placement actions for key positions (not every tile)
        step_x = max(1, map_width // 8)  # 8x8 grid of potential positions
        step_y = max(1, map_height // 8)
        
        for x in range(0, map_width - props.width + 1, step_x):
            for y in range(0, map_height - props.height + 1, step_y):
                region = Region(x, y, x + props.width - 1, y + props.height - 1)
                actions.append(PlaceBuildingAction(building_type, region))
    
    # Add digging actions for common regions
    dig_sizes = [(1, 1), (2, 2), (3, 3), (4, 4)]  # Common dig sizes
    step = max(1, map_width // 6)  # 6x6 grid of potential dig positions
    
    for width, height in dig_sizes:
        for x in range(0, map_width - width + 1, step):
            for y in range(0, map_height - height + 1, step):
                region = Region(x, y, x + width - 1, y + height - 1)
                actions.append(DigAction(region))
    
    # Add priority actions
    for task_type in TaskType:
        for priority in [1, 5, 9]:  # Low, medium, high priority
            actions.append(PriorityAction(task_type, priority))
    
    # Add duplicant skill assignments (assuming max 3 duplicants)
    for duplicant_id in range(3):
        for skill in SkillType:
            actions.append(DuplicantAction(duplicant_id, skill))
    
    # Limit to MAX_ACTIONS to keep action space manageable
    if len(actions) > MAX_ACTIONS:
        # Prioritize essential actions
        essential_actions = [a for a in actions if isinstance(a, (NoOpAction, PlaceBuildingAction))]
        other_actions = [a for a in actions if not isinstance(a, (NoOpAction, PlaceBuildingAction))]
        
        # Keep all essential actions and sample from others
        remaining_slots = MAX_ACTIONS - len(essential_actions)
        if remaining_slots > 0:
            actions = essential_actions + other_actions[:remaining_slots]
        else:
            actions = essential_actions[:MAX_ACTIONS]
    
    return actions