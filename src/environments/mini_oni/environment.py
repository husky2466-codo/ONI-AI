"""
Mini-ONI Environment implementation.

A simplified version of Oxygen Not Included for reinforcement learning training.
Implements the core environment with 64x64 tile limit, essential buildings only,
and 100-cycle episode limit.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import random

from .game_state import GameState, Tile, Building, Duplicant, Resources
from .building_types import BuildingType, ESSENTIAL_BUILDINGS
from .actions import (
    Action, ActionType, PlaceBuildingAction, DigAction, 
    PriorityAction, DuplicantAction, NoOpAction, generate_action_space
)
from .objectives import ObjectiveSystem, ObjectiveRewards


class MiniONIEnvironment:
    """
    Mini-ONI Environment for reinforcement learning.
    
    Features:
    - 64x64 tile maximum map size
    - Rectangular starter base constraints
    - Essential building types only (10-15 types)
    - 100 cycle maximum episode length
    - Clear success/failure criteria
    """
    
    def __init__(
        self,
        map_width: int = 64,
        map_height: int = 64,
        max_cycles: int = 100,
        num_duplicants: int = 3,
        starter_base_size: Tuple[int, int] = (16, 12),
        random_seed: Optional[int] = None,
        objective_rewards: Optional[ObjectiveRewards] = None
    ):
        """
        Initialize Mini-ONI environment.
        
        Args:
            map_width: Width of the game map (max 64)
            map_height: Height of the game map (max 64)
            max_cycles: Maximum cycles per episode (max 100)
            num_duplicants: Number of duplicants to start with
            starter_base_size: (width, height) of initial cleared area
            random_seed: Random seed for reproducibility
            objective_rewards: Custom objective reward configuration
        """
        # Enforce constraints
        self.map_width = min(map_width, 64)
        self.map_height = min(map_height, 64)
        self.max_cycles = min(max_cycles, 100)
        self.num_duplicants = num_duplicants
        self.starter_base_size = starter_base_size
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize objective system
        self.objective_system = ObjectiveSystem(objective_rewards)
        
        # Generate action space
        self.action_space = generate_action_space(self.map_width, self.map_height)
        self.num_actions = len(self.action_space)
        
        # Initialize state
        self.game_state: Optional[GameState] = None
        self.episode_step = 0
        
        # Episode statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'max_oxygen_tiles': 0,
            'max_happy_duplicants': 0,
            'buildings_placed': 0,
            'tiles_dug': 0
        }
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        # Create new game state
        self.game_state = GameState(
            width=self.map_width,
            height=self.map_height,
            max_cycles=self.max_cycles
        )
        
        # Initialize rectangular starter base
        self._initialize_starter_base()
        
        # Place initial duplicants
        self._initialize_duplicants()
        
        # Place essential starting buildings
        self._initialize_starting_buildings()
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_stats = {
            'total_reward': 0.0,
            'max_oxygen_tiles': 0,
            'max_happy_duplicants': 0,
            'buildings_placed': 0,
            'tiles_dug': 0
        }
        
        # Reset objective system
        self.objective_system.reset()
        
        return self._get_observation()
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action_idx: Index of action in action_space
            
        Returns:
            (observation, reward, done, info)
        """
        if self.game_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Get action
        if 0 <= action_idx < len(self.action_space):
            action = self.action_space[action_idx]
        else:
            action = NoOpAction()  # Invalid action becomes no-op
        
        # Execute action
        action_success = self._execute_action(action)
        
        # Update game simulation
        self._update_simulation()
        
        # Calculate reward
        reward = self._calculate_reward(action, action_success)
        
        # Check if episode is done
        done = self.game_state.is_episode_done()
        
        # Update episode stats
        self.episode_step += 1
        self.episode_stats['total_reward'] += reward
        self.episode_stats['max_oxygen_tiles'] = max(
            self.episode_stats['max_oxygen_tiles'],
            self.game_state.get_breathable_tiles()
        )
        self.episode_stats['max_happy_duplicants'] = max(
            self.episode_stats['max_happy_duplicants'],
            self.game_state.get_happy_duplicants()
        )
        
        # Prepare info
        objective_metrics = self.objective_system.evaluate_objectives(self.game_state)
        
        info = {
            'action_success': action_success,
            'cycle': self.game_state.cycle,
            'episode_step': self.episode_step,
            'breathable_tiles': self.game_state.get_breathable_tiles(),
            'happy_duplicants': self.game_state.get_happy_duplicants(),
            'living_duplicants': self.game_state.get_living_duplicants(),
            'success_score': self.game_state.get_success_score(),
            'episode_stats': self.episode_stats.copy(),
            'objectives': objective_metrics,
            'objective_summary': self.objective_system.get_objective_summary()
        }
        
        return self._get_observation(), reward, done, info
    
    def _initialize_starter_base(self):
        """Initialize rectangular starter base area."""
        # Calculate center position for starter base
        start_x = (self.map_width - self.starter_base_size[0]) // 2
        start_y = (self.map_height - self.starter_base_size[1]) // 2
        
        # Clear starter base area (make it gas/air)
        for x in range(start_x, start_x + self.starter_base_size[0]):
            for y in range(start_y, start_y + self.starter_base_size[1]):
                tile = self.game_state.get_tile(x, y)
                if tile:
                    tile.material_state = "gas"
                    tile.element_id = 1  # Oxygen
                    tile.mass = 1.0  # Light breathable air
                    tile.temperature = 20.0  # Comfortable temperature
        
        # Add some solid foundation at the bottom
        foundation_y = start_y + self.starter_base_size[1] - 1
        for x in range(start_x, start_x + self.starter_base_size[0]):
            tile = self.game_state.get_tile(x, foundation_y)
            if tile:
                tile.material_state = "solid"
                tile.element_id = 0  # Sandstone
                tile.mass = 1000.0
    
    def _initialize_duplicants(self):
        """Initialize starting duplicants."""
        # Place duplicants in center of starter base
        center_x = self.map_width // 2
        center_y = self.map_height // 2
        
        for i in range(self.num_duplicants):
            duplicant = Duplicant(
                duplicant_id=i,
                name=f"Duplicant_{i+1}",
                x=center_x + i - 1,  # Spread them out slightly
                y=center_y,
                is_alive=True,
                stress_level=10.0,  # Low initial stress
                happiness=60.0,  # Slightly happy initially
                assigned_skill=None,
                current_task=None
            )
            self.game_state.duplicants.append(duplicant)
    
    def _initialize_starting_buildings(self):
        """Place essential starting buildings."""
        center_x = self.map_width // 2
        center_y = self.map_height // 2
        
        # Place a cot for sleeping
        cot_building = Building(
            building_type=BuildingType.COT,
            x=center_x - 3,
            y=center_y - 2,
            width=2,
            height=1,
            is_operational=True
        )
        self.game_state.place_building(cot_building)
        
        # Place an outhouse
        outhouse_building = Building(
            building_type=BuildingType.OUTHOUSE,
            x=center_x + 2,
            y=center_y - 3,
            width=1,
            height=2,
            is_operational=True
        )
        self.game_state.place_building(outhouse_building)
        
        # Place a manual generator for power
        generator_building = Building(
            building_type=BuildingType.MANUAL_GENERATOR,
            x=center_x - 1,
            y=center_y + 2,
            width=2,
            height=2,
            is_operational=True
        )
        self.game_state.place_building(generator_building)
    
    def _execute_action(self, action: Action) -> bool:
        """Execute an action and return success status."""
        if not action.is_valid(self.game_state):
            return False
        
        if isinstance(action, PlaceBuildingAction):
            return self._execute_place_building(action)
        elif isinstance(action, DigAction):
            return self._execute_dig(action)
        elif isinstance(action, PriorityAction):
            return self._execute_priority(action)
        elif isinstance(action, DuplicantAction):
            return self._execute_duplicant_action(action)
        elif isinstance(action, NoOpAction):
            return True  # No-op always succeeds
        
        return False
    
    def _execute_place_building(self, action: PlaceBuildingAction) -> bool:
        """Execute building placement action."""
        props = ESSENTIAL_BUILDINGS[action.building_type]
        
        building = Building(
            building_type=action.building_type,
            x=action.region.x1,
            y=action.region.y1,
            width=props.width,
            height=props.height,
            is_operational=True,
            power_consumption=props.power_consumption
        )
        
        success = self.game_state.place_building(building)
        if success:
            self.episode_stats['buildings_placed'] += 1
        
        return success
    
    def _execute_dig(self, action: DigAction) -> bool:
        """Execute digging action."""
        tiles_dug = 0
        
        for x in range(action.region.x1, action.region.x2 + 1):
            for y in range(action.region.y1, action.region.y2 + 1):
                if self.game_state.dig_tile(x, y):
                    tiles_dug += 1
        
        if tiles_dug > 0:
            self.episode_stats['tiles_dug'] += tiles_dug
            return True
        
        return False
    
    def _execute_priority(self, action: PriorityAction) -> bool:
        """Execute priority setting action."""
        self.game_state.task_priorities[action.task_type.value] = action.priority_level
        return True
    
    def _execute_duplicant_action(self, action: DuplicantAction) -> bool:
        """Execute duplicant skill assignment action."""
        if 0 <= action.duplicant_id < len(self.game_state.duplicants):
            duplicant = self.game_state.duplicants[action.duplicant_id]
            if duplicant.is_alive:
                duplicant.assigned_skill = action.skill_assignment.value
                return True
        return False
    
    def _update_simulation(self):
        """Update game simulation (simplified)."""
        # Advance cycle every 10 steps (simplified time progression)
        if self.episode_step % 10 == 0:
            self.game_state.cycle += 1
        
        # Update resources
        self.game_state.update_resources()
        
        # Simple duplicant stress/happiness updates
        for duplicant in self.game_state.duplicants:
            if duplicant.is_alive:
                # Stress increases over time
                duplicant.stress_level += 0.1
                
                # Happiness affected by oxygen and temperature
                breathable_tiles = self.game_state.get_breathable_tiles()
                total_tiles = self.map_width * self.map_height
                oxygen_ratio = breathable_tiles / total_tiles
                
                if oxygen_ratio > 0.3:
                    duplicant.happiness += 0.2
                else:
                    duplicant.happiness -= 0.5
                
                # Clamp values
                duplicant.stress_level = max(0, min(100, duplicant.stress_level))
                duplicant.happiness = max(0, min(100, duplicant.happiness))
                
                # Death from extreme stress
                if duplicant.stress_level >= 100:
                    duplicant.is_alive = False
    
    def _calculate_reward(self, action: Action, action_success: bool) -> float:
        """Calculate reward for current step."""
        reward = 0.0
        
        # Get objective-based rewards
        objective_reward = self.objective_system.get_objective_rewards(self.game_state)
        reward += objective_reward
        
        # Action success bonus
        if action_success and not isinstance(action, NoOpAction):
            reward += 1.0  # Bonus for successful actions
        
        # Sparse rewards (episodic)
        if self.game_state.cycle >= self.max_cycles:
            reward += 100.0  # Survival bonus
        
        # Death penalty
        living_duplicants = self.game_state.get_living_duplicants()
        if living_duplicants < len(self.game_state.duplicants):
            dead_count = len(self.game_state.duplicants) - living_duplicants
            reward -= dead_count * 50.0  # -50 per dead duplicant
        
        # Infrastructure milestones
        if len(self.game_state.buildings) >= 5:
            reward += 20.0  # Bonus for building infrastructure
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Flattened observation array combining spatial and global features
        """
        # Create spatial observation (simplified 32x32 grid)
        spatial_obs = self._get_spatial_observation()
        
        # Create global features
        global_obs = self._get_global_observation()
        
        # Combine observations
        spatial_flat = spatial_obs.flatten()
        combined_obs = np.concatenate([spatial_flat, global_obs])
        
        return combined_obs.astype(np.float32)
    
    def _get_spatial_observation(self) -> np.ndarray:
        """Get downscaled spatial observation (32x32x8 channels)."""
        # Downscale from 64x64 to 32x32
        scale_x = self.map_width / 32
        scale_y = self.map_height / 32
        
        spatial_obs = np.zeros((32, 32, 8), dtype=np.float32)
        
        for obs_y in range(32):
            for obs_x in range(32):
                # Map to original coordinates
                orig_x = int(obs_x * scale_x)
                orig_y = int(obs_y * scale_y)
                
                tile = self.game_state.get_tile(orig_x, orig_y)
                if tile:
                    # Channel 0-2: Material state (solid/liquid/gas)
                    if tile.material_state == "solid":
                        spatial_obs[obs_y, obs_x, 0] = 1.0
                    elif tile.material_state == "liquid":
                        spatial_obs[obs_y, obs_x, 1] = 1.0
                    else:  # gas
                        spatial_obs[obs_y, obs_x, 2] = 1.0
                    
                    # Channel 3: Element ID (normalized)
                    spatial_obs[obs_y, obs_x, 3] = tile.element_id / 10.0
                    
                    # Channel 4: Temperature (normalized)
                    spatial_obs[obs_y, obs_x, 4] = (tile.temperature + 50) / 100.0  # -50 to 50 -> 0 to 1
                    
                    # Channel 5: Building type
                    if tile.building:
                        building_id = list(BuildingType).index(tile.building.building_type)
                        spatial_obs[obs_y, obs_x, 5] = building_id / len(BuildingType)
                    
                    # Channel 6: Duplicant presence
                    for dup in self.game_state.duplicants:
                        if (abs(dup.x - orig_x) < 1 and abs(dup.y - orig_y) < 1 and 
                            dup.is_alive):
                            spatial_obs[obs_y, obs_x, 6] = 1.0
                            break
                    
                    # Channel 7: Breathable oxygen
                    if tile.is_breathable():
                        spatial_obs[obs_y, obs_x, 7] = 1.0
        
        return spatial_obs
    
    def _get_global_observation(self) -> np.ndarray:
        """Get global features (64 dimensions)."""
        global_obs = np.zeros(64, dtype=np.float32)
        
        # Basic stats (0-9)
        global_obs[0] = self.game_state.cycle / self.max_cycles  # Normalized cycle
        global_obs[1] = len(self.game_state.buildings) / 20.0  # Building count
        global_obs[2] = self.game_state.get_living_duplicants() / self.num_duplicants
        global_obs[3] = self.game_state.get_happy_duplicants() / self.num_duplicants
        global_obs[4] = self.game_state.get_breathable_tiles() / (self.map_width * self.map_height)
        global_obs[5] = self.game_state.get_dangerous_temperature_tiles() / (self.map_width * self.map_height)
        
        # Resources (10-19)
        global_obs[10] = min(self.game_state.resources.oxygen, 1000) / 1000.0
        global_obs[11] = min(self.game_state.resources.food, 1000) / 1000.0
        global_obs[12] = min(self.game_state.resources.water, 1000) / 1000.0
        global_obs[13] = min(self.game_state.resources.power, 1000) / 1000.0
        global_obs[14] = self.game_state.resources.power_generation / 1000.0
        global_obs[15] = self.game_state.resources.power_consumption / 1000.0
        
        # Building counts by category (20-35)
        building_counts = {}
        for building in self.game_state.buildings:
            category = ESSENTIAL_BUILDINGS[building.building_type].category
            building_counts[category] = building_counts.get(category, 0) + 1
        
        categories = ["life_support", "infrastructure", "utilities", "food", "power", "water", "ventilation"]
        for i, category in enumerate(categories):
            if i < 16:  # Ensure we don't exceed array bounds
                global_obs[20 + i] = building_counts.get(category, 0) / 5.0  # Normalize by max expected
        
        # Duplicant stats (36-50)
        for i, dup in enumerate(self.game_state.duplicants):
            if i < 5:  # Max 5 duplicants tracked
                base_idx = 36 + i * 3
                global_obs[base_idx] = 1.0 if dup.is_alive else 0.0
                global_obs[base_idx + 1] = dup.happiness / 100.0
                global_obs[base_idx + 2] = dup.stress_level / 100.0
        
        # Task priorities (51-60)
        priority_tasks = ["build", "dig", "supply", "operate", "research"]
        for i, task in enumerate(priority_tasks):
            if i < 10:
                priority = self.game_state.task_priorities.get(task, 5)
                global_obs[51 + i] = priority / 9.0  # Normalize 1-9 to 0-1
        
        return global_obs
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' for display, 'rgb_array' for array)
            
        Returns:
            RGB array if mode='rgb_array', None otherwise
        """
        if mode == 'human':
            self._render_text()
            return None
        elif mode == 'rgb_array':
            return self._render_rgb()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_text(self):
        """Render text-based visualization."""
        print(f"\n=== Mini-ONI Environment (Cycle {self.game_state.cycle}/{self.max_cycles}) ===")
        print(f"Living Duplicants: {self.game_state.get_living_duplicants()}/{len(self.game_state.duplicants)}")
        print(f"Happy Duplicants: {self.game_state.get_happy_duplicants()}")
        print(f"Breathable Tiles: {self.game_state.get_breathable_tiles()}")
        print(f"Buildings: {len(self.game_state.buildings)}")
        print(f"Success Score: {self.game_state.get_success_score():.2f}")
        
        # Show objective progress
        print("\n" + self.objective_system.get_objective_progress_text())
        
        # Simple grid visualization (show center area)
        center_x, center_y = self.map_width // 2, self.map_height // 2
        print("\nGrid (center 16x8 area):")
        for y in range(center_y - 4, center_y + 4):
            row = ""
            for x in range(center_x - 8, center_x + 8):
                tile = self.game_state.get_tile(x, y)
                if tile:
                    if tile.building:
                        row += "B"  # Building
                    elif tile.material_state == "solid":
                        row += "#"  # Solid
                    elif tile.is_breathable():
                        row += "O"  # Oxygen
                    else:
                        row += "."  # Empty
                else:
                    row += " "
            print(row)
    
    def _render_rgb(self) -> np.ndarray:
        """Render RGB array visualization."""
        # Create RGB image (simplified)
        img = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        
        for y in range(self.map_height):
            for x in range(self.map_width):
                tile = self.game_state.get_tile(x, y)
                if tile:
                    if tile.building:
                        img[y, x] = [100, 100, 255]  # Blue for buildings
                    elif tile.material_state == "solid":
                        img[y, x] = [139, 69, 19]  # Brown for solid
                    elif tile.is_breathable():
                        img[y, x] = [135, 206, 235]  # Light blue for oxygen
                    else:
                        img[y, x] = [64, 64, 64]  # Dark gray for other gas
        
        # Mark duplicant positions
        for dup in self.game_state.duplicants:
            if dup.is_alive:
                x, y = int(dup.x), int(dup.y)
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    img[y, x] = [255, 255, 0]  # Yellow for duplicants
        
        return img
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions in current state."""
        mask = np.zeros(self.num_actions, dtype=bool)
        
        for i, action in enumerate(self.action_space):
            mask[i] = action.is_valid(self.game_state)
        
        return mask
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed environment information."""
        return {
            'map_size': (self.map_width, self.map_height),
            'max_cycles': self.max_cycles,
            'num_duplicants': self.num_duplicants,
            'action_space_size': self.num_actions,
            'current_cycle': self.game_state.cycle if self.game_state else 0,
            'episode_step': self.episode_step,
            'starter_base_size': self.starter_base_size
        }
    
    def get_objective_status(self) -> Dict[str, Any]:
        """Get current objective status and progress."""
        if self.game_state is None:
            return {}
        
        return self.objective_system.get_objective_summary()
    
    def is_primary_objective_met(self) -> bool:
        """Check if primary objective (oxygen maintenance) is met."""
        return self.objective_system.is_primary_objective_met()
    
    def is_secondary_objective_met(self) -> bool:
        """Check if secondary objective (water routing) is met."""
        return self.objective_system.is_secondary_objective_met()
    
    def is_tertiary_objective_met(self) -> bool:
        """Check if tertiary objective (duplicant happiness) is met."""
        return self.objective_system.is_tertiary_objective_met()
    
    def are_all_objectives_met(self) -> bool:
        """Check if all objectives are currently met."""
        return self.objective_system.are_all_objectives_met()