"""
Demo script for Mini-ONI Objective System.

Demonstrates the three main objectives and their evaluation:
1. Primary: Oxygen maintenance (>500g/tile)
2. Secondary: Polluted water routing
3. Tertiary: Duplicant happiness (>50%)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environments.mini_oni.environment import MiniONIEnvironment
from src.environments.mini_oni.objectives import ObjectiveRewards
from src.environments.mini_oni.actions import PlaceBuildingAction, DigAction, Region
from src.environments.mini_oni.building_types import BuildingType


def demo_basic_objectives():
    """Demonstrate basic objective evaluation."""
    print("=== Mini-ONI Objective System Demo ===\n")
    
    # Create environment with custom objective rewards
    custom_rewards = ObjectiveRewards(
        oxygen_tile_reward=0.15,  # Slightly higher oxygen reward
        happiness_reward=0.08,    # Higher happiness reward
        water_routing_reward=0.1  # Higher water reward
    )
    
    env = MiniONIEnvironment(
        map_width=32, 
        map_height=32, 
        max_cycles=50,
        objective_rewards=custom_rewards
    )
    
    print("Environment initialized with custom objective rewards")
    print(f"Map size: {env.map_width}x{env.map_height}")
    print(f"Max cycles: {env.max_cycles}")
    print(f"Action space size: {env.num_actions}")
    
    # Reset and show initial state
    obs = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    
    # Show initial objective status
    print("\n=== Initial Objective Status ===")
    env.render(mode='human')
    
    return env


def demo_oxygen_objective(env):
    """Demonstrate oxygen objective progression."""
    print("\n=== Oxygen Objective Demo ===")
    
    # Take some steps and monitor oxygen objective
    for step in range(10):
        # Try to place oxygen-producing buildings
        action_taken = False
        
        # Look for oxygen diffuser placement actions
        for i, action in enumerate(env.action_space):
            if isinstance(action, PlaceBuildingAction):
                if (action.building_type == BuildingType.OXYGEN_DIFFUSER and 
                    action.is_valid(env.game_state)):
                    obs, reward, done, info = env.step(i)
                    action_taken = True
                    print(f"Step {step+1}: Placed oxygen diffuser, reward: {reward:.2f}")
                    break
        
        if not action_taken:
            # Take no-op action
            obs, reward, done, info = env.step(0)
            print(f"Step {step+1}: No-op action, reward: {reward:.2f}")
        
        # Show oxygen objective progress
        objectives = info['objectives']
        print(f"  Oxygen ratio: {objectives['oxygen_ratio']:.3f}")
        print(f"  Breathable tiles: {objectives['breathable_tiles']}")
        print(f"  Oxygen reward: {objectives['total_oxygen_reward']:.2f}")
        print(f"  Objective status: {objectives['oxygen_objective_status']}")
        
        if done:
            break
    
    print(f"\nPrimary objective met: {env.is_primary_objective_met()}")


def demo_water_objective(env):
    """Demonstrate water objective progression."""
    print("\n=== Water Objective Demo ===")
    
    # Try to build water management system
    water_buildings_placed = 0
    
    for step in range(15):
        action_taken = False
        
        # Look for water-related building placement
        for i, action in enumerate(env.action_space):
            if isinstance(action, PlaceBuildingAction):
                if (action.building_type in [BuildingType.WATER_SIEVE, BuildingType.LIQUID_PUMP] and 
                    action.is_valid(env.game_state)):
                    obs, reward, done, info = env.step(i)
                    action_taken = True
                    water_buildings_placed += 1
                    print(f"Step {step+1}: Placed {action.building_type.value}, reward: {reward:.2f}")
                    break
        
        if not action_taken:
            obs, reward, done, info = env.step(0)
        
        # Show water objective progress every 5 steps
        if step % 5 == 4:
            objectives = info['objectives']
            print(f"  Water buildings: {objectives['water_buildings_count']}")
            print(f"  Has water sieve: {objectives['has_water_sieve']}")
            print(f"  Has liquid pump: {objectives['has_liquid_pump']}")
            print(f"  System functional: {objectives['water_system_functional']}")
            print(f"  Water reward: {objectives['total_water_reward']:.2f}")
        
        if done or water_buildings_placed >= 2:
            break
    
    print(f"\nSecondary objective met: {env.is_secondary_objective_met()}")


def demo_happiness_objective(env):
    """Demonstrate happiness objective progression."""
    print("\n=== Happiness Objective Demo ===")
    
    # Monitor duplicant happiness over time
    for step in range(20):
        # Try to place comfort buildings
        action_taken = False
        
        for i, action in enumerate(env.action_space):
            if isinstance(action, PlaceBuildingAction):
                if (action.building_type in [BuildingType.COT, BuildingType.WASH_BASIN] and 
                    action.is_valid(env.game_state)):
                    obs, reward, done, info = env.step(i)
                    action_taken = True
                    print(f"Step {step+1}: Placed {action.building_type.value}, reward: {reward:.2f}")
                    break
        
        if not action_taken:
            obs, reward, done, info = env.step(0)
        
        # Show happiness progress every 5 steps
        if step % 5 == 4:
            objectives = info['objectives']
            print(f"  Happy duplicants: {objectives['happy_duplicants']}/{objectives['living_duplicants']}")
            print(f"  Happiness ratio: {objectives['happiness_ratio']:.3f}")
            print(f"  Stressed duplicants: {objectives['stressed_duplicants']}")
            print(f"  Happiness reward: {objectives['total_happiness_reward']:.2f}")
        
        if done:
            break
    
    print(f"\nTertiary objective met: {env.is_tertiary_objective_met()}")


def demo_overall_progress(env):
    """Demonstrate overall objective progress tracking."""
    print("\n=== Overall Objective Progress ===")
    
    # Get final objective summary
    summary = env.get_objective_status()
    
    print(f"Overall objective score: {summary['overall_score']:.3f}")
    print(f"Objectives completed: {summary['episode_stats']['objectives_completed']}/3")
    print(f"All objectives met: {env.are_all_objectives_met()}")
    
    print("\nDetailed objective status:")
    for obj_name, obj_data in summary['objectives'].items():
        print(f"  {obj_name.replace('_', ' ').title()}:")
        print(f"    Status: {obj_data['status']}")
        print(f"    Progress: {obj_data['completion_percentage']:.1f}%")
        print(f"    Current: {obj_data['current_value']:.3f} / {obj_data['target_value']:.3f}")
        print(f"    Cycles maintained: {obj_data['cycles_maintained']}/{obj_data['cycles_required']}")
    
    print("\nEpisode statistics:")
    stats = summary['episode_stats']
    print(f"  Peak oxygen ratio: {stats['peak_oxygen_ratio']:.3f}")
    print(f"  Peak happiness ratio: {stats['peak_happiness_ratio']:.3f}")
    print(f"  Total oxygen reward: {stats['total_oxygen_reward']:.2f}")
    print(f"  Total water reward: {stats['total_water_reward']:.2f}")
    print(f"  Total happiness reward: {stats['total_happiness_reward']:.2f}")


def demo_objective_rewards():
    """Demonstrate different objective reward configurations."""
    print("\n=== Objective Reward Configuration Demo ===")
    
    # Test different reward configurations
    configs = [
        ("Balanced", ObjectiveRewards()),
        ("Oxygen Focus", ObjectiveRewards(oxygen_tile_reward=0.2, oxygen_threshold_bonus=100.0)),
        ("Happiness Focus", ObjectiveRewards(happiness_reward=0.15, happiness_threshold_bonus=50.0)),
        ("Water Focus", ObjectiveRewards(water_routing_reward=0.2, water_system_bonus=80.0))
    ]
    
    for config_name, rewards in configs:
        print(f"\n{config_name} Configuration:")
        env = MiniONIEnvironment(map_width=16, map_height=16, objective_rewards=rewards)
        env.reset()
        
        # Take a few steps
        total_reward = 0
        for _ in range(5):
            obs, reward, done, info = env.step(0)
            total_reward += reward
        
        objectives = info['objectives']
        print(f"  Total reward after 5 steps: {total_reward:.2f}")
        print(f"  Oxygen reward: {objectives['total_oxygen_reward']:.2f}")
        print(f"  Water reward: {objectives['total_water_reward']:.2f}")
        print(f"  Happiness reward: {objectives['total_happiness_reward']:.2f}")


def main():
    """Run the complete objective system demo."""
    try:
        # Basic demo
        env = demo_basic_objectives()
        
        # Demonstrate each objective
        demo_oxygen_objective(env)
        demo_water_objective(env)
        demo_happiness_objective(env)
        
        # Show overall progress
        demo_overall_progress(env)
        
        # Show different reward configurations
        demo_objective_rewards()
        
        print("\n=== Demo Complete ===")
        print("The objective system successfully tracks and evaluates:")
        print("1. ✓ Primary objective: Oxygen maintenance")
        print("2. ✓ Secondary objective: Water routing system")
        print("3. ✓ Tertiary objective: Duplicant happiness")
        print("4. ✓ Overall progress scoring and tracking")
        print("5. ✓ Configurable reward systems")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())