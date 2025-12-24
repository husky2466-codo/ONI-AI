"""
Demo script for Mini-ONI Environment.

Shows basic usage of the Mini-ONI environment including initialization,
reset, stepping, and rendering.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environments.mini_oni.environment import MiniONIEnvironment
from src.environments.mini_oni.actions import ActionType


def main():
    """Run Mini-ONI environment demo."""
    print("=== Mini-ONI Environment Demo ===\n")
    
    # Create environment
    print("Creating Mini-ONI environment...")
    env = MiniONIEnvironment(
        map_width=32,
        map_height=32,
        max_cycles=20,  # Short demo
        num_duplicants=3,
        starter_base_size=(12, 8),
        random_seed=42
    )
    
    print(f"Environment created with {env.num_actions} possible actions")
    print(f"Map size: {env.map_width}x{env.map_height}")
    print(f"Max cycles: {env.max_cycles}")
    
    # Reset environment
    print("\nResetting environment...")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Show initial state
    print("\nInitial state:")
    env.render(mode='human')
    
    # Run a few steps
    print("\n=== Running Environment Steps ===")
    
    total_reward = 0.0
    step_count = 0
    
    for step in range(50):  # Run for 50 steps max
        # Get valid actions
        action_mask = env.get_action_mask()
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            print("No valid actions available!")
            break
        
        # Choose a random valid action
        action_idx = np.random.choice(valid_actions)
        action = env.action_space[action_idx]
        
        # Take step
        obs, reward, done, info = env.step(action_idx)
        
        total_reward += reward
        step_count += 1
        
        # Print step info
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action.action_type.value}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Success: {info['action_success']}")
        print(f"  Cycle: {info['cycle']}")
        print(f"  Living duplicants: {info['living_duplicants']}")
        print(f"  Happy duplicants: {info['happy_duplicants']}")
        print(f"  Breathable tiles: {info['breathable_tiles']}")
        print(f"  Success score: {info['success_score']:.3f}")
        
        # Show state every 10 steps
        if (step + 1) % 10 == 0:
            print(f"\n--- State at step {step + 1} ---")
            env.render(mode='human')
        
        if done:
            print(f"\nEpisode finished at step {step + 1}")
            break
    
    # Final results
    print(f"\n=== Episode Summary ===")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward / step_count:.3f}")
    
    final_info = env.get_info()
    print(f"Final cycle: {final_info['current_cycle']}/{final_info['max_cycles']}")
    
    # Show final state
    print("\nFinal state:")
    env.render(mode='human')
    
    # Demonstrate action space analysis
    print(f"\n=== Action Space Analysis ===")
    action_types = {}
    for action in env.action_space:
        action_type = action.action_type.value
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    print("Action type distribution:")
    for action_type, count in action_types.items():
        print(f"  {action_type}: {count} actions")
    
    # Test RGB rendering
    print(f"\n=== RGB Rendering Test ===")
    rgb_array = env.render(mode='rgb_array')
    print(f"RGB array shape: {rgb_array.shape}")
    print(f"RGB array dtype: {rgb_array.dtype}")
    print(f"RGB value range: [{rgb_array.min()}, {rgb_array.max()}]")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()