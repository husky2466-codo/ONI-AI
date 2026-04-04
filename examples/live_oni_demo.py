#!/usr/bin/env python3
"""
Smoke test: connect to live ONI game and run 5 steps.

Usage: python3 examples/live_oni_demo.py [host] [port]
  host: IP of machine running ONI (default: 10.0.0.10)
  port: bridge port (default: 9999)
"""
import sys
import logging

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, ".")
from src.environments.live_oni import LiveONIEnvironment

host = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.10"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

print(f"Connecting to ONI at {host}:{port}...")
env = LiveONIEnvironment(host=host, port=port, step_timeout=10.0)

obs = env.reset()
print(f"reset() — obs shape: {obs.shape}, non-zero global features: {(obs[8192:] != 0).sum()}")

for step_num in range(5):
    obs, reward, done, info = env.step(0)  # action 0 = no_op
    print(
        f"step {step_num + 1}: cycle={info['cycle']} "
        f"reward={reward:.3f} done={done} "
        f"dups={len(info['duplicants'])} "
        f"alerts={info['alerts']}"
    )
    if done:
        print("Episode done.")
        break

env.close()
print("Done.")
