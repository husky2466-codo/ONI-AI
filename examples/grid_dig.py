#!/usr/bin/env python3
"""
Sends dig commands across a grid and prints every ACK.
Helps find which cells are solid.

Usage: python3 examples/grid_dig.py [host] [port]
"""
import sys
import asyncio
import json

HOST = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.10"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

# Grid to probe — centered on base (x 118-132, y 205-220)
X_RANGE = range(118, 133)   # 15 columns
Y_RANGE = range(205, 221)   # y 205 to 220 (above the floor at 202)


async def main():
    print(f"Connecting to {HOST}:{PORT}...")
    reader, writer = await asyncio.open_connection(HOST, PORT)
    print("Connected. Scanning for solid cells...\n")

    solid = []
    not_solid = []
    invalid = []
    errors = []

    for y in Y_RANGE:
        for x in X_RANGE:
            # Send dig
            writer.write(json.dumps({
                "type": "action", "action": "dig",
                "cell_x": x, "cell_y": y
            }).encode() + b"\n")
            await writer.drain()

            # Wait for ACK (skip state messages)
            ack = await read_next_ack(reader, timeout=3.0)

            if ack is None:
                errors.append((x, y, "timeout"))
                print(f"  ({x:3d},{y:3d}) TIMEOUT")
            elif ack.get("success"):
                solid.append((x, y))
                print(f"  ({x:3d},{y:3d}) SOLID - dig queued!")
            else:
                err = ack.get("error", "")
                if "not solid" in err:
                    not_solid.append((x, y))
                elif "Invalid cell" in err:
                    invalid.append((x, y))
                else:
                    errors.append((x, y, err))
                # Only print failures that aren't "not solid" (too noisy)
                if "not solid" not in err:
                    print(f"  ({x:3d},{y:3d}) {err}")

        print(f"  -- row y={y} done --")

    writer.close()

    print(f"\n=== Results ===")
    print(f"Solid cells (dig queued): {len(solid)}")
    for c in solid:
        print(f"  {c}")
    print(f"Not solid: {len(not_solid)}")
    print(f"Invalid: {len(invalid)}")
    print(f"Errors/timeouts: {len(errors)}")
    for e in errors:
        print(f"  {e}")


async def read_next_ack(reader, timeout=3.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
        except asyncio.TimeoutError:
            return None
        if not line:
            return None
        try:
            msg = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if msg.get("type") != "state":
            return msg
    return None


asyncio.run(main())
