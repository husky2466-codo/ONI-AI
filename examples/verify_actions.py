#!/usr/bin/env python3
"""
Verify that actions reach the game and produce ACKs.

Usage: python3 examples/verify_actions.py [host] [port]
"""
import sys
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)

HOST = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.10"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999


async def main():
    print(f"Connecting to {HOST}:{PORT}...")
    reader, writer = await asyncio.open_connection(HOST, PORT)
    print("Connected.")

    # --- Read first state message to see real coordinates ---
    print("\nWaiting for state tick...")
    state_data = None
    while True:
        line = await reader.readline()
        if not line:
            print("Connection closed.")
            return
        try:
            msg = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "state":
            state_data = msg["data"]
            break

    print(f"  cycle={state_data.get('cycle')}  time={state_data.get('time'):.1f}")
    dups = state_data.get("duplicants", [])
    buildings = state_data.get("buildings", [])
    print(f"  duplicants: {[(d['name'], d['x'], d['y']) for d in dups]}")
    print(f"  buildings:  {[(b['type'], b['x'], b['y']) for b in buildings[:3]]}")

    # --- Send a no_op first, confirm ACK ---
    print("\nSending no_op...")
    writer.write(json.dumps({"type": "action", "action": "no_op"}).encode() + b"\n")
    await writer.drain()

    ack = await read_ack(reader)
    print(f"  ACK: {ack}")

    # --- Send a dig at a likely solid cell ---
    # Pick a cell well away from duplicants. Use x=5, y=5 as a guess —
    # if it fails with "not solid" that still confirms the action reached the game.
    dig_x, dig_y = 5, 5
    print(f"\nSending dig at ({dig_x}, {dig_y})...")
    writer.write(json.dumps({
        "type": "action", "action": "dig",
        "cell_x": dig_x, "cell_y": dig_y
    }).encode() + b"\n")
    await writer.drain()

    ack = await read_ack(reader)
    print(f"  ACK: {ack}")

    if ack.get("success"):
        print("  -> Dig order placed in game. Watch for duplicants digging!")
    else:
        print(f"  -> Game rejected dig: {ack.get('error')}")
        print("     (Try different coordinates — need a solid tile)")

    writer.close()
    await writer.wait_closed()
    print("\nDone.")


async def read_ack(reader: asyncio.StreamReader, timeout: float = 5.0) -> dict:
    """Read lines until we get a non-state message (the ACK)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
        except asyncio.TimeoutError:
            return {"error": "timeout waiting for ACK"}
        if not line:
            return {"error": "connection closed"}
        try:
            msg = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if msg.get("type") != "state":
            return msg
    return {"error": "timeout"}


asyncio.run(main())
