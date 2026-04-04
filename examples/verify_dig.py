#!/usr/bin/env python3
"""
Send a dig at a coordinate near the base that's likely solid.
Usage: python3 examples/verify_dig.py [host] [port] [x] [y]
"""
import sys
import asyncio
import json

HOST = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.10"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
DIG_X = int(sys.argv[3]) if len(sys.argv) > 3 else 122
DIG_Y = int(sys.argv[4]) if len(sys.argv) > 4 else 210  # above the base (dups at y~203)


async def main():
    reader, writer = await asyncio.open_connection(HOST, PORT)
    print(f"Connected. Sending dig at ({DIG_X}, {DIG_Y})...")

    writer.write(json.dumps({
        "type": "action", "action": "dig",
        "cell_x": DIG_X, "cell_y": DIG_Y
    }).encode() + b"\n")
    await writer.drain()

    # Read until we get a non-state message
    deadline = asyncio.get_event_loop().time() + 6.0
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
        except asyncio.TimeoutError:
            print("Timeout — cell is probably not solid or not valid.")
            print("Try a y value higher up (more rock). Duplicants are at y~196-205.")
            break
        msg = json.loads(line.strip())
        if msg.get("type") != "state":
            print(f"ACK: {msg}")
            if msg.get("success"):
                print("SUCCESS — dig order placed. Watch the game for a dig marker.")
            else:
                print(f"FAILED: {msg.get('error')}")
                print(f"Try: python3 examples/verify_dig.py {HOST} {PORT} {DIG_X} {DIG_Y + 5}")
            break

    writer.close()
    await writer.wait_closed()


asyncio.run(main())
