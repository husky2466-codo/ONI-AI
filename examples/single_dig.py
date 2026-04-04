#!/usr/bin/env python3
"""Send one dig, wait up to 10 seconds for any response."""
import sys, asyncio, json

HOST = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.10"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
X = int(sys.argv[3]) if len(sys.argv) > 3 else 124
Y = int(sys.argv[4]) if len(sys.argv) > 4 else 210

async def main():
    reader, writer = await asyncio.open_connection(HOST, PORT)
    print(f"Connected. Sending dig at ({X}, {Y})...")

    writer.write(json.dumps({"type": "action", "action": "dig", "cell_x": X, "cell_y": Y}).encode() + b"\n")
    await writer.drain()
    print("Sent. Waiting up to 10s for ANY message (state or ack)...")

    for i in range(20):
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=0.5)
            if not line:
                print("Connection closed.")
                break
            msg = json.loads(line.strip())
            print(f"  [{i:02d}] type={msg.get('type')} | {json.dumps(msg)[:120]}")
            if msg.get("type") != "state":
                print("  ^^^ That's our ACK!")
                break
        except asyncio.TimeoutError:
            print(f"  [{i:02d}] (no data)")

    writer.close()

asyncio.run(main())
