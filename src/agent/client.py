# src/agent/client.py
import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from src.agent.protocol import build_no_op, parse_state_message, StateMessage

logger = logging.getLogger(__name__)


class BridgeClient:
    """Async TCP client that connects to the ONIBridge mod."""

    def __init__(self, host: str, port: int = 9999, reconnect_delay: float = 5.0):
        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Connect (or reconnect) to the game bridge. Retries until successful."""
        while True:
            try:
                self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
                logger.info("Connected to ONIBridge at %s:%d", self.host, self.port)
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.warning("Connection failed: %s — retrying in %.1fs", e, self.reconnect_delay)
                await asyncio.sleep(self.reconnect_delay)

    async def state_stream(self) -> AsyncIterator[StateMessage]:
        """Yield StateMessage objects as they arrive from the game."""
        while True:
            try:
                assert self._reader is not None
                line = await self._reader.readline()
                if not line:
                    logger.warning("Connection closed by game — reconnecting")
                    await self.connect()
                    continue
                raw = line.decode("utf-8").strip()
                if not raw:
                    continue
                msg = parse_state_message(raw)
                if msg is not None:
                    yield msg
            except (ConnectionResetError, BrokenPipeError):
                logger.warning("Connection reset — reconnecting")
                await self.connect()

    async def send_action(self, action: dict[str, Any]) -> None:
        """Send an action command to the game."""
        if self._writer is None:
            logger.error("Not connected — cannot send action")
            return
        try:
            line = json.dumps(action) + "\n"
            self._writer.write(line.encode("utf-8"))
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError) as e:
            logger.warning("Send failed: %s", e)

    async def close(self) -> None:
        """Close the connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()


async def run_stub(host: str, port: int = 9999) -> None:
    """Stub runner — connect, print every state, send no_op each tick."""
    client = BridgeClient(host, port)
    await client.connect()

    async for state in client.state_stream():
        print(
            f"[Cycle {state.cycle}] "
            f"resources={state.data.get('resources')} "
            f"alerts={state.data.get('alerts')}"
        )
        await client.send_action(build_no_op())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    asyncio.run(run_stub(host))
