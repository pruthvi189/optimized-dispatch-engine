"""Thread-safe bridge between the simulation runner thread and FastAPI's
asyncio event loop, plus a fan-out hub for WebSocket subscribers."""

import asyncio
import queue


class WebSocketHub:
    """Fans out snapshots to subscribed WebSocket clients.

    The runner thread pushes snapshots into `queue`; an asyncio task
    (`pump`) polls the queue with a bounded timeout and broadcasts. REST
    callers can read the latest snapshot without touching the queue.
    """

    def __init__(self):
        self.queue = queue.Queue(maxsize=200)
        self.clients = set()
        self.latest = None
        self._task = None
        self._loop = None

    def start(self, loop):
        self._loop = loop
        self._task = loop.create_task(self._pump())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def subscribe(self, ws):
        self.clients.add(ws)

    def unsubscribe(self, ws):
        self.clients.discard(ws)

    def publish(self, snapshot):
        """Called from the runner thread. Stores latest and queues a copy."""
        self.latest = snapshot
        try:
            self.queue.put_nowait(snapshot)
        except queue.Full:
            # Drop oldest so streaming stays fresh under load.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(snapshot)
            except queue.Full:
                pass

    async def _pump(self):
        while True:
            try:
                snapshot = await asyncio.to_thread(self.queue.get, timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0)
                continue
            if not self.clients:
                continue
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_json(snapshot)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.unsubscribe(ws)
