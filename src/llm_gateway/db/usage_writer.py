"""Batched, off-the-hot-path persistence of usage records."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway.db.models import UsageRecord
from llm_gateway.db.repository import add_usage_records

logger = logging.getLogger(__name__)


class UsageWriter:
    """Queues usage records and flushes them in batches every ``interval_s``."""

    def __init__(
        self, session_factory: async_sessionmaker, *, interval_s: float = 1.0, max_batch: int = 500
    ) -> None:
        self._session_factory = session_factory
        self._interval_s = interval_s
        self._max_batch = max_batch
        self._queue: asyncio.Queue[UsageRecord] = asyncio.Queue(maxsize=100_000)
        self._task: asyncio.Task[None] | None = None
        self._flush_lock = asyncio.Lock()

    def submit(self, record: UsageRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            logger.error(
                "usage queue full; dropping usage record", extra={"rid": record.request_id}
            )

    async def flush(self) -> int:
        async with self._flush_lock:
            written = 0
            while not self._queue.empty():
                batch: list[UsageRecord] = []
                while not self._queue.empty() and len(batch) < self._max_batch:
                    batch.append(self._queue.get_nowait())
                try:
                    async with self._session_factory() as session, session.begin():
                        await add_usage_records(session, batch)
                    written += len(batch)
                except Exception:
                    logger.exception("failed to persist usage batch", extra={"size": len(batch)})
            return written

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            await self.flush()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="usage-writer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.flush()
