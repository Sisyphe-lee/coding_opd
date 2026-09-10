"""Small wall-clock spans for OPD's async pipeline; no GPU synchronization."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

_context: ContextVar[dict] = ContextVar("opd_profile_context", default={})
_host = socket.gethostname()
_warned = False


@lru_cache(maxsize=8)
def _writer(directory: str, pid: int) -> logging.Logger:
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger = logging.Logger(f"opd_profile.{pid}.{directory}")
    logger.addHandler(logging.FileHandler(Path(directory) / f"{_host}.{pid}.jsonl"))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@contextmanager
def profile_span(name: str, **fields):
    """Record a completed span and inherit only explicit scalar identifiers.

    Nested/parallel spans overlap: their durations must not be summed as step
    time. An empty CODING_OPD_PROFILE_DIR disables recording. Separate files
    per process avoid cross-process locks; logging supplies the thread lock.
    """
    directory = os.environ.get("CODING_OPD_PROFILE_DIR", "")
    if not directory:
        yield {}
        return
    fields = {**_context.get(), **fields}
    token = _context.set(fields)
    wall_start, start = time.time_ns(), time.perf_counter_ns()
    status = "ok"
    try:
        yield fields
    except BaseException as exc:
        status = type(exc).__name__
        raise
    finally:
        elapsed = time.perf_counter_ns() - start
        _context.reset(token)
        record = {
            **fields, "event": name, "start_ns": wall_start, "start_monotonic_ns": start,
            "duration_s": elapsed / 1e9, "status": status,
            "host": _host, "pid": os.getpid(),
        }
        try:
            _writer(directory, os.getpid()).info(json.dumps(record, default=str))
        except OSError:
            # Observability must not discard a successfully generated sample.
            global _warned
            if not _warned:
                logging.getLogger(__name__).exception("Cannot write OPD profile")
                _warned = True


class ProfiledLLMClient:
    """Time the existing async client, including routing/queueing and inference."""

    def __init__(self, client, role: str):
        self.client = client
        self.role = role

    async def generate(self, **kwargs):
        fields = {"request_id": kwargs["request_id"], "input_tokens": len(kwargs["prompt_ids"])}
        if self.role == "rollout":
            fields["session_id"] = kwargs["request_id"]
        with profile_span(f"{self.role}_request", **fields) as span:
            output = await self.client.generate(**kwargs)
            span["output_tokens"] = len(output.token_ids)
            return output
