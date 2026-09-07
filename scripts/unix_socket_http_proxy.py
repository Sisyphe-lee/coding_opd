#!/usr/bin/env python3
"""Expose an HTTP Unix-domain socket on container-local TCP loopback.

This is intentionally a byte-for-byte transport proxy.  It lets an agent CLI
that only accepts an HTTP URL reach a host-mounted model socket while the task
container remains on ``--network none``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib


async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(64 * 1024):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(ConnectionError, RuntimeError):
            writer.write_eof()


async def _forward(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    unix_socket: str,
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_unix_connection(unix_socket)
    except OSError:
        client_writer.close()
        await client_writer.wait_closed()
        return

    try:
        await asyncio.gather(
            _copy(client_reader, upstream_writer),
            _copy(upstream_reader, client_writer),
        )
    finally:
        upstream_writer.close()
        client_writer.close()
        await asyncio.gather(
            upstream_writer.wait_closed(),
            client_writer.wait_closed(),
            return_exceptions=True,
        )


async def _serve(args: argparse.Namespace) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: _forward(reader, writer, unix_socket=args.unix_socket),
        host=args.host,
        port=args.port,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unix-socket", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
