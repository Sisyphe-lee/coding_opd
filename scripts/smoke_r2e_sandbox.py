#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from uni_agent.sandbox.docker import DockerSandbox


DEFAULT_IMAGE = "namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d"


async def smoke(binary: str, image: str, disable_cgroups: bool) -> None:
    run_args = ["--network", "none"]
    if disable_cgroups:
        run_args.append("--cgroups=disabled")

    sandbox = DockerSandbox(
        image=image,
        docker_binary=binary,
        run_args=run_args,
        pull_policy="never",
    )
    async with sandbox:
        setup = await sandbox.exec(
            [
                "bash",
                "-lc",
                "mv /r2e_tests /root/r2e_tests && ln -s /root/r2e_tests /testbed/r2e_tests",
            ]
        )
        if setup.exit_code != 0:
            raise RuntimeError(setup.stderr or setup.stdout)

        result = await sandbox.exec(["bash", "./run_tests.sh"], timeout=120, workdir="/testbed")
        output = result.stdout + result.stderr
        if result.exit_code != 1 or "1 failed, 9 passed" not in output:
            raise RuntimeError(f"unexpected R2E baseline result (exit={result.exit_code}):\n{output[-4000:]}")

        print(
            json.dumps(
                {
                    "binary": binary,
                    "image": image,
                    "network": "none",
                    "baseline": "9 passed, 1 failed",
                }
            )
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=str(repo_root / "scripts" / "podman_sandbox"))
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disable-cgroups", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    asyncio.run(smoke(args.binary, args.image, args.disable_cgroups))


if __name__ == "__main__":
    main()
