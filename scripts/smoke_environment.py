#!/usr/bin/env python3
"""Report the resolved core environment without loading model weights."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform


PACKAGES = ("coding-opd", "torch", "transformers", "vllm", "verl", "uni-agent", "datasets", "ray")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in PACKAGES},
    }

    for module_name in ("coding_opd", "verl", "uni_agent"):
        importlib.import_module(module_name)

    import torch

    result["torch_cuda"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_device_count"] = torch.cuda.device_count()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
