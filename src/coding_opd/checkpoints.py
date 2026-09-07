from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


_STEP_RE = re.compile(r"^global_step_(\d+)$")


@dataclass(frozen=True)
class CheckpointSummary:
    root: Path
    step: int
    checkpoint_dir: Path
    actor_world_size: int
    has_transfer_queue: bool
    hf_model_files: tuple[Path, ...]


def checkpoint_step(path: Path) -> int:
    match = _STEP_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"checkpoint directory must be named global_step_<N>: {path}")
    return int(match.group(1))


def latest_checkpoint(root: Path) -> Path:
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise ValueError(f"missing checkpoint marker: {marker}")
    raw = marker.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        raise ValueError(f"invalid checkpoint marker {marker}: {raw!r}")
    checkpoint = root / f"global_step_{int(raw)}"
    if not checkpoint.is_dir():
        raise ValueError(f"marker points to a missing checkpoint: {checkpoint}")
    return checkpoint


def validate_checkpoint(
    root: Path,
    *,
    expected_step: int | None = None,
    require_hf_model: bool = False,
) -> CheckpointSummary:
    root = root.resolve()
    checkpoint = latest_checkpoint(root)
    step = checkpoint_step(checkpoint)
    if expected_step is not None and step != expected_step:
        raise ValueError(f"expected global step {expected_step}, found {step}")

    data_path = checkpoint / "data.pt"
    actor_dir = checkpoint / "actor"
    fsdp_config_path = actor_dir / "fsdp_config.json"
    for required in (data_path, fsdp_config_path):
        if not required.is_file():
            raise ValueError(f"missing checkpoint artifact: {required}")

    try:
        fsdp_config = json.loads(fsdp_config_path.read_text(encoding="utf-8"))
        world_size = int(fsdp_config["world_size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid FSDP config: {fsdp_config_path}") from error
    if world_size < 1:
        raise ValueError(f"invalid actor world size {world_size} in {fsdp_config_path}")

    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            shard = actor_dir / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
            if not shard.is_file() or shard.stat().st_size == 0:
                raise ValueError(f"missing or empty actor shard: {shard}")

    hf_dir = actor_dir / "huggingface"
    hf_model_files = tuple(
        sorted(
            path
            for pattern in ("*.safetensors", "*.bin")
            for path in hf_dir.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        )
    )
    if require_hf_model and not hf_model_files:
        raise ValueError(f"checkpoint has config/tokenizer but no exported HF weights: {hf_dir}")

    return CheckpointSummary(
        root=root,
        step=step,
        checkpoint_dir=checkpoint,
        actor_world_size=world_size,
        has_transfer_queue=(checkpoint / "transfer_queue").exists(),
        hf_model_files=hf_model_files,
    )
