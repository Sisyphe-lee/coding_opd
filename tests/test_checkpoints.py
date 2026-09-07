from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_opd.checkpoints import latest_checkpoint, validate_checkpoint


def _checkpoint(root: Path, *, step: int = 2, world_size: int = 2, hf: bool = False) -> Path:
    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")
    (checkpoint / "data.pt").write_bytes(b"data")
    (actor / "fsdp_config.json").write_text(json.dumps({"world_size": world_size}), encoding="utf-8")
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"state")
    if hf:
        hf_dir = actor / "huggingface"
        hf_dir.mkdir()
        (hf_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return checkpoint


def test_validate_checkpoint_checks_all_resumable_actor_shards(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, step=2, world_size=2)
    summary = validate_checkpoint(tmp_path, expected_step=2)
    assert summary.checkpoint_dir == checkpoint
    assert summary.actor_world_size == 2
    assert not summary.hf_model_files


def test_validate_checkpoint_requires_exported_weights_when_requested(tmp_path: Path) -> None:
    _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="no exported HF weights"):
        validate_checkpoint(tmp_path, require_hf_model=True)


def test_validate_checkpoint_detects_missing_rank(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "actor" / "optim_world_size_2_rank_1.pt").unlink()
    with pytest.raises(ValueError, match="missing or empty actor shard"):
        validate_checkpoint(tmp_path)


def test_latest_checkpoint_rejects_stale_marker(tmp_path: Path) -> None:
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7", encoding="utf-8")
    with pytest.raises(ValueError, match="missing checkpoint"):
        latest_checkpoint(tmp_path)
