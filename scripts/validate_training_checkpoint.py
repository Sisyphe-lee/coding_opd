#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_opd.checkpoints import validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a resumable veRL actor checkpoint.")
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--require-hf-model", action="store_true")
    args = parser.parse_args()

    summary = validate_checkpoint(
        args.checkpoint_root,
        expected_step=args.expected_step,
        require_hf_model=args.require_hf_model,
    )
    print(
        json.dumps(
            {
                "actor_world_size": summary.actor_world_size,
                "checkpoint_dir": str(summary.checkpoint_dir),
                "has_transfer_queue": summary.has_transfer_queue,
                "hf_model_files": [str(path) for path in summary.hf_model_files],
                "step": summary.step,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
