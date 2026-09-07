#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_opd.eval_data import sha256_file
from coding_opd.eval_reporting import build_external_eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a frozen quick-panel score from one full external-evaluation result."
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("runtime_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    manifest = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    actual_manifest_hash = sha256_file(args.runtime_manifest)
    recorded_manifest_hash = result.get("runtime_manifest_sha256")
    if recorded_manifest_hash != actual_manifest_hash:
        raise ValueError(
            "result/runtime manifest hash mismatch: "
            f"recorded {recorded_manifest_hash}, found {actual_manifest_hash}"
        )
    result["result_path"] = str(args.result.resolve())
    summary = build_external_eval_summary(result, manifest)
    summary["runtime_manifest"] = str(args.runtime_manifest.resolve())
    summary["runtime_manifest_sha256"] = actual_manifest_hash
    output = args.output or args.result.with_name("summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "benchmark": summary["benchmark"],
                "full_score_percent": summary["full"]["score_percent"],
                "quick_score_percent": summary["quick"]["score_percent"],
                "summary": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
