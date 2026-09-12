import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_training_run.py"


def test_export_merges_resume_steps_and_profiles(tmp_path):
    first = tmp_path / "first.log"
    first.write_text(
        "step:1 - actor/distillation/loss:0.2 - perf/mfu/actor:0.1 - training/global_step:1\n"
        "step:2 - actor/distillation/loss:0.15 - training/global_step:2\n"
    )
    second = tmp_path / "second.log"
    second.write_text(
        "step:2 - actor/distillation/loss:0.14 - training/global_step:2\n"
        "step:3 - actor/distillation/loss:np.float64(0.1) - training/global_step:3\n"
    )
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "worker.jsonl").write_text(
        json.dumps({"event": "tool", "duration_s": 1.0, "status": "ok"}) + "\n"
        + json.dumps({"event": "tool", "duration_s": 3.0, "status": "error"})
        + "\n"
    )
    record = tmp_path / "record"

    for segment, log, profile_dir in (
        ("first", first, None),
        ("second", second, profile),
    ):
        command = [
            sys.executable,
            str(SCRIPT),
            "--record-dir",
            str(record),
            "--segment",
            segment,
            "--log",
            str(log),
            "--algorithm",
            "vanilla",
            "--status",
            "completed",
            "--checkpoint-dir",
            "/checkpoints/run",
        ]
        if profile_dir:
            command += ["--profile-dir", str(profile_dir)]
        subprocess.run(command, check=True)

    with (record / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["1", "2", "3"]
    assert rows[1]["actor/distillation/loss"] == "0.14"
    assert rows[2]["actor/distillation/loss"] == "0.1"

    with (record / "profile_summary.csv").open() as handle:
        profiles = list(csv.DictReader(handle))
    assert profiles[0]["event"] == "tool"
    assert profiles[0]["count"] == "2"
    assert profiles[0]["mean_s"] == "2.0"
    assert profiles[0]["errors"] == "1"

    summary = json.loads((record / "summary.json").read_text())
    assert summary["last_step"] == 3
    assert summary["final_loss"] == 0.1
    assert [item["name"] for item in summary["segments"]] == ["first", "second"]
