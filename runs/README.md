# Training runs

Each subdirectory is one logical training run. A run contains only compact,
reviewable artifacts:

- `summary.json`: algorithm, checkpoint lineage, source log segments and status.
- `metrics.csv`: one row per optimizer step, including loss, MFU and stage timing.
- `profile_summary.csv`: aggregate counts and durations for lightweight profile events.

Raw logs, per-process profile JSONL, GPU samples, rollouts and checkpoints remain
under the ignored `runtime/` tree. New training uses `runs/$RUN_NAME`. Resume reads
`run_record_name` from the source checkpoint root and updates the same directory.
