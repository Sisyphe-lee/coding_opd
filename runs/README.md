# Training runs

Each subdirectory is one logical training run. A run contains only compact,
reviewable artifacts:

- `summary.json`: algorithm, checkpoint lineage, source log segments and status.
- `config.yaml`: the fully resolved training configuration needed to reproduce the run.
- `loss_curve.png`: distillation loss by optimizer step.
- `metrics.csv`: one row per optimizer step, including loss, MFU and stage timing.
- `profile_summary.csv`: aggregate counts and durations for lightweight profile events.

Raw logs, per-process profile JSONL, GPU samples, rollouts and checkpoints remain
under the ignored `runtime/` tree. New training uses `runs/$RUN_NAME`. Resume reads
`run_record_name` from the source checkpoint root and updates the same directory.

W&B is optional. Keep the default console logger, or set
`TRAIN_LOGGER="['console','wandb']"` after configuring W&B credentials.
