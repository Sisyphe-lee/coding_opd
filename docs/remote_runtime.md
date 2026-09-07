# Deployment configuration

Private machine handoff notes are excluded from this public snapshot. No hosted
models, image archives, credentials, or managed cluster access are supplied.

Launchers expose settings through environment variables such as `REPO_ROOT`,
`RUNTIME_ROOT`, `MODEL_PATH`, `CUDA_VISIBLE_DEVICES`, and `TASKS_PER_REPLICA`.
Consult each launcher for its interface. Example paths under `/personal` must be
provisioned or overridden on your own machine.

Keep models, datasets, OCI backups, and results on durable storage. Use compatible
local storage for Podman overlay layers and compilation caches. Do not use an
unsupported network filesystem for a native overlay upper directory.

For Codex evaluation, configure `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, and
`TORCHINDUCTOR_CACHE_DIR` on local disk. Four TP1 replicas with
`TASKS_PER_REPLICA=8 MAX_NUM_SEQS=8` run up to 32 tasks. Start conservatively and
monitor final task results as well as resource pressure.

Materialize the frozen manifests and check images before evaluation. Resource-only
resume retains completed scores; incomplete tasks restart. Do not change model,
sampling, or grading semantics during resume.

No new license is assigned by this snapshot. Third-party submodules retain their
own licenses and version pins. See LEGAL.md and the upstream repositories.
