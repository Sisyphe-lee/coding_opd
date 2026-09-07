# Coding OPD public workspace

- Repository: `git@github.com:Sisyphe-lee/coding_opd.git`, branch `main`.
- This is a fresh-history public snapshot. Private deployment instructions and
  credentials are not included. Do not assume access to the original cluster.
- Use Python 3.12 and pinned top-level submodules. Do not recursively initialize
  Uni-Agent's nested veRL submodule.
- Run the complete test suite before deployment. Inspect GPU/process ownership
  before launching work; never stop another project's processes.
- R2E-128 and R2E-512 manifests define training. Verified-64/500 and DeepSWE-Tura20/113
  are external evaluation only. SWE-Smith is a legacy smoke fixture.
- Baseline loss is sampled-token `k3`, without policy-gradient or task rewards.
  The asynchronous trainer is not strictly on-policy; report policy lag.
- Infrastructure entry point: `scripts/run_swesmith_opd_async_smoke.sh`, batch 32,
  `parameter_sync_step=2`. This fixture does not support model-quality claims.
- Keep task containers network-isolated. Never commit models, datasets, archives,
  checkpoints, logs, credentials, or private deployment configuration.
- `/personal`, `/workspaces`, and `/var/lib` paths are deployment examples.
  Configure paths for your machine; use compatible local storage for overlay
  image stores and JIT caches, and durable storage for results.
- See README.md and docs/datasets.md for protocols and entry points.
