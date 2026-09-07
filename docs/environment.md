# Coding OPD environment

This document describes the environment currently used by Coding OPD. Dataset selection and
frozen task manifests are documented separately in [`datasets.md`](datasets.md); measured
SWE-Smith throughput is recorded in [`performance_optimization.md`](performance_optimization.md).

## Validated software stack

The project uses the following pinned sources and package versions:

| Component | Pinned source/version |
|---|---|
| Uni-Agent | `d1b6e6dbe6b9a7f7f4a5a22c800e4f3bbfdd8fda` |
| veRL | `60577a786225c15e2e859f47fc7f4e55bcc13720` |
| R2E-Gym | `0d94c4eb9431cd195c55a7ea3abd54006c9a1735` |
| SWE-Smith fixture | `9b74ac08118a85c39c356802f7961893af73e07f` |
| Python | 3.12 |
| PyTorch | 2.11.0, CUDA 13.0 wheel |
| vLLM | 0.24.0 |
| Transformers | 5.9.0 |
| datasets | 5.0.0 |
| Ray | 2.55.1 |

Python packages are resolved from the pinned `third_party/verl/uv.lock` with the FSDP and vLLM
extras. Uni-Agent is installed without its nested veRL checkout; imports resolve to the
repository's top-level `third_party/verl` source.

R2E-Gym is the active research training environment. SWE-Smith is installed from its pinned
source archive with `--no-deps` only to reproduce the measured infrastructure and throughput
fixture. It is not a current training or evaluation dataset.

## Local and B300 profiles

Both profiles use a repository-root `.venv` and the same source commits:

- `scripts/bootstrap_python_env.sh dev` creates the lightweight local CPU environment for
  editing, tests, data preparation, and configuration validation. It does not contain model
  checkpoints or the full CUDA training stack.
- `scripts/bootstrap_python_env.sh train` creates the remote CUDA/FSDP/vLLM environment. In the
  Robby checkout, `.venv` points to `/var/lib/coding-opd-venv`.

The remote machine has 8 NVIDIA B300 GPUs with approximately 275 GiB visible memory per GPU,
NV18 links, about 1 TiB host RAM, and 256 CPUs. Some system interfaces incorrectly report the
GPU as `NVIDIA L20D`; kernel selection, memory limits, and MFU calculations must use the actual
B300 hardware. The validated kernel settings are:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
attention_backend=FLASH_ATTN
gdn_prefill_backend=triton
```

The student and teacher checkpoints remain remote-only:

- Student: `/personal/coding_opd_runtime/models/Qwen3.5-9B`
- Teacher: `/personal/coding_opd_runtime/models/Qwen3.8-27B`

## R2E-Gym integration

R2E-Gym's complete package is deliberately not installed into the training environment. Its
upstream dependency set pins `datasets==2.19` and includes cloud, Kubernetes, and provider
packages that conflict with or are unnecessary for the selected veRL lock (`datasets==5.0.0`).

The project instead pins the upstream source and implements the required compatibility layer in
`src/coding_opd`:

1. dataset validation and deterministic manifest generation;
2. task-image selection and sandbox lifecycle;
3. repository shell/edit tools;
4. withheld-test restoration and binary reward calculation;
5. trajectory metadata used by OPD training and diagnostics.

`RaySafeDockerSandbox` avoids subprocess/SIGCHLD deadlocks in Ray workers. Its shell tool does
not depend on tmux because R2E containers run without networking and the validated Orange3 image
does not include tmux.

## Nested Docker and image delivery

The outer Robby container has no host Docker socket. `scripts/bootstrap_docker_daemon.sh` starts
a project-scoped rootful nested daemon using `/var/lib/coding-opd-docker`. It tries `overlay2`
first and falls back to `vfs`; `vfs` is slower and consumes more disk.

The outer container cannot program host NAT rules, so the nested daemon disables bridge
networking and iptables. Agent-controlled task containers must use `--network none`; do not give
them host networking.

Robby does not pull task images from public registries. Initial acquisition happens on a machine
with registry access, preferably through a verified mainland Docker Hub mirror and with the
workstation HTTP(S) proxy explicitly bypassed:

```bash
scripts/download_oci_images.py --direct ...
```

Every layer digest and size is verified before the OCI archive is uploaded under
your own OSS project prefix (`OPD_OSS_PREFIX`), under `docker_images/`. Repeated deployments load
that OSS copy. Only images selected by the frozen R2E-128/R2E-512 manifests should be
materialized; there is no requirement to download the complete upstream task pool.

## Current validation status

Completed:

- local and remote environment import/tests;
- CUDA kernels and vLLM model loading on B300;
- nested Docker startup and imported-image execution;
- one real Orange3 trajectory, deterministic R2E tests, and reward calculation;
- aligned Qwen teacher logprobs and Qwen student K3 loss/backward;
- one optimizer step through the complete R2E path;
- eight-GPU asynchronous infrastructure/throughput validation using the SWE-Smith fixture.
- four optimizer steps on the frozen R2E-128 runtime pool;
- a two-process R2E-128 checkpoint save/resume run through global step 2, including restoration of
  all four FSDP model/optimizer/RNG/LR-scheduler shards plus dataloader state and final HF export;
- code and validation gates for manifest-backed SWE-bench Verified / DeepSWE evaluation.

The end-to-end R2E smoke completed with `training/global_step=1`, distillation loss `0.08714`,
gradient norm `3.48986`, and reward `0.9`. This validates the integration, not model quality or
the frozen training pools.

Still pending:

- materialize and verify R2E-512 (not required for the checkpoint/resume milestone);
- materialize the v2 Verified/DeepSWE runtime bundles and their selected images;
- build DeepSWE's separate verifier images and run the first quick external evaluations.

Before remote GPU work, run `.venv/bin/pytest -q` both locally and remotely and require all
collected project tests to pass. Formal runs additionally require a dedicated Ray cluster,
known-free GPUs, and local/remote Git SHA parity as described in `AGENTS.md`.
