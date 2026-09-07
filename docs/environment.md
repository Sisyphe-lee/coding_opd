# Coding OPD environment

This document specifies the reproducible Coding OPD environment recipe. Dataset selection and
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
| FlashAttention | 2.8.3 |
| fla-core / flash-linear-attention | 0.5.2 / 0.5.2 |
| SWE-bench test harness | 4.1.0 |
| Codex CLI reference binary | 0.145.0 (installed separately; not supplied by Python bootstrap) |

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
- `scripts/bootstrap_python_env.sh train` creates the CUDA/FSDP/vLLM environment. Keep `.venv`
  at the checkout root, or symlink it to your provisioned environment on durable storage.
  Do not assume that container-local Python packages survive container replacement.

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

The Codex configurations use `scripts/podman_sandbox`: native overlay at
`/workspaces/coding-opd-podman-overlay`, runroot `/run/coding-opd-podman-overlay`, and `crun`.
The reference system tool versions are Podman 4.9.3, Skopeo 1.13.3, and crun 1.14.1.
These are system dependencies, not Python packages. Provision an overlay-compatible filesystem
and inspect permissions before use; paths can be overridden through the wrapper's environment.

For the alternative Docker backend, `scripts/bootstrap_docker_daemon.sh` starts a project-scoped
nested daemon using `/var/lib/coding-opd-docker`. It tries `overlay2` first and falls back to
`vfs`; `vfs` is slower and consumes more disk. Do not confuse this daemon with the Podman store.

The outer container cannot program host NAT rules, so the nested daemon disables bridge
networking and iptables. Agent-controlled task containers must use `--network none`; do not give
them host networking.

For machines without public registry access, initial acquisition happens on a machine
with registry access, preferably through a verified mainland Docker Hub mirror and with the
workstation HTTP(S) proxy explicitly bypassed:

```bash
scripts/download_oci_images.py --direct ...
```

Every layer digest and size is verified before the OCI archive is uploaded under
your own OSS project prefix (`OPD_OSS_PREFIX`), under `docker_images/`. Repeated deployments load
that OSS copy. Only images selected by the frozen R2E-128/R2E-512 manifests should be
materialized; there is no requirement to download the complete upstream task pool.

## Reproduction preflight

1. Initialize the three pinned top-level submodules and install the chosen Python profile.
   Use the lock and bootstrap together: the bootstrap deliberately overrides CUDA wheels and
   adds explicitly versioned compatibility packages. Optional OSS delivery requires your own
   `OPD_OSS_PREFIX`; no private archive access is included.
2. Provision the Student and Teacher weights. Directory names do not pin weight bytes: record
   model source revision and file hashes for each experiment, or identify the exact OPD export.
3. Install the reference Codex binary and system container tools. The YAML files contain bind
   mount paths; adapt `TASK_CONFIG` as well as launcher paths when using another machine.
4. Fetch fixed dataset revisions, materialize the selected frozen pools, import task images,
   and build DeepSWE verifier images. Check image availability before model startup.
5. Keep vLLM/Triton/Inductor JIT caches on local disk using `VLLM_CACHE_ROOT`,
   `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR`. Keep results and portable OCI backups durable.
6. Run all project tests and an isolated sandbox check. Inspect GPU ownership and host resources.
   Training requires a dedicated Ray cluster; the Codex evaluator does not require Ray.
7. Record code commit, resolved package/system versions, model identity, dataset manifest hashes,
   image digests, sampling configuration, and resource settings with the run. Keep code parity
   between development and execution environments.

This document is a configuration recipe, not a deployment progress log or a claim that a fresh
checkout contains models, datasets, prebuilt images, or completed evaluation results.
