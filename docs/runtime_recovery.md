# Runtime recovery after container replacement

## Storage contract

Keep models, datasets, completed records, source bundles, OCI archives and Python
dependencies under `/personal/coding_opd_runtime`. The checkout `.venv` now points
to `/personal/coding_opd_runtime/environments/coding-opd-py312` (Python 3.12).
Cache Python downloads under `cache/uv` and system package downloads under
`cache/apt`. Host-installed Podman/crun packages still require reinstalling after
a container replacement; cached packages are recovery inputs, not executables.

The runtime Podman overlay store remains `/workspaces/coding-opd-podman-overlay`.
It is disposable. On 2026-09-07 a native-overlay probe on CPFS failed with
`upper fs missing required features` and failed upper xattrs; `/dev/fuse` was
absent. Do not copy the active overlay directory to CPFS as a portable backup.
Keep OCI archives and pinned verifier sources and re-import them instead.

## Restore and resume

`scripts/restore_runtime_and_resume_student.sh` serially restores DeepSWE agent
and verifier images, R2E-512 (including R2E-128), and canonical Verified-500.
It checks required images, a real isolated sandbox, and the test suite before
resuming Student on GPUs 0–3, four TP=1 replicas with four task slots each.
It never stops existing GPU processes and refuses an occupied GPU (>20 GiB).
The importer stops adding images if runtime disk free space falls below 100 GiB.

Prerequisites: restored Python environment; podman, skopeo, crun and working OSS
authorization; complete R2E-512 and DeepSWE archives on Personal. The Verified
downloader must publish its final archive only after size and SHA256 checks.
The DeepSWE source backup is:
`${OPD_OSS_PREFIX}/installers/deepswe-v1.1-tasks-0b9fabbb.tar.gz` (supply your own prefix).
The unpacked source is checked against the frozen tree hash before image builds.

Before invoking, require local/remote clean `lcy-dev` Git parity and passing local
tests. Export `EXPECTED_GIT_SHA` (recovery commit), `RESUME_FROM_RESULT`, the
explicitly reviewed `RESUME_SOURCE_GIT_SHA`, and a unique `RUN_NAME`. Launch with
`nohup`, redirected logs and stdin `/dev/null`. No ssctl port is persisted.

The prior eight-GPU phase `verified_student_full_8gpu_c8_resume_20260906` retained
216 completed tasks (111 passed) and added no records before the environment
failed. Resource-only resume imports these records with their exact original
provenance. Model/sampling/context/commit requirements and official grading are
unchanged; the pending auto-compaction experiment must not be mixed into this run.
Failures during recovery stop the pipeline before evaluation. An incomplete import
can be rerun and skips existing images; an incomplete archive extraction requires
inspection rather than deletion or overwriting.
