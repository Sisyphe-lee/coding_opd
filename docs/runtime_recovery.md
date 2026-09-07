# Runtime recovery recipe

Keep models, runtime parquet, frozen metadata, completed records, verifier sources,
and portable OCI archives on durable storage. Container-local system packages,
overlay image stores, and JIT caches may need reinstalling or rebuilding.

Use a filesystem compatible with native overlay for the Podman store. Copying an
active overlay directory onto an unsupported network filesystem is not a portable
backup strategy; retain verified OCI archives and re-import them instead.

## Restore dependencies and images

Provision the Python and system dependencies from [environment.md](environment.md).
Restore the image sets corresponding to the frozen manifests described in
[datasets.md](datasets.md), not an obsolete training split. R2E-512 includes R2E-128.
DeepSWE requires both task images and independently built verifier images.

`scripts/restore_runtime_and_resume_student.sh` is an optional deployment-specific
helper. It restores DeepSWE, R2E-512, and Verified-500 serially, checks images and a
real sandbox, then launches Student evaluation. It assumes pre-provisioned archives
and source bundles, working OSS authorization, and the directory layout in the script.
Supply your own `OPD_OSS_PREFIX` if the verifier source archive must be downloaded.
The importer requires at least 100 GiB free before adding images.

The helper's launch is fixed to GPUs 0–3, TP1, four tasks per replica; it refuses
GPUs using more than 20 GiB but that threshold is not a substitute for ownership
inspection. To use a different allocation, restore/check assets separately and
invoke the evaluation launcher with explicit resource settings.

## Resume evaluation without changing its meaning

- Stop only the intended source run, retaining result.json, per-task records, and logs.
- Use a new run/output directory and set `RESUME_FROM_RESULT` to the stopped source.
- Keep the same model, dataset, sampling, context, and scoring protocol. Only supported
  resource-allocation fields may change. Code changes require an explicitly reviewed
  `RESUME_SOURCE_GIT_SHA`; do not approve a semantic change as a resource-only resume.
- Preserve all completed records, including zeros. Incomplete tasks restart from scratch.
- Verify inherited counts and task IDs before interpreting new results.

Require clean code parity and passing tests before launch. Detach long-running jobs
with redirected logs and stdin from `/dev/null`. Do not terminate other projects'
processes. An incomplete import can skip existing images; inspect partial extraction
or checksum failures rather than deleting unrelated storage.
