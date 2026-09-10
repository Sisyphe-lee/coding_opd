# Podman tool-call latency

The sandbox wrapper uses a local Podman API service for `exec` and `cp` when
`$CODING_OPD_PODMAN_RUNROOT/podman.sock` exists (default runroot:
`/run/coding-opd-podman-overlay`). Container lifecycle and image operations
continue using the local CLI. Without the socket, the wrapper uses the original
CLI path. A failed API call is returned without retrying the tool command.

The standard external-evaluation and R2E training launchers automatically run
the following preflight before starting their workers. It reuses a healthy
service or starts one in the background, using the same storage environment:

```bash
bash scripts/run_podman_service.sh --ensure
```

Concurrent launches share a startup lock. A stale socket with no listener is
removed before restarting; a listening but unhealthy service fails the preflight
rather than silently starting a slow evaluation. Startup
logs and the service PID are under the runroot. For manual foreground/service
management, the original invocation remains available:

```bash
nohup bash scripts/run_podman_service.sh \
  > /personal/coding_opd_runtime/logs/podman_service.log 2>&1 </dev/null &
```

The socket is private to the outer container and must never be mounted into task
sandboxes. Task networking and tool semantics are unchanged. To revert, stop the
service and remove its socket after active tool calls have drained. The service
is container-local; the standard launchers start it again after an outer
container restart. Custom launchers must call the same preflight explicitly.
The presence of this document alone does not enable the service.
Launcher-only tests can set `CODING_OPD_PODMAN_SERVICE=false` to skip the
preflight without requiring a local container runtime.

The 256K evaluation enabled this path live on 2026-09-09 at 19:04:56 without
restarting models. In an initial short window, 132 completed shell calls averaged
0.60 seconds versus 12.34 seconds in the preceding three minutes; 35 editor calls
averaged 0.23 seconds versus 8.93 seconds. Task logs have one-second timestamp
resolution, and these are workload observations rather than matched-task trials.

## Diagnosis, 2026-09-09

On the Teacher 256K evaluation node, a sampled CLI call took 4.34 seconds with
0.61 seconds of CPU time; syscall sampling predominantly found waits on the
store's `storage.lock`. Lock owners were ordinary concurrent tool calls and
their cleanup processes, rather than a single stuck process. The two evaluation
nodes had identical CPU models, quotas, Podman 4.9.3, crun 1.14.1, and overlay
configuration. The slower store had 1,011 images and a 3.83 MB layer index versus
599 images and 0.63 MB on the other node; this is a possible amplifier of CLI
initialization cost, not a separately established causal result.

Alternating same-container `exec true` checks under the running workload took
1.95–4.00 seconds through the local CLI and 0.064–0.087 seconds through a
persistent service. Changing `GOMAXPROCS` did not improve the local CLI path.
These are tool-dispatch measurements, not an end-to-end evaluation speedup.
