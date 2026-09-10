# Training profiling

R2E training records lightweight wall-clock spans by default under
`$LOG_DIR/$RUN_NAME.profile/<hostname>.<pid>.jsonl`. Set
`CODING_OPD_PROFILE_DIR=` to disable. The launcher forwards the setting to all
Ray processes. No new service, package, CUDA synchronization, or training
schedule change is required. Existing step metrics and GPU CSV sampling remain
the source for Actor time, switching, weight sync, and GPU occupancy.

Each completed span has `event`, `start_ns` (Unix time), `duration_s` (monotonic
elapsed time), `status`, host/PID, and its available sample identifiers. Records
contain counts/identifiers, not prompts, token arrays, tool commands, or outputs.
Cancellation and exceptions retain their original behavior and are recorded.
An unfinished span has no record yet. Separate files isolate concurrent processes;
ContextVar isolates concurrent sessions and propagates to asyncio child tasks.

| Event | Boundary |
|---|---|
| `prompt` | Prompt dispatch through finished/failure status publication to TQ. |
| `session_wait_and_run` | Waiting for a session slot plus running the episode. |
| `agent_episode` | Slot acquired through Ray runner completion, gateway finalization and trajectory dump. |
| `agent_task` | Runner on its Ray worker, including sandbox setup/cleanup. |
| `tool` | Shell/editor invocation, including subprocess or filesystem work. |
| `sandbox` | One Podman command; `operation`, image and exit code identify setup, exec, copy or removal. |
| `rollout_request` | Existing LLM client call, including routing, retries, server queueing and inference; excludes gateway tokenization/decoding. |
| `teacher_request` | Teacher LLM client call, including server queueing and forward pass. |
| `teacher_score` | All Teacher requests for a session plus output conversion. |
| `tq_write` | Trajectory conversion/serialization and asynchronous data write. |
| `rollout_prefetch` | Trainer submission of one future batch after sampling the first mini-batch; rollout itself completes asynchronously. |

Join framework/Teacher events by `(uid, session)` within the run directory.
`agent_task` maps those identifiers to `session_id`, which joins model requests
from the gateway process. `step` is the prompt's dispatch step, not necessarily
the optimizer step that consumes it. Continue to report training policy lag.
Slot waiting is the difference of `start_monotonic_ns` between `agent_episode`
and `session_wait_and_run` in the same process for the
same `(uid, session)`. TQ data-write completion precedes the prompt's final status
publication; use `prompt` completion when reasoning about sample readiness.

Spans nest and overlap: do not add their durations as step wall time. Client
timings do not separately measure GPU compute and server queue time. GPU CSV
and vLLM statistics supply complementary evidence; use a short existing
Nsight/PyTorch profiler capture only if kernel-level diagnosis becomes necessary.
JSONL writes use the standard logging file handler, one open file per process;
no payload copies, cross-process locks, background collectors or per-event fsync.

During the first sandbox profiling check, a NumPy import/simple matrix product
printed its result but hung at Python exit under the 8 GiB address-space limit.
On the same image, `OPENBLAS_NUM_THREADS=1` completed in about 0.16 seconds with
the same result; default threads with a 32 GiB limit also completed normally.
R2E task containers now set `OPENBLAS_NUM_THREADS=1`, retaining the 8 GiB
protection. This setting applies to sandbox CPU commands, not the GPU workers.

Formal training uses one checkpoint-safe prefetched batch. Save and final steps
stop lookahead, drain the remaining batch, and require zero unconsumed prompt
markers before persisting dataloader state. Compare steady-state steps only;
initialization, JIT and checkpoint-drain steps have different timing. Always
report measured policy lag and trajectory span with throughput. Adopted results
are summarized in `training_infrastructure.md`.
