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

On 2026-09-08, the zero-warmup R2E-512 runs
`r2e512_{tcod,vanilla}_431_tok21504_profile_blas1_20260908_140105`
showed a scheduling gap after all 32 prompts finished and before the next
batch started. Across six gaps, TCOD averaged 56.10 seconds and Vanilla
53.91 seconds. During those windows the three standalone rollout GPUs had
zero utilization in 98.99% and 96.91% of the five-second samples, respectively.
Evidence is in the run's `.profile/` and `.gpu.csv` files under the remote
runtime log directory. These are observed idle windows, not a measured speedup.

The async launcher now defaults to `ASYNC_PREFETCH=true` with
`ASYNC_WARMUP_BATCHES=0`. After sampling the first mini-batch, the trainer
submits one extra batch so standalone rollout can run during Actor updates.
The next step consumes that submission instead of advancing the dataloader
again. Save/final steps submit no lookahead; the remaining batch drains through
the normal sampler, including one-for-one replacement of failed/stale prompts.
Before saving, the trainer requires zero unconsumed prompt markers. This keeps
the dataloader consistent with the weights without persisting a rollout queue.
`FIRST_CHECKPOINT_STEP=16` adds one early save when checkpointing is enabled;
with `SAVE_FREQ=64`, saves occur at 16, 64, 128, 192, 256 (and the final step
if it differs). The early save drains prefetch too and is not repeated after
resuming past step 16. Set `FIRST_CHECKPOINT_STEP=-1` to disable that extra save.
Batch size, GPU allocation and weight-sync frequency remain launcher settings.
Set `ASYNC_PREFETCH=false` (keeping warmup zero) for
the serial-dispatch comparison. Prefetch requires off-policy threshold >= 2;
continue reporting measured policy lag and trajectory spans.

The CPU regression exercises the pinned veRL sampler and its actual dataloader
save path, including save/final boundaries, out-of-order completion, failure
and stale-sample replacement, and resumption without lost/duplicate rows.
Runtime throughput after enabling prefetch must be measured separately; skip
initialization/JIT steps and checkpoint drain steps when comparing steady state.
