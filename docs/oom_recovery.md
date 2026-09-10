# September 8 training OOM recovery

Vanilla run `r2e512_opd_431_tok21504_parallel3_20260908_0005` completed
step 89. In step 90, sample 30 changed Pillow's `ImageSequence.all_frames`
to copy and append frames in an infinite loop without advancing the image.
Two agent-launched `test_all_frames` processes survived 120-second command
timeouts and reached 499.63 GB and 307.30 GB RSS. Ray reported 974.18/1024 GB
host memory and killed a rollout worker. The step 64 checkpoint survived;
the old container's final kernel termination record was not recovered.

The shared sandbox now applies an inherited 8 GiB per-process virtual-address
limit using `ulimit -v`. This is not an aggregate container RSS limit: nested
Podman currently disables cgroups. GNU `timeout --signal=KILL` runs inside the
container so timed-out command groups are killed, including children ignoring
SIGTERM. A client timeout or task cancellation additionally removes the sandbox.
Both agent tools and post-agent tests use the same execution path. These limits
change task execution behavior identically for Vanilla and TCOD.

TCOD run `r2e512_tcod_431_tok21504_grow2_20260908_004507` completed step 6
and failed while hybrid vLLM woke its KV cache after an Actor update. Sampled
GPU peaks on its four Actor cards (3–6) were about 270,100 MiB out of 275,040 MiB.
The exception is in `wake_up(tags=["kv_cache"])`, not Actor forward/backward.
Actor weights and optimizer state remain resident; startup profiling does not
guarantee sufficient space after training's lazy allocations. The precise
allocation breakdown was not logged, so lazy optimizer state is a plausible
contributor rather than a separately measured cause.

Recovery reduces hybrid `ROLLOUT_GPU_MEMORY_UTILIZATION` from 0.8 to 0.6,
adding roughly 54 GiB headroom per Actor GPU. Standalone rollout remains 0.7,
Teacher 0.5, Actor token budget 21,504, and topology 4+3+1. Apply the same
setting to both algorithms. Resume Vanilla from step 64; TCOD has no checkpoint
and restarts from the original Student. Observe multiple post-update KV wakeups
before claiming runtime recovery is verified.
