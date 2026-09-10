# Coding OPD algorithms

`src/coding_opd/algorithms/` owns algorithm decisions; R2E task execution,
Teacher sampled-token scoring, and Actor K3 loss remain shared. `get_algorithm`
selects `vanilla.py`, `tcod.py` or `adaptive.py` by an explicit name. The common
interface is `rollout_max_turns(step, max_turns)` and the optional online
`retained_turns(entropies)` frontier. Vanilla and TCOD use all valid generated
tokens within their horizons. Adaptive selects a prefix during interaction.

The R2E launchers accept:

```bash
OPD_ALGORITHM=vanilla bash scripts/run_r2e_opd_train.sh
OPD_ALGORITHM=tcod TCOD_GROWTH_INTERVAL=2 bash scripts/run_r2e_opd_train.sh
OPD_ALGORITHM=adaptive ADAPTIVE_THRESHOLD=0.1 bash scripts/run_r2e_opd_train.sh
```

Default: `vanilla`. TCOD limits actual ReAct interaction to
`min(max_turns, 1 + step // growth_interval)`, with a default interval of 2.
For steps 0/1, 2/3, and 16 the limits are 1, 2, and 9 turns respectively;
with the R2E default maximum of 24, step 46 reaches the full horizon.
The interval is a configurable curriculum parameter, not checkpoint save frequency
and not a measured optimum for coding tasks. Existing token caps and natural
episode termination can still end an episode earlier.

Pure OPD training defaults `RUN_TASK_EVALUATION=false`: after Agent interaction,
return the trace directly for Teacher scoring instead of running `run_tests.sh`.
Agent-selected tools/tests still execute. Framework validation always evaluates;
standalone tasks default to evaluation. Skipped evaluation returns a neutral
reward placeholder with `evaluation_ran=false` and no accuracy/resolved result;
the training reward placeholder is not a measured success rate. Set
`RUN_TASK_EVALUATION=true` explicitly when training-time executable metrics are
needed. Both Vanilla and TCOD use the same setting.

Progress uses the trainer `global_steps` attached when a prompt is dispatched.
It stays fixed for that prompt, even if asynchronous training advances while
the task waits or runs. This is not a count of tool calls or optimizer minibatches.
Resume uses the restored trainer's progress without another curriculum counter.
Comparisons can use the existing training step axis; policy lag remains governed
by the existing asynchronous configuration.

`OPDGatewayAgentFramework` passes explicit step and train/validation metadata to
`run_r2e_task`. TCOD resolves YAML plus per-sample task overrides before limiting
`agent.max_steps`; copies isolate concurrent tasks. Validation keeps the full
configured horizon, including when validation carries a nonzero global step.
Standalone evaluation should use the default Vanilla runner behavior. Selecting
TCOD without explicit framework progress raises an error rather than silently
restarting the curriculum at zero. Task logs record TCOD step and horizon.

## Adaptive: original t0100 rule, online execution

The initial rule matches ALFWorld's `adaptive_t0100_fast8_seed44.yaml`:

- Compute Teacher Top-16 **partial entropy** `-sum(p * log(p))` at each generated
  token, using the original probabilities, natural logarithms and no Top-K
  renormalization. Average over the generated tokens in each turn.
- Use the mean of the first three turns as the fixed baseline. Starting at turn
  four, stop when the latest three-turn mean exceeds the baseline by at least
  `ADAPTIVE_THRESHOLD` (default **0.1**). This is a window-average condition, not
  three separate threshold crossings. As in the original, early windows can
  overlap baseline turns.
- Exclude the triggering turn and retain at least three turns. A trigger at
  turn six retains turns one through five. That sixth turn must be generated
  and scored to detect the frontier; its tools and all subsequent turns are
  skipped. Without a trigger, normal turn/token/time limits apply.

The rule is causal, so online execution preserves the original prefix-selection
semantics under fixed Student weights and identical Teacher scoring. Validation
bypasses Adaptive and retains the full configured horizon. The 0.1 threshold is
an ALFWorld starting point, not a measured optimum for coding.

The formal `scripts/run_r2e_opd_train.sh` entry point uses batch=minibatch=32, one
optimizer update and one weight sync per batch. Vanilla and TCOD checkpoint-safely
prefetch one following batch during the Actor update; steady-state measurement showed
trajectory staleness mean/max 1 and span max 1. Adaptive keeps the complete batch
barrier because its online Teacher frontier depends on the current policy, so it
disables prefetch. All algorithms set `parameter_sync_step=1`, `ppo_epochs=1`, warmup=0
and disable hybrid GPU lending. A reported step is one optimizer update; the TCOD
curriculum uses that step.

The legacy asynchronous performance-smoke launcher retains its measured recipe.
`OPD_SYNC_ROLLOUTS=true` on that launcher adds a batch barrier; use the formal training
entry point for the current bounded-lag configuration.

Implementation boundaries:

- `algorithms/adaptive.py`: pure frontier rule, independent of Ray and vLLM.
- `adaptive_teacher.py` / `vllm_server.py`: one Teacher forward
  returns sampled-token labels plus this turn's entropy and Top-16 covered mass.
  The actual Student token is preserved even when its Teacher rank is outside
  Top-16. Diagnostics never become a Top-K distillation loss.
- `adaptive_rollout.py` / `opd_gateway.py`: generation decoration and
  session lifecycle wiring over Uni-Agent's existing Gateway. Reuse the latest
  scored prefix for each conversation chain; crop at exact token boundaries and
  attach its labels. No second end-of-episode Teacher pass or vendor-tree edit.
- `opd_framework.py`: inject training-only session options, enqueue existing
  labels, and wait for batch completion when requested.

`ADAPTIVE_TURN` logs contain session, turn, entropy, `top16_mass` and stopped.
The covered mass is recorded to check whether renormalization would make a
material difference; it does not affect this version's decision. Lightweight
profiling adds `adaptive_teacher_turn` spans. Teacher prefill still processes the
growing context, and each episode waits for its per-turn score. Actual speedup
must therefore be measured against the saved tail, not assumed from fewer turns.

Adaptive uses the same formal defaults as Vanilla and TCOD:

```bash
OPD_ALGORITHM=adaptive ADAPTIVE_THRESHOLD=0.1 \
bash scripts/run_r2e_opd_train.sh
```

The normal launch preflight, frozen R2E manifest, dedicated Ray head and Git
parity requirements still apply. GPU correctness/performance validation remains
required before treating Adaptive as a measured training result.

## Shared training / Verified evaluation protocol

`configs/coding_react.yaml` is the single agent-facing configuration for R2E-Gym
training and SWE-bench Verified evaluation. YAML anchors share the prompt, agent,
tools and sandbox settings. The shared solver uses the upstream prompt, stateful
shell, editor and submit tools, complete history, temperature 0.8, top-p 0.9,
up to 100 turns, a 120-second shell timeout and native repetition detection.
Training uses a 32,768-token total-context prefix. Evaluation may set a different
explicit context budget while preserving the same prompt, tools and history
semantics. TCOD/Adaptive apply their training horizon decisions; evaluation uses
the full configured horizon.

Native vLLM repetition detection stops 1–64-token patterns repeated 16 times
consecutively. This is an initial conservative heuristic, not a measured optimum
or a detector of repeated tool interactions. The engine stops generation online;
the ReAct loop ends the episode before executing any tools in the terminal reply.
The retained prefix is scored normally; no trajectory filtering or new loss mask
is added. Context and per-generation limits are also enforced before engine calls,
with the original `length` reason preserved through the gateway.

`scripts/run_external_eval.sh swebench_verified` uses that same solver and
`src/coding_opd/swe_bench_task.py` collects its patch, destroys the agent container,
and applies only the patch in a new clean verifier using the official SWE-bench
APIs. Verification gets a separate 1,800-second budget. Use `EVAL_TIER=quick` for
Verified-64. Results carry protocol `shared-react-v1` and the configuration hash;
old Codex scores belong to a different protocol. Codex registrations reuse the
same grader, preserving the old optional evaluation entry point.

The formal launcher uses 2 Actor + 4 rollout + 2 Teacher GPUs, gradient checkpointing,
a 32,768-token Actor budget, and checkpoints at steps 16, 64, 128, 192 and 256.
