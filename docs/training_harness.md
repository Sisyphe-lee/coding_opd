# Training and evaluation harness

R2E training and SWE-bench Verified ReAct evaluation use
`configs/coding_react.yaml`. They share the upstream Uni-Agent prompt,
`str_replace_editor`, `stateful_shell`, `submit`, complete conversation history,
temperature 0.8 and top-p 0.9. The project wrapper only handles terminal length
and repetition reasons needed by training.

The stateful shell keeps cwd, exports, functions and activated environments across
tool calls. Every shell command has a 120-second timeout. Native vLLM repetition
detection stops 1–64-token patterns repeated 16 times before another tool is
executed. Both settings apply consistently to Vanilla, TCOD and Adaptive.

## Length and update semantics

Training uses a 32,768-token total-context prefix and at most 100 agent turns.
The full prefix, including tool observations, stays in one Actor training sample;
only assistant tokens receive the K3 loss. The Actor uses gradient checkpointing
and a 32,768-token packed-microbatch budget.

Evaluation may use a larger explicit context budget, commonly 65,536 or 262,144
tokens. Matching the prompt, tools and history semantics does not require equal
training and evaluation lengths. For sampled-token OPD, Teacher labels at a prefix
position depend only on the preceding prefix, so generating a tail that the Actor
will discard adds no supervision to that prefix. Arbitrarily training later chunks
without their full preceding context would change the conditioning and is not used.

Formal training uses batch=minibatch=32 and one optimizer update and weight sync per
batch. Vanilla and TCOD prefetch one checkpoint-safe batch; steady-state measurement
found trajectory staleness mean/max 1 and trajectory span max 1. Adaptive keeps a
complete batch barrier and disables prefetch. A training step is one optimizer update.

## Scoring isolation

Training does not execute task grading or mix task reward into the baseline K3
loss. SWE-bench evaluation destroys the agent sandbox, transfers only the patch,
and runs the official grader in a fresh network-isolated container. Hidden tests
are therefore unavailable to the agent. Verified and DeepSWE data never enter
training, Teacher supervision or reward shaping.

The 32K choice is supported by the B300 Actor check in
`docs/performance_optimization.md`: compared with 20K without checkpointing, 32K
with checkpointing reduced peak allocated memory from 216.00 to 69.96 GiB per GPU
and cost 23.2% more time per token. This joint comparison does not isolate the
checkpointing overhead.
