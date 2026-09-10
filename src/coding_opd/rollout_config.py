"""Bind the shared ReAct policy to training and evaluation inference servers."""

from omegaconf import OmegaConf
from uni_agent.tasks.config import TaskConfigResolver


def configure_coding_rollout(config, task_config_path=None, task_name="r2e_gym"):
    if task_config_path is None:
        runner = config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs
        task_config_path = runner.task_config_path
    entries = TaskConfigResolver.from_file(str(task_config_path)).defaults_by_name
    entry = entries.get(task_name, next(iter(entries.values())))
    agent = entry["agent"]
    model = agent["model"]
    policy = {
        "max_context_tokens": model["max_total_tokens"],
        "max_tokens_per_turn": model.get("max_tokens_per_turn", model["max_total_tokens"]),
        "repetition_detection": agent.get("repetition_detection", {}),
    }
    OmegaConf.update(config, "actor_rollout_ref.rollout.custom.coding_react", policy, force_add=True)

    from verl.workers.rollout.replica import RolloutReplicaRegistry

    def load():
        from coding_opd.vllm_server import OPDReplica
        return OPDReplica

    RolloutReplicaRegistry.register("vllm", load)
