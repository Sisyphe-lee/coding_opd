import asyncio
from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest
from coding_opd.opd_gateway import OPDGatewayActor
from coding_opd.adaptive_rollout import AdaptiveRolloutClient, AdaptiveSession
from coding_opd.adaptive_teacher import extract_adaptive_logprobs
from coding_opd.algorithms import get_algorithm
from coding_opd.opd_framework import OPDAgentFrameworkRolloutAdapter, OPDGatewayAgentFramework
from uni_agent.framework.framework import GatewayAgentFramework
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.session import Trajectory
from verl.workers.rollout.replica import TokenOutput


def test_original_frontier_and_online_prefix_agree():
    algorithm = get_algorithm("adaptive")
    # Three-turn average drift first exceeds 0.1 at the sixth turn. The
    # triggering turn is excluded, leaving five turns for training.
    entropy = [0.2, 0.2, 0.2, 0.29, 0.29, 0.5, 0.0]
    assert [algorithm.retained_turns(entropy[:n]) for n in range(1, 8)] == [None] * 5 + [5, 5]
    assert algorithm.retained_turns([0.2, 0.2, 0.2, 0.6]) == 3
    assert algorithm.retained_turns([0.2] * 24) is None
    assert algorithm.retained_turns([math.nan, 0.2, 0.2, 0.9]) is None
    assert algorithm.rollout_max_turns(128, 24) == 24


def test_top16_partial_entropy_keeps_outside_sample_and_uses_response_suffix():
    # Actual Student token is rank 17, outside Teacher Top-16.
    row = {i: SimpleNamespace(logprob=math.log(0.04), rank=i + 1) for i in range(16)}
    row[99] = SimpleNamespace(logprob=math.log(0.001), rank=17)
    prompt_row = {7: SimpleNamespace(logprob=0.0, rank=1)}
    output = SimpleNamespace(prompt_token_ids=[8, 7, 99], prompt_logprobs=[None, prompt_row, row])
    result = {}
    extract_adaptive_logprobs(output, start=1, result=result)
    assert result["prompt_ids"] == [[7], [99], [0]]
    assert result["prompt_logprobs"] == [[0.0], [math.log(0.001)], [0.0]]
    assert result["opd_top16_mass"] == pytest.approx(0.64)
    assert result["opd_entropy"] == pytest.approx(-0.64 * math.log(0.04))
    assert result["opd_entropy"] != pytest.approx(math.log(16))


class CharacterTokenizer:
    eos_token_id = ord("\n")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = "".join(f"{m['role']}:{m.get('content', '')}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return self.encode(text) if tokenize else text

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class Student:
    def __init__(self):
        self.calls = []

    async def generate(self, request_id, **kwargs):
        self.calls.append(request_id)
        return TokenOutput(token_ids=list(map(ord, "answer")), stop_reason="completed",
                           extra_fields={"min_global_steps": 9, "max_global_steps": 9})


class Teacher:
    def __init__(self, entropies):
        self.entropies = iter(entropies)
        self.calls = []

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        self.calls.append((request_id, list(prompt_ids), dict(sampling_params)))
        await asyncio.sleep(0)
        return TokenOutput(token_ids=[0], extra_fields={
            "prompt_logprobs": [[-i / 100] for i in range(len(prompt_ids) - 1)] + [[0.0]],
            "opd_entropy": next(self.entropies), "opd_top16_mass": 0.96,
        })


def make_backend(entropies):
    student, teacher = Student(), Teacher(entropies)
    manager = SimpleNamespace(_resolve_teacher_key=lambda _: "teacher", teacher_client={"teacher": teacher})
    return AdaptiveRolloutClient(student, manager), student, teacher


def test_real_gateway_stops_online_and_reuses_aligned_teacher_labels(monkeypatch):
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    backend, student, teacher = make_backend([0.2] * 3 + [0.29] * 2 + [0.5])
    actor = OPDGatewayActor(GatewayActorConfig(tokenizer=CharacterTokenizer()), backend)
    actor._server_base_url = "http://test"

    async def run():
        await actor.create_session("train", sampling_params={
            "_opd_adaptive": {"threshold": 0.1, "routing_key": None},
        })
        messages = [{"role": "user", "content": "solve"}]
        tools_executed = 0
        for _ in range(24):
            response = await actor._handle_openai_chat_completions("train", {"messages": messages})
            choice = json.loads(response.body)["choices"][0]
            if choice["finish_reason"] == "length":
                break
            tools_executed += 1
            messages.extend([choice["message"], {"role": "user", "content": "continue"}])
        state = backend.sessions["train"]
        assert state.stopped and len(state.entropies) == 6
        trajectories = await actor.finalize_session("train")
        assert "train" not in backend.sessions
        assert len(student.calls) == len(teacher.calls) == 6
        assert tools_executed == 5
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        tokens = trajectory.prompt_ids + trajectory.response_ids
        assert tokens == teacher.calls[4][1]  # exactly the fifth, retained prefix
        assert sum(trajectory.response_mask) == 5 * len("answer")
        expected = [-i / 100 for i in range(len(tokens) - 1)] + [0.0]
        assert trajectory.extra_fields["teacher_logprobs"].squeeze().tolist() == pytest.approx(expected)
        assert trajectory.extra_fields["teacher_ids"].squeeze().tolist() == tokens[1:] + [0]
        assert trajectory.extra_fields["min_global_steps"] == trajectory.extra_fields["max_global_steps"] == 9
        # The framework must enqueue these labels without a second Teacher call.
        framework = object.__new__(OPDGatewayAgentFramework)
        framework._distillation_enabled = True

        async def no_rescore(*args):
            pytest.fail("online labels were rescored")

        async def enqueue(self, **kwargs):
            assert kwargs["trajectories"] == trajectories

        monkeypatch.setattr(framework, "_attach_teacher_logprobs", no_rescore)
        monkeypatch.setattr(GatewayAgentFramework, "_write_session_trajectories_to_tq", enqueue)
        await framework._write_session_trajectories_to_tq(
            uid="x", session_index=0, trajectories=trajectories, sample_fields={}, global_steps=9, partition_id="train",
        )
        # Validation on the same actor bypasses Adaptive and Teacher entirely.
        await actor.create_session("val")
        await actor._handle_openai_chat_completions("val", {"messages": [{"role": "user", "content": "solve"}]})
        val = await actor.finalize_session("val")
        assert len(teacher.calls) == 6
        assert "teacher_logprobs" not in val[0].extra_fields

    asyncio.run(run())


def test_scored_prefixes_cover_fragmentation_and_rewrites():
    state = AdaptiveSession(get_algorithm("adaptive"))
    def accept(tokens):
        state.accept(tokens, {"opd_entropy": 0.2, "opd_top16_mass": 0.9,
                              "prompt_logprobs": [[-t / 10] for t in tokens[1:]] + [[0.0]]})
    accept([1, 2, 3, 4])
    accept([1, 2, 3, 4, 5, 6])
    assert len(state.prefixes) == 1
    accept([1, 2, 7, 8])
    trajectories = [Trajectory([1, 2], [3, 4], [1, 1]),
                    Trajectory([1, 2, 3, 4, 5], [6], [1]),
                    Trajectory([1, 2], [7, 8], [1, 1])]
    for trajectory in state.attach(trajectories):
        tokens = trajectory.prompt_ids + trajectory.response_ids
        assert trajectory.extra_fields["teacher_logprobs"].squeeze().tolist() == pytest.approx(
            [-t / 10 for t in tokens[1:]] + [0.0])


def test_batch_barrier_waits_only_when_requested(monkeypatch):
    events = []
    adapter = OPDAgentFrameworkRolloutAdapter()
    adapter.framework_worker = SimpleNamespace(generate_sequences=SimpleNamespace(
        remote=lambda _: events.append("submit") or "batch-future"))
    monkeypatch.setattr("coding_opd.opd_framework.ray.get", lambda future: events.append("all-rollouts-finished"))
    adapter.generate_sequences("batch")
    assert events == ["submit"]
    events.clear()
    adapter.synchronous_rollouts = True
    adapter.generate_sequences("batch")
    events.append("actor-update")
    assert events == ["submit", "all-rollouts-finished", "actor-update"]


@pytest.mark.parametrize("training", [True, False])
def test_framework_injects_adaptive_only_into_training(monkeypatch, training):
    framework = object.__new__(OPDGatewayAgentFramework)
    framework._teacher_key = "data_source"
    kwargs = dict(runner_config=SimpleNamespace(runner_kwargs={"opd_algorithm": "adaptive"}),
                  sample_fields={"data_source": "r2e", "tools_kwargs": {"_opd_progress": {"training": training}}},
                  sampling_params={"temperature": 0.6})
    original = deepcopy(kwargs)
    async def parent(self, **received):
        assert ("_opd_adaptive" in received["sampling_params"]) is training
        if training:
            assert received["sampling_params"]["_opd_adaptive"] == {"threshold": 0.1, "routing_key": "r2e"}
    monkeypatch.setattr(GatewayAgentFramework, "_run_agent_episode", parent)
    asyncio.run(framework._run_agent_episode(**kwargs))
    assert kwargs == original


def test_teacher_server_suffix_is_request_local(monkeypatch):
    # Exercise the real server extension against a CPU stand-in for the heavy
    # vLLM server. In particular, overlap requests with different suffix starts.
    import importlib
    import sys
    import types

    module_name = "verl.workers.rollout.vllm_rollout.vllm_async_server"
    upstream = types.ModuleType(module_name)
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.RepetitionDetectionParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    calls = []

    class Server:
        def __init__(self):
            self.config = SimpleNamespace(custom=None)  # Teacher's actual default.

        async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
            assert "opd_entropy_start" not in sampling_params
            calls.append(dict(sampling_params))
            await asyncio.sleep(0)
            rows = [None] + [{t: SimpleNamespace(logprob=-float(t), rank=1)} for t in prompt_ids[1:]]
            result = {}
            upstream.extract_prompt_logprobs(
                SimpleNamespace(prompt_token_ids=prompt_ids, prompt_logprobs=rows,
                                outputs=[SimpleNamespace(finish_reason=request_id if request_id in
                                                         ("repetition", "length") else "stop")]), 16, result,
            )
            return TokenOutput(token_ids=[0], extra_fields=result)

    upstream.vLLMHttpServer = Server
    upstream.vLLMReplica = object
    upstream.extract_prompt_logprobs = lambda output, num_prompt_logprobs, result_dict: result_dict.update(ordinary=True)
    monkeypatch.setitem(sys.modules, module_name, upstream)
    # Import through the package exactly as Ray does when resolving the class.
    package = types.ModuleType("verl.workers.rollout.vllm_rollout")
    package.vllm_async_server = upstream
    monkeypatch.setitem(sys.modules, package.__name__, package)
    sys.modules.pop("coding_opd.vllm_server", None)
    try:
        extension = importlib.import_module("coding_opd.vllm_server")
        server = extension.OPDServer()

        async def run():
            results = await asyncio.gather(
                server.generate([0, 1, 2], {"opd_entropy_start": 0}, "a"),
                server.generate([0, 1, 2], {"opd_entropy_start": 1}, "b"),
                server.generate([0, 1, 2], {}, "ordinary"),
            )
            assert results[0].extra_fields["opd_entropy"] == pytest.approx((math.exp(-1) + 2 * math.exp(-2)) / 2)
            assert results[1].extra_fields["opd_entropy"] == pytest.approx(2 * math.exp(-2))
            assert results[2].extra_fields == {"ordinary": True}
            server.config.custom = {"coding_react": {
                "max_context_tokens": 10, "max_tokens_per_turn": 4,
                "repetition_detection": {"min_pattern_size": 1, "max_pattern_size": 64, "min_count": 16},
            }}
            for reason in ("repetition", "length"):
                output = await server.generate(list(range(8)), {"max_tokens": 100}, reason)
                assert output.stop_reason == reason
                assert calls[-1]["max_tokens"] == 2
                assert calls[-1]["repetition_detection"].min_count == 16
            server.global_steps = 9
            count = len(calls)
            output = await server.generate(list(range(10)), {}, "full")
            assert output.stop_reason == "length" and output.token_ids == []
            assert len(calls) == count  # No engine request when observations fill the context.
        asyncio.run(run())
    finally:
        sys.modules.pop("coding_opd.vllm_server", None)
