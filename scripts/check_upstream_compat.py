#!/usr/bin/env python3
"""Check the Uni-Agent extension contract against the selected veRL source."""

from __future__ import annotations

import inspect
import json

from uni_agent.framework.entry import AgentFrameworkRolloutAdapter


def main() -> None:
    create_signature = inspect.signature(AgentFrameworkRolloutAdapter.create)
    parameters = create_signature.parameters
    result = {
        "adapter_create_signature": str(create_signature),
        "accepts_llm_client": "llm_client" in parameters,
        "accepts_teacher_client": "teacher_client" in parameters,
        "teacher_client_currently_rejected": "does not support teacher_client"
        in inspect.getsource(AgentFrameworkRolloutAdapter.create),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if not result["accepts_teacher_client"]:
        raise SystemExit("Uni-Agent adapter no longer exposes teacher_client")


if __name__ == "__main__":
    main()
