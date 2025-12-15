from __future__ import annotations

import logging

from agent_gem.generator import EnvironmentGenerator, GenerationRequest


def test_environment_generator_start_log_formats(tmp_path, caplog) -> None:
    generator = EnvironmentGenerator(llm=object(), taskdb=tmp_path)
    request = GenerationRequest(
        agent_type="general_agent",
        topic="testing",
        num=0,
        difficulty=1,
    )

    with caplog.at_level(logging.INFO, logger="agent_gem.generator"):
        generator.generate(request)

    record = next(
        (
            record
            for record in caplog.records
            if record.name == "agent_gem.generator" and record.msg.startswith("Starting generation:")
        ),
        None,
    )
    assert record is not None
    assert str(tmp_path) in record.getMessage()
