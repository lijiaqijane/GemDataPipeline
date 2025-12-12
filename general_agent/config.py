from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM configuration supporting both self-hosted vLLM and OpenAI-compatible APIs."""

    provider: str
    base_url: str
    model: str
    api_key: str | None
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables, supporting vLLM, OpenAI, and Deepseek."""
        provider = os.getenv("LLM_PROVIDER", "vllm").lower()
        if provider not in {"vllm", "openai", "volcano", "deepseek"}:
            provider = "vllm"

        if provider == "vllm":
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            model = os.getenv("VLLM_MODEL", "local-model")
            api_key = os.getenv("VLLM_API_KEY")
        elif provider in {"volcano", "deepseek"}:
            # Volcano Engine Deepseek v3.2
            base_url = os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            model = os.getenv("VOLCANO_MODEL", "deepseek-v3.2")
            api_key = os.getenv("VOLCANO_API_KEY")
        else:  # openai
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")

        timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        return cls(provider=provider, base_url=base_url, model=model, api_key=api_key, timeout=timeout)
