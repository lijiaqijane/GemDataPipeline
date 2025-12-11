from __future__ import annotations

from dataclasses import dataclass
import os


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
        """Load configuration from environment variables, preferring vLLM."""
        provider = os.getenv("LLM_PROVIDER", "vllm").lower()
        if provider not in {"vllm", "openai"}:
            provider = "vllm"

        if provider == "vllm":
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            model = os.getenv("VLLM_MODEL", "local-model")
            api_key = os.getenv("VLLM_API_KEY")
        else:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")

        timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        return cls(provider=provider, base_url=base_url, model=model, api_key=api_key, timeout=timeout)

