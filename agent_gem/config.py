from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str | None
    timeout: float = 120.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        if provider not in {"deepseek", "volcano", "openai", "vllm"}:
            raise RuntimeError(f"Unknown LLM provider {provider}")

        # Deepseek via Volcano Ark (OpenAI-compatible endpoint)
        # We treat both "deepseek" and "volcano" providers as using VOLCANO_* env vars,
        # and allow DEEPSEEK_* overrides for flexibility.
        if provider in {"deepseek", "volcano"}:
            base_url = os.getenv(
                "VOLCANO_BASE_URL",
                "https://ark.cn-beijing.volces.com/api/v3",
            )
            model = os.getenv("VOLCANO_MODEL", "deepseek-v3-2-251201")
            api_key = os.getenv("VOLCANO_API_KEY")

            # Optional overrides using DEEPSEEK_* if provided
            base_url = os.getenv("DEEPSEEK_BASE_URL", base_url)
            model = os.getenv("DEEPSEEK_MODEL", model)
            api_key = (
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("DEEPSEEK_API")
                or api_key
            )
        elif provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            model = os.getenv("VLLM_MODEL", "local-model")
            api_key = os.getenv("VLLM_API_KEY")

        timeout = float(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        return cls(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
