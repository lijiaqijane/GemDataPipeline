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
        """Load configuration from environment variables, supporting vLLM, OpenAI, and Deepseek."""
        # 默认使用 Deepseek/Volcano，便于开箱即用
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        if provider not in {"vllm", "openai", "volcano", "deepseek"}:
            provider = "vllm"

        if provider == "vllm":
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            model = os.getenv("VLLM_MODEL", "local-model")
            api_key = os.getenv("VLLM_API_KEY")
        elif provider in {"volcano", "deepseek"}:
            # Volcano Engine Deepseek v3.2
            base_url = os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            # 通过 /models 实测可用的 Deepseek 模型（火山提供）
            model = os.getenv("VOLCANO_MODEL", "deepseek-v3-2-251201")
            # 默认内置示例 Key，实际使用请通过环境变量覆盖
            api_key = os.getenv("VOLCANO_API_KEY", "47041ffc-3c83-49ee-9d79-4f70592850d2")
        else:  # openai
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")

        timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        return cls(provider=provider, base_url=base_url, model=model, api_key=api_key, timeout=timeout)
