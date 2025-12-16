from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-based client for sync and async chat completions."""

    def __init__(
        self,
        config: LLMConfig,
        client: Optional[OpenAI] = None,
        async_client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.config = config
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self._aclient = async_client or AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        logger.debug(
            "LLM client initialized (provider=%s, model=%s, base_url=%s, timeout=%s, retries=%s)",
            config.provider,
            config.model,
            config.base_url,
            config.timeout,
            config.max_retries,
        )

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(LLMConfig.from_env())

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> str:
        logger.info(
            "Requesting chat completion (model=%s, messages=%d)",
            self.config.model,
            len(messages),
        )
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Unexpected LLM response: {response}")
        logger.debug("LLM response preview: %s", _preview_text(content))
        return content

    async def achat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> str:
        logger.info(
            "Requesting async chat completion (model=%s, messages=%d)",
            self.config.model,
            len(messages),
        )
        response = await self._aclient.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Unexpected LLM response: {response}")
        logger.debug("LLM async response preview: %s", _preview_text(content))
        return content

    def simple_complete(self, prompt: str, **kwargs: Any) -> str:
        return self.chat_completion([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()

    async def aclose(self) -> None:
        self.close()
        if hasattr(self._aclient, "close"):
            await self._aclient.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


def _preview_text(text: str, limit: int = 320) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}... (truncated)"
