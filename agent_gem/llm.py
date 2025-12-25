from __future__ import annotations

import json
import logging
from pathlib import Path
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
        self._log_io = config.log_io
        self._log_path = Path(config.log_io_file) if config.log_io_file else None
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
        self._log_io_payload(
            "prompt",
            {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
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
        self._log_io_payload("response", {"model": self.config.model, "content": content})
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
        self._log_io_payload(
            "prompt_async",
            {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
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
        self._log_io_payload("response_async", {"model": self.config.model, "content": content})
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

    def _log_io_payload(self, kind: str, payload: dict[str, Any]) -> None:
        """Optionally log full LLM I/O for debugging; controlled via env."""
        if not self._log_io:
            return
        try:
            # Log full input/output without truncation when enabled
            text = json.dumps(payload, ensure_ascii=False)
        except Exception:
            try:
                text = str(payload)
            except Exception:
                text = "<unserializable payload>"
        if self._log_path:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(f"[{kind}] {text}\n")
            except Exception:
                logger.debug("Failed to write LLM IO log", exc_info=True)

    def chat_with_agent(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, str]],
        tool_call_map: Dict[str, str],
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_sub_turns: int = 20,
    ) -> str:
        """Call chat completion with thinking and tools."""
        search_content = []
        sub_turn = 1
        while True:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "thinking": {
                        "type": "enabled",
                    }
                },
            )
            messages.append(response.choices[0].message)
            reasoning_content = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content
            tool_calls = response.choices[0].message.tool_calls

            print(f"Turn {sub_turn}\n{reasoning_content=}\n{content=}\n{tool_calls=}")

            if tool_calls is None or sub_turn >= max_sub_turns:
                break
            for tool in tool_calls:
                tool_function = tool_call_map[tool.function.name]
                tool_result = tool_function(**json.loads(tool.function.arguments))
                if tool.function.name == "search":
                    formatted_tool_result = _format_tool_result(tool_result)
                else:
                    formatted_tool_result = tool_result
                search_content.append(formatted_tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "content": formatted_tool_result,
                    }
                )
            sub_turn += 1

        return content, search_content


def _preview_text(text: str, limit: int = 320) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}... (truncated)"


def _format_tool_result(tool_result: Any) -> str:
    formatted_text = "Here are the search results/documents:\n\n"

    for index, item in enumerate(tool_result, 1):
        title = item.get("title", "No Title")
        url = item.get("url", "No URL")
        summary = item.get("summary", "No Summary")

        entry = (
            f"Source [{index}]:\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Summary: {summary}\n"
            f"----------------------------------------\n"
        )
        formatted_text += entry

    return formatted_text
