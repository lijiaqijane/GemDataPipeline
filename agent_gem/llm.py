from __future__ import annotations

import json
import random
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI, APIError, RateLimitError, APITimeoutError

from .config import LLMConfig

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        timeout: float = 120.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.timeout = timeout

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on the exception type and attempt count."""
        if attempt >= self.max_retries:
            return False

        # Always retry on rate limits
        if isinstance(exception, RateLimitError):
            return True

        # Retry on timeout errors
        if isinstance(exception, APITimeoutError):
            return True

        # Retry on specific HTTP status codes
        if isinstance(exception, APIError):
            status_code = getattr(exception, 'status_code', None)
            # Retry on server errors (5xx) and some client errors (429 is rate limit, handled above)
            if status_code and 500 <= status_code < 600:
                return True
            # Retry on 429 (rate limit) - though RateLimitError should catch this
            if status_code == 429:
                return True

        return False

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt using exponential backoff with optional jitter."""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)

        if self.jitter:
            # Add random jitter up to 25% of the delay
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay


def retry_with_backoff(retry_config: RetryConfig):
    """Decorator to add exponential backoff retry logic to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not retry_config.should_retry(e, attempt):
                        logger.warning(
                            "LLM request failed after %d attempts: %s",
                            attempt + 1,
                            str(e)
                        )
                        raise e

                    delay = retry_config.get_delay(attempt)
                    logger.info(
                        "LLM request failed (attempt %d/%d): %s. Retrying in %.2f seconds...",
                        attempt + 1,
                        retry_config.max_retries + 1,
                        str(e),
                        delay
                    )
                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception
        return wrapper
    return decorator


class LLMClient:
    """OpenAI-based client for sync and async chat completions."""

    def __init__(
        self,
        config: LLMConfig,
        client: Optional[OpenAI] = None,
        async_client: Optional[AsyncOpenAI] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        self.config = config

        # Create retry config with improved defaults from config
        self.retry_config = retry_config or RetryConfig(
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            backoff_factor=config.retry_backoff_factor,
            jitter=config.retry_jitter,
            timeout=config.timeout,
        )

        # Initialize clients with minimal retries (let our custom logic handle it)
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=self.retry_config.timeout,
            max_retries=0,  # Disable OpenAI's built-in retries
        )
        self._aclient = async_client or AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=self.retry_config.timeout,
            max_retries=0,  # Disable OpenAI's built-in retries
        )
        self._log_io = config.log_io
        self._log_path = Path(config.log_io_file) if config.log_io_file else None
        logger.debug(
            "LLM client initialized (provider=%s, model=%s, base_url=%s, timeout=%s, custom_retries=%s)",
            config.provider,
            config.model,
            config.base_url,
            self.retry_config.timeout,
            self.retry_config.max_retries,
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
        self._log_io_payload("prompt", {"model": self.config.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})

        last_exception = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
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

            except Exception as e:
                last_exception = e
                if not self.retry_config.should_retry(e, attempt):
                    logger.warning(
                        "LLM request failed after %d attempts: %s",
                        attempt + 1,
                        str(e)
                    )
                    raise e

                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.info(
                        "LLM request failed (attempt %d/%d): %s. Retrying in %.2f seconds...",
                        attempt + 1,
                        self.retry_config.max_retries + 1,
                        str(e),
                        delay
                    )
                    time.sleep(delay)

        # This should never be reached, but just in case
        raise last_exception

    async def achat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> str:
        import asyncio

        logger.info(
            "Requesting async chat completion (model=%s, messages=%d)",
            self.config.model,
            len(messages),
        )
        self._log_io_payload("prompt_async", {"model": self.config.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})

        last_exception = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
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

            except Exception as e:
                last_exception = e
                if not self.retry_config.should_retry(e, attempt):
                    logger.warning(
                        "LLM async request failed after %d attempts: %s",
                        attempt + 1,
                        str(e)
                    )
                    raise e

                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.info(
                        "LLM async request failed (attempt %d/%d): %s. Retrying in %.2f seconds...",
                        attempt + 1,
                        self.retry_config.max_retries + 1,
                        str(e),
                        delay
                    )
                    await asyncio.sleep(delay)

        # This should never be reached, but just in case
        raise last_exception

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

            if tool_calls is None or sub_turn >= 30:
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
