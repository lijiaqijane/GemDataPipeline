from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import requests

from .config import LLMConfig


class LLMClient:
    """Unified LLM client that supports vLLM and OpenAI-compatible endpoints."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(LLMConfig.from_env())

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Call chat completion and return the generated text."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        url = f"{self.config.base_url}/chat/completions"
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.config.timeout)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive path
            raise RuntimeError(f"Unexpected LLM response format: {data}") from exc

    def simple_complete(self, prompt: str, **kwargs: Any) -> str:
        """Convert plain prompt into chat messages and call chat completion."""
        messages = [
            {"role": "system", "content": "You are a tool synthesis and task generation assistant."},
            {"role": "user", "content": prompt},
        ]
        return self.chat_completion(messages, **kwargs)

