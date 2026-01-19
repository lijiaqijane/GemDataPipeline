"""
Model adapter for code_agent module.

This module provides model interface adaptation, using agent_gem.llm.LLMClient
instead of app.model.common for better module independence.
"""

from __future__ import annotations

import os
import threading
import logging
from dataclasses import dataclass
from typing import Literal, Any

try:
    from agent_gem.llm import LLMClient
    from agent_gem.config import LLMConfig
    LLM_CLIENT_AVAILABLE = True
except ImportError:
    LLM_CLIENT_AVAILABLE = False
    LLMClient = None
    LLMConfig = None


logger = logging.getLogger(__name__)

# Thread-local storage for cost tracking
thread_cost = threading.local()
thread_cost.process_cost = 0.0
thread_cost.process_input_tokens = 0
thread_cost.process_output_tokens = 0

# Global ModelAdapter instance
_global_model_adapter: "ModelAdapter | None" = None


@dataclass
class ModelConfig:
    """Configuration for model usage."""
    
    model_name: str | None = None  # If None, uses LLMConfig.from_env()
    temperature: float = 0.0
    max_tokens: int = 8192
    response_format: Literal["text", "json_object"] = "text"
    # Prices are in USD per 1K tokens
    input_price_per_1m_tokens: float = 2.0
    output_price_per_1m_tokens: float = 3.0


class ModelAdapter:
    """
    Adapter for model interface using LLMClient.
    
    This class provides a unified interface for model calls using agent_gem.llm.LLMClient.
    Falls back to app.model.common if LLMClient is not available.
    """
    
    def __init__(self, config: ModelConfig | None = None, llm_client: LLMClient | None = None):
        """
        Initialize the model adapter.
        
        Args:
            config: Model configuration. If None, uses default config.
            llm_client: Optional LLMClient instance. If None, creates from env.
        """
        self.config = config or ModelConfig()
        self._llm_client = llm_client
        self._app_model = None
        self._initialized = False
        self._use_llm_client = LLM_CLIENT_AVAILABLE
        # Per-agent configuration: agent_name -> {temperature, max_tokens}
        self._agent_configs: dict[str, dict[str, float | int]] = {}
    
    def _initialize_model(self):
        """Initialize the model from LLMClient or app.model.common."""
        if self._initialized:
            return
        
        try:
            if self._llm_client is None:
                self._llm_client = LLMClient.from_env()
            logger.info(f"Initialized LLMClient with model: {self._llm_client.config.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLMClient: {e}, falling back to app.model.common")
            self._use_llm_client = False
        
        self._initialized = True
    
    def set_agent_config(self, agent_name: str, temperature: float | None = None, max_tokens: int | None = None):
        """
        Set configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., "context_retrieval_agent")
            temperature: Temperature for this agent (None = use default)
            max_tokens: Max tokens for this agent (None = use default)
        """
        if agent_name not in self._agent_configs:
            self._agent_configs[agent_name] = {}
        if temperature is not None:
            self._agent_configs[agent_name]["temperature"] = temperature
        if max_tokens is not None:
            self._agent_configs[agent_name]["max_tokens"] = max_tokens
        logger.debug(f"Set config for agent {agent_name}: temperature={temperature}, max_tokens={max_tokens}")
    
    def call(
        self,
        messages: list[dict],
        tools: Any = None,
        response_format: Literal["text", "json_object"] | None = None,
        agent_name: str | None = None,
        **kwargs,
    ) -> tuple[str, float, int, int]:
        """
        Call the model with messages.
        
        Args:
            messages: List of message dictionaries
            tools: Optional tools for function calling (not supported by LLMClient yet)
            response_format: Response format ("text" or "json_object")
            agent_name: Optional agent name to use agent-specific config
            **kwargs: Additional arguments (temperature, max_tokens can override agent config)
        
        Returns:
            Tuple of (content, cost, input_tokens, output_tokens)
        """
        if not self._initialized:
            self._initialize_model()
        
        response_format = response_format or self.config.response_format
        
        # Get temperature: kwargs > agent config > default config
        if "temperature" in kwargs:
            temperature = kwargs["temperature"]
        elif agent_name and agent_name in self._agent_configs and "temperature" in self._agent_configs[agent_name]:
            temperature = self._agent_configs[agent_name]["temperature"]
        else:
            temperature = self.config.temperature
        
        # Get max_tokens: kwargs > agent config > default config
        if "max_tokens" in kwargs:
            max_tokens = kwargs["max_tokens"]
        elif agent_name and agent_name in self._agent_configs and "max_tokens" in self._agent_configs[agent_name]:
            max_tokens = self._agent_configs[agent_name]["max_tokens"]
        else:
            max_tokens = self.config.max_tokens
        
        if self._use_llm_client and self._llm_client:
            # Use LLMClient
            try:
                # Handle response_format for JSON mode
                if response_format == "json_object":
                    # For JSON mode, we need to use the underlying OpenAI client
                    # LLMClient's chat_completion doesn't support response_format directly
                    response = self._llm_client._client.chat.completions.create(
                        model=self._llm_client.config.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content
                    if content is None:
                        raise RuntimeError(f"Unexpected LLM response: {response}")
                    
                    # Get tokens from response if available
                    if hasattr(response, 'usage') and response.usage:
                        input_tokens = response.usage.prompt_tokens or 0
                        output_tokens = response.usage.completion_tokens or 0
                    else:
                        # Estimate tokens (rough estimate: ~4 chars per token)
                        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
                        input_tokens = total_chars // 4
                        output_tokens = len(content) // 4
                else:
                    # Use standard chat_completion
                    content = self._llm_client.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    # Estimate tokens (rough estimate: ~4 chars per token)
                    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
                    input_tokens = total_chars // 4
                    output_tokens = len(content) // 4
                
                # Cost calculation based on configurable per-1K-token prices
                input_price = getattr(self.config, "input_price_per_1m_tokens", 0.0)
                output_price = getattr(self.config, "output_price_per_1m_tokens", 0.0)
                cost = (
                    (input_tokens / 1000000.0) * input_price
                    + (output_tokens / 1000000.0) * output_price
                )
                
                # Update thread-local stats
                thread_cost.process_input_tokens += input_tokens
                thread_cost.process_output_tokens += output_tokens
                thread_cost.process_cost += cost
                
                return content, cost, input_tokens, output_tokens
                
            except Exception as e:
                logger.error(f"LLMClient call failed: {e}")
                raise
        else:
            raise RuntimeError("Model not initialized")
    
    def get_overall_exec_stats(self) -> dict:
        """
        Get overall execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self._initialized:
            self._initialize_model()
        
        if self._use_llm_client and self._llm_client:
            model_name = self._llm_client.config.model
        elif self._app_model:
            return self._app_model.get_overall_exec_stats()
        else:
            model_name = self.config.model_name or "unknown"
        
        return {
            "model": model_name,
            "total_input_tokens": thread_cost.process_input_tokens,
            "total_output_tokens": thread_cost.process_output_tokens,
            "total_tokens": (
                thread_cost.process_input_tokens + thread_cost.process_output_tokens
            ),
            "total_cost": thread_cost.process_cost,
        }
    
    
    def set_max_tokens(self, max_tokens: int):
        """Set the maximum tokens."""
        self.config.max_tokens = max_tokens


def get_model_adapter() -> ModelAdapter:
    """Get or create the global ModelAdapter instance."""
    global _global_model_adapter
    if _global_model_adapter is None:
        _global_model_adapter = ModelAdapter()
    return _global_model_adapter


def set_model_adapter(adapter: ModelAdapter):
    """Set the global ModelAdapter instance."""
    global _global_model_adapter
    _global_model_adapter = adapter
