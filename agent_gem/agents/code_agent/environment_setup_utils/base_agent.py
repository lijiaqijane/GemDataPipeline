"""
Base Agent class for code_agent module.

This module provides the abstract base class for all agents, adapted from
app.agents.agent for use in code_agent module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import json
import logging
from collections.abc import Callable

# Import directly from message_thread to avoid circular import via package __init__
from .message_thread import MessageThread, FunctionCallIntent

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides per-agent message thread, tool call tracking, and default dispatch_intent.
    """
    
    api_functions: list[str] = []

    def __init__(self, agent_id: str):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
        """
        # Each agent has its own thread
        self.msg_thread = MessageThread()
        self.agent_id = agent_id
        # Tracking of tool calls
        self.tool_call_sequence: list[dict] = []
        self.tool_call_layers: list[list[dict]] = []
        self.curr_tool: str | None = None
        self.iteration_num = 0
        self.finish_status = True
    
    def add_user_message(self, text: str):
        """Add a user message to the thread."""
        self.msg_thread.add_user(text)

    def add_system_message(self, text: str):
        """Add a system message to the thread."""
        self.msg_thread.add_system(text)

    def add_model_message(self, text: str, tools: list):
        """Add a model message to the thread."""
        self.msg_thread.add_model(text, tools)

    @abstractmethod
    def run_task(self, print_callback: Callable[[dict], None] | None = None) -> tuple[str, str, bool]:
        """
        Execute the agent's primary function.
        
        Args:
            print_callback: Optional callback for printing progress
        
        Returns:
            Tuple of (output, summary, success):
            - output (str): raw tool or LLM output
            - summary (str): one-line summary
            - success (bool): whether the action succeeded
        """
        pass
    
    def init_msg_thread(self) -> None:
        """Initialize the message thread. Override in subclasses if needed."""
        pass

    def dispatch_intent(
        self,
        intent: FunctionCallIntent,
    ) -> tuple[str, str, bool]:
        """
        Dispatch a FunctionCallIntent to call the agent's tool methods.
        
        Args:
            intent: Function call intent to dispatch
        
        Returns:
            Tuple of (result, summary, success)
        """
        if intent.func_name not in self.api_functions:
            error = f"Unknown function name {intent.func_name}."
            summary = "You called a tool that does not exist."
            return error, summary, False

        func_obj = getattr(self, intent.func_name)
        try:
            self.curr_tool = intent.func_name
            call_res = func_obj(**intent.arg_values)
        except Exception as e:
            logger.exception(f"Error in tool call {intent.func_name}: {e}")
            error = str(e)
            summary = "Tool raised an exception."
            call_res = (error, summary, False)

        logger.debug("Result of dispatch_intent: %s", call_res)

        # Record the call
        result, _, ok = call_res
        self.tool_call_sequence.append(
            intent.to_dict_with_result(ok, result, self.agent_id)
        )

        return call_res

    def start_new_layer(self):
        """Start a new layer of tool calls."""
        self.tool_call_layers.append([])

    def reset_tool_sequence(self):
        """Reset the tool call sequence."""
        self.tool_call_sequence = []

    def dump_tool_sequence(self, output_dir: str):
        """
        Dump the tool call sequence to a file.
        
        Args:
            output_dir: Directory to save the tool sequence
        """
        os.makedirs(output_dir, exist_ok=True)
        seq_file = os.path.join(output_dir, 'tool_sequence.json')
        with open(seq_file, 'w') as f:
            json.dump(self.tool_call_sequence, f, indent=2)
