"""Utility functions for the general agent."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any


def check_sandbox_fusion(url: str | None = None, timeout: int = 3, max_attempts: int = 3) -> bool:
    """Check if SandboxFusion service is available.
    
    Args:
        url: SandboxFusion service URL (default: from SANDBOX_FUSION_URL env var)
        timeout: Connection timeout in seconds
        max_attempts: Maximum number of connection attempts
        
    Returns:
        True if service is available, False otherwise
    """
    if url is None:
        url = os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080")
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Extract host and port from URL
            host = url.replace("http://", "").replace("https://", "").split("/")[0]
            if ":" in host:
                host, port = host.split(":")
                port = int(port)
            else:
                port = int(os.getenv("SANDBOX_FUSION_PORT", "8080"))
            
            # Check port connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result != 0:
                if attempt < max_attempts:
                    continue
                return False
            
            # Try calling the API
            api_url = f"{url.rstrip('/')}/run_code"
            payload = {"code": "print(1)", "language": "python"}
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=timeout)
            return True
            
        except (socket.error, urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt < max_attempts:
                continue
            return False
        except Exception:
            if attempt < max_attempts:
                continue
            return False
    
    return False


def validate_environment() -> tuple[bool, str]:
    """Validate environment configuration.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check SandboxFusion
    sandbox_url = os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080")
    if not check_sandbox_fusion(sandbox_url):
        return False, f"SandboxFusion service is not available at {sandbox_url}"
    
    # Check LLM configuration
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    if provider in {"volcano", "deepseek"}:
        api_key = os.getenv("VOLCANO_API_KEY")
        if not api_key:
            return False, "VOLCANO_API_KEY is not set"
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY is not set"
    # vLLM may not require API key
    
    return True, ""

