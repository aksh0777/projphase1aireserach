"""
LLM Configuration and Invocation Utilities
Architecture Reference: docs/EXPLANATIONS.md - Section 4: "LLM Layer & Invocation"

First Principle:
An LLM is a next-token predictor conditioned on prefix context.
This utility abstracts model initialization, token estimation, and provides
resilient fallback execution when running with placeholder keys or during connectivity dropouts.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("LLM")


def is_placeholder_key(key: Optional[str]) -> bool:
    """Check if the API key is missing or a placeholder string."""
    if not key:
        return True
    cleaned = key.strip().lower()
    return (
        cleaned == ""
        or "placeholder" in cleaned
        or "your_gemini_api_key" in cleaned
        or "your_key" in cleaned
        or "dummy" in cleaned
        or len(cleaned) < 15
    )


def estimate_tokens(text: str) -> int:
    """Rough heuristic token estimator (~4 chars per token) when provider usage is unavailable."""
    return max(1, len(text) // 4)


def get_llm():
    """
    Get the configured LangChain ChatGoogleGenerativeAI instance,
    or None if running in mock/placeholder mode.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if is_placeholder_key(api_key):
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("Could not initialize ChatGoogleGenerativeAI: %s. Using resilient generator.", exc)
        return None


def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    mock_generator=None,
) -> Tuple[str, Dict[str, int]]:
    """
    Unified LLM invocation function.
    Returns: (generated_text, token_usage_dict)
    """
    llm = get_llm()
    prompt_tokens = estimate_tokens(prompt + (system_instruction or ""))

    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = []
            if system_instruction:
                messages.append(SystemMessage(content=system_instruction))
            messages.append(HumanMessage(content=prompt))

            response = llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                # In case multimodal blocks are returned
                content = "\n".join([str(c) for c in content])
            content = str(content).strip()

            completion_tokens = estimate_tokens(content)
            # Check for usage metadata if provided by provider
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = usage.get("input_tokens", prompt_tokens)
                completion_tokens = usage.get("output_tokens", completion_tokens)

            return content, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        except Exception as exc:
            logger.warning("Live LLM invocation error: %s. Switching to resilient generator.", exc)

    # Resilient generator fallback
    if mock_generator:
        content = mock_generator(prompt)
    else:
        content = "Analysis completed based on retrieved context."

    completion_tokens = estimate_tokens(content)
    return content, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
