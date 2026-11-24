"""
LLM Configuration and Invocation Utilities
Architecture Reference: docs/EXPLANATIONS.md - Section 4 and Chapter 13

First Principle:
An LLM is a next-token predictor conditioned on prefix context. This module
abstracts model initialisation and token accounting, and enforces ONE rule
that protects experimental integrity:

    A run is either LIVE or MOCK. It never silently becomes MOCK halfway.

* MOCK mode is entered only EXPLICITLY (no usable API key, or the caller forces
  it). Every MOCK call is labelled ``mode="mock"`` in its usage record.
* If a LIVE call fails after all retries, ``LLMUnavailableError`` is raised.
  The caller records a FAILED / PARTIAL run. Mock text is never substituted
  (it used to be: that was the root cause of the cross-topic "battery" bug).
  The legacy behaviour can be re-enabled only with ``ALLOW_MOCK_FALLBACK=1``
  and is then recorded as ``mode="mock_fallback"``.
* Run-specific credentials / mode are carried in ContextVars set by
  ``run_llm_context`` so concurrent Streamlit sessions never share an API key
  through ``os.environ``.
"""

import contextlib
import contextvars
import logging
import os
import time
from typing import Any, Dict, Iterator, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("LLM")

_BANNER_PRINTED = False

_RUN_API_KEY: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("run_api_key", default=None)
_FORCE_MOCK: contextvars.ContextVar[bool] = contextvars.ContextVar("force_mock", default=False)

DEFAULT_MODEL = "gemini-3.5-flash"


class LLMUnavailableError(RuntimeError):
    """Raised when a LIVE LLM call cannot be completed (or MOCK is not permitted)."""


def is_placeholder_key(key: Optional[str]) -> bool:
    """Check if the API key is missing or a placeholder string."""
    if not key:
        return True
    cleaned = key.strip().lower()
    return (
        cleaned == ""
        or cleaned.startswith("your")
        or "placeholder" in cleaned
        or "your_gemini_api_key" in cleaned
        or "your_key" in cleaned
        or "your-key" in cleaned
        or "dummy" in cleaned
        or len(cleaned) < 15
    )


@contextlib.contextmanager
def run_llm_context(api_key: Optional[str] = None, force_mock: bool = False) -> Iterator[None]:
    """Scope an API key / forced-mock flag to the current execution context only."""
    t1 = _RUN_API_KEY.set(api_key.strip() if api_key else None)
    t2 = _FORCE_MOCK.set(bool(force_mock))
    try:
        yield
    finally:
        _RUN_API_KEY.reset(t1)
        _FORCE_MOCK.reset(t2)


def resolve_api_key() -> Optional[str]:
    """Run-scoped key first, then the process environment."""
    return _RUN_API_KEY.get() or os.getenv("GEMINI_API_KEY")


def is_live_mode() -> bool:
    """True only if a real key is available AND mock was not forced for this run."""
    if _FORCE_MOCK.get():
        return False
    return not is_placeholder_key(resolve_api_key())


def get_model_name() -> str:
    """The single generation model identifier used across the whole project."""
    name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    if "2.5-flash" in name:  # discontinued upstream
        name = DEFAULT_MODEL
    return name


def mock_fallback_allowed() -> bool:
    return os.getenv("ALLOW_MOCK_FALLBACK", "").strip() == "1"


def print_mode_banner() -> None:
    """Print a loud one-time console banner stating whether LIVE or RESILIENT mode is active."""
    global _BANNER_PRINTED
    if not _BANNER_PRINTED:
        if is_live_mode():
            print("\n" + "=" * 64)
            print(f">>> MODE: LIVE (Gemini active, model={get_model_name()}) <<<")
            print("=" * 64 + "\n")
        else:
            print("\n" + "!" * 64)
            print(">>> MODE: MOCK (placeholder generators - no real API key / forced) <<<")
            print("!" * 64 + "\n")
        _BANNER_PRINTED = True


def estimate_tokens(text: str) -> int:
    """Rough heuristic token estimator (~4 chars per token). Only used when provider usage is unavailable."""
    return max(1, len(text) // 4)


def get_llm(temperature: float = 0.2, model: Optional[str] = None):
    """Return the LangChain ChatGoogleGenerativeAI instance, or None in MOCK mode."""
    if not is_live_mode():
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or get_model_name(),
            google_api_key=resolve_api_key(),
            temperature=temperature,
        )
    except Exception as exc:
        logger.warning("Could not initialise ChatGoogleGenerativeAI: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Usage accounting helpers
# ---------------------------------------------------------------------------
def merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Sum two usage dicts (e.g. first attempt + JSON repair attempt)."""
    out = dict(a)
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        out[k] = int(a.get(k, 0)) + int(b.get(k, 0))
    out["estimated"] = bool(a.get("estimated")) or bool(b.get("estimated"))
    if a.get("mode") != b.get("mode"):
        out["mode"] = "mixed"
    return out


def usage_record(node: str, usage: Dict[str, Any]) -> Dict[str, Any]:
    """Serialisable per-call telemetry record stored in state['metadata']['llm_calls']."""
    return {
        "node": node,
        "mode": usage.get("mode", "unknown"),
        "model": usage.get("model", ""),
        "temperature": usage.get("temperature"),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "estimated": bool(usage.get("estimated", False)),
    }


def _extract_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
            elif hasattr(c, "text"):
                parts.append(c.text)
            else:
                parts.append(str(c))
        content = "\n".join(parts)
    return str(content).strip()


def _is_daily_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text and ("PerDay" in text or "per day" in text.lower())


def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    mock_generator=None,
    *,
    temperature: float = 0.2,
    allow_mock: bool = True,
    node: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """
    Unified LLM invocation.

    Returns ``(text, usage)`` where usage has prompt/completion/total tokens,
    ``estimated`` (True if the provider reported no usage), ``mode``
    (``live`` | ``mock`` | ``mock_fallback``), ``model`` and ``temperature``.

    Raises ``LLMUnavailableError`` if LIVE mode is active and the call fails,
    or if MOCK mode is active but ``allow_mock`` is False (e.g. the Judge).
    """
    print_mode_banner()
    model_name = get_model_name()
    prompt_tokens = estimate_tokens(prompt + (system_instruction or ""))

    if is_live_mode():
        llm = get_llm(temperature=temperature)
        last_exc: Optional[Exception] = None
        attempts = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
        if llm is not None:
            for attempt in range(1, attempts + 1):
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    messages = []
                    if system_instruction:
                        messages.append(SystemMessage(content=system_instruction))
                    messages.append(HumanMessage(content=prompt))

                    response = llm.invoke(messages)
                    content = _extract_text(response.content)

                    completion_tokens = estimate_tokens(content)
                    estimated = True
                    usage_md = getattr(response, "usage_metadata", None)
                    if usage_md and usage_md.get("input_tokens") is not None and usage_md.get("output_tokens") is not None:
                        prompt_tokens = int(usage_md["input_tokens"])
                        completion_tokens = int(usage_md["output_tokens"])
                        estimated = False

                    return content, {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "estimated": estimated,
                        "mode": "live",
                        "model": model_name,
                        "temperature": temperature,
                    }
                except Exception as exc:  # noqa: BLE001 - provider raises many types
                    last_exc = exc
                    if _is_daily_quota_error(exc):
                        logger.warning("Daily free-tier quota exhausted; not retrying.")
                        break
                    if attempt < attempts:
                        wait_sec = attempt * 3
                        logger.warning("Live LLM error (attempt %d/%d): %s. Retrying in %ds...", attempt, attempts, str(exc)[:200], wait_sec)
                        time.sleep(wait_sec)

        # LIVE call failed. Do NOT silently substitute mock content.
        if mock_generator is not None and mock_fallback_allowed():
            content = mock_generator(prompt)
            ct = estimate_tokens(content)
            return content, {
                "prompt_tokens": prompt_tokens, "completion_tokens": ct,
                "total_tokens": prompt_tokens + ct, "estimated": True,
                "mode": "mock_fallback", "model": model_name, "temperature": temperature,
            }
        raise LLMUnavailableError(
            f"LIVE call failed for node '{node or 'unknown'}' on model '{model_name}': {str(last_exc)[:300] if last_exc else 'LLM client unavailable'}"
        )

    # MOCK mode (explicit)
    if not allow_mock or mock_generator is None:
        raise LLMUnavailableError(
            f"MOCK mode is active and mock output is not permitted for node '{node or 'unknown'}'."
        )
    content = mock_generator(prompt)
    ct = estimate_tokens(content)
    return content, {
        "prompt_tokens": prompt_tokens, "completion_tokens": ct,
        "total_tokens": prompt_tokens + ct, "estimated": True,
        "mode": "mock", "model": "mock-generator", "temperature": temperature,
    }
