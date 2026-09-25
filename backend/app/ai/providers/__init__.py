"""LLM provider abstraction.

One interface, three implementations (Gemini, OpenAI, deterministic fallback) so
the five legacy `/api/*` contracts never change when the provider does — and so a
provider outage degrades the response instead of returning 500.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class LLMProvider:
    """Interface. Implementations must never raise for a normal completion failure —
    `complete` returns `None` and the caller falls back."""

    name: str = "base"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str | None:
        raise NotImplementedError


class FallbackProvider(LLMProvider):
    """Deterministic last resort. Mirrors the prototype's heuristic fallbacks so
    the UI stays fully functional with no API key configured."""

    name = "fallback"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str | None:
        return None


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        from google.genai import Client  # lazy: optional extra

        self._client = Client(api_key=settings.gemini_api_key)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str | None:
        from google.genai import types

        config: dict[str, object] = {}
        if system:
            config["system_instruction"] = system
        if json_mode:
            config["response_mime_type"] = "application/json"
        config["max_output_tokens"] = max_tokens or settings.llm_max_output_tokens
        config["temperature"] = 0.2

        response = self._client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(**config),  # type: ignore[arg-type]
        )
        text = getattr(response, "text", None)
        return text.strip() if text else None


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        import openai  # lazy: optional extra

        self._client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str | None:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, object] = {
            "model": settings.openai_model,
            "messages": messages,
            "max_tokens": max_tokens or settings.llm_max_output_tokens,
            "temperature": 0.2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        content = response.choices[0].message.content
        return content.strip() if content else None


_provider: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """Singleton, chosen by config. Any construction failure degrades to fallback."""
    global _provider
    if _provider is not None:
        return _provider

    configured = settings.llm_provider
    try:
        if configured == "gemini" and settings.gemini_api_key:
            _provider = GeminiProvider()
        elif configured == "openai" and settings.openai_api_key:
            _provider = OpenAIProvider()
        else:
            if configured in ("gemini", "openai"):
                log.warning("llm_key_missing", provider=configured, falling_back_to="fallback")
            _provider = FallbackProvider()
    except Exception as exc:  # noqa: BLE001 - provider construction must never break the API
        log.warning("llm_provider_failed", provider=configured, error=str(exc))
        _provider = FallbackProvider()

    log.info("llm_ready", provider=_provider.name)
    return _provider


def reset_provider() -> None:
    """Test hook."""
    global _provider
    _provider = None
