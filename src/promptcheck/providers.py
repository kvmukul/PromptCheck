"""Model providers. Everything is plain `urllib`, so there are no dependencies."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class ProviderError(RuntimeError):
    pass


@dataclass
class Request:
    prompt: str
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    mock_output: str | None = None  # only the mock provider reads this


@dataclass
class Completion:
    text: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider:
    name = "base"

    def __init__(self, model: str, options: dict | None = None):
        self.model = model
        self.options = options or {}

    def complete(self, req: Request) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError


class MockProvider(Provider):
    """Offline provider: returns `mock_output` if the test sets it, else echoes the prompt."""

    name = "mock"

    def complete(self, req: Request) -> Completion:
        text = req.mock_output if req.mock_output is not None else req.prompt
        return Completion(text=text, latency_ms=1.0)


def post_json(url: str, headers: dict, body: dict, timeout: float = 60.0,
              retries: int = 3, sleep=time.sleep) -> dict:
    """POST JSON, retrying rate limits (429) and server errors (5xx) with backoff."""
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in (429, 500, 502, 503, 504, 529) and attempt < retries:
                sleep(2 ** attempt)
                continue
            raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                sleep(2 ** attempt)
                continue
            raise ProviderError(f"network error calling {url}: {exc}") from exc
    raise ProviderError("unreachable")  # pragma: no cover


class AnthropicProvider(Provider):
    name = "anthropic"

    def complete(self, req: Request) -> Completion:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        body: dict = {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if req.system:
            body["system"] = req.system
        start = time.perf_counter()
        data = post_json(
            self.options.get("base_url", "https://api.anthropic.com/v1/messages"),
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            body,
            timeout=float(self.options.get("timeout", 60)),
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return Completion(text, (time.perf_counter() - start) * 1000,
                          usage.get("input_tokens"), usage.get("output_tokens"))


class OpenAICompatibleProvider(Provider):
    """OpenAI, and anything that speaks the same chat-completions API (Ollama, vLLM, ...).

    Options: base_url, api_key_env (default OPENAI_API_KEY), token_param
    (default "max_tokens"; some newer OpenAI models want "max_completion_tokens").
    """

    name = "openai"

    def complete(self, req: Request) -> Completion:
        key_env = self.options.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_env, "")
        if not key and "base_url" not in self.options:
            raise ProviderError(f"{key_env} is not set")
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": req.temperature,
            self.options.get("token_param", "max_tokens"): req.max_tokens,
        }
        headers = {"authorization": f"Bearer {key}"} if key else {}
        start = time.perf_counter()
        data = post_json(
            self.options.get("base_url", "https://api.openai.com/v1/chat/completions"),
            headers, body, timeout=float(self.options.get("timeout", 60)),
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(data)[:200]}") from exc
        usage = data.get("usage", {})
        return Completion(text, (time.perf_counter() - start) * 1000,
                          usage.get("prompt_tokens"), usage.get("completion_tokens"))


PROVIDERS: dict[str, type[Provider]] = {
    "mock": MockProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAICompatibleProvider,
}
DEFAULT_MODELS = {"mock": "mock", "anthropic": "claude-sonnet-5-5"}


def get_provider(name: str, model: str | None = None, options: dict | None = None) -> Provider:
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider {name!r} (known: {', '.join(PROVIDERS)})")
    model = model or DEFAULT_MODELS.get(name)
    if not model:
        raise ProviderError(f"provider {name!r} needs a model (set `model` in [suite])")
    return PROVIDERS[name](model, options)
