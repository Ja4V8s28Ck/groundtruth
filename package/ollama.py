"""A small, friendly client for a local Ollama server.

Every model call in Groundtruth goes through this module. It talks to
`http://localhost:11434` by default and nowhere else, which is the whole point:
a student's notes and answers never leave the machine.

Only `requests` is required. No Ollama SDK, no cloud keys, no telemetry.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterable, Iterator

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "qwen3:4b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

CHAT = "completion"
EMBED = "embedding"

# Reasoning models such as qwen3 emit <think>...</think> before the answer.
# We strip it so it never leaks into a JSON payload or onto the screen.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)

# Cached across reruns so the sidebar does not re-probe Ollama on every click.
_TAGS_CACHE: dict[str, tuple[float, list[str]]] = {}
_INFO_CACHE: dict[tuple[str, str], set[str]] = {}
_TTL_SECONDS = 10.0


def clear_cache() -> None:
    """Forget what we know about installed models, e.g. after an `ollama pull`."""
    _TAGS_CACHE.clear()
    _INFO_CACHE.clear()


def strip_thinking(text: str) -> str:
    """Remove <think> blocks that reasoning models like qwen3 prepend."""
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _UNCLOSED_THINK.sub("", cleaned)
    return cleaned.strip()


def _extract_json(raw: str) -> Any:
    """Pull a JSON value out of a model response.

    Ollama enforces the schema for us, but a small model can still wrap it in a
    code fence or add a stray sentence. This is the safety net that keeps the
    demo from dying on stage.
    """
    raw = strip_thinking(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fenced = _FENCE.search(raw)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            raw = fenced.group(1).strip()

    start = min(
        (i for i in (raw.find("{"), raw.find("[")) if i != -1),
        default=-1,
    )
    end = max(raw.rfind("}"), raw.rfind("]"))
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise OllamaError(
        "The model did not return usable JSON. Try again, or pick a larger model in the sidebar."
    )


class OllamaError(RuntimeError):
    """An error whose message is safe to show to a non-technical user."""


class OllamaClient:
    """Thin wrapper over the handful of Ollama endpoints Groundtruth needs."""

    def __init__(
        self,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 300,
    ) -> None:
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---------------------------------------------------------------- plumbing

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        model = payload.get("model", "")
        try:
            response = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve` "
                "and make sure it is running."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError(
                "Ollama timed out. A larger model on CPU can be slow - try a smaller one."
            ) from exc

        if response.status_code == 404 and "model" in response.text.lower():
            raise OllamaError(f"Model '{model}' is not installed. Run:  ollama pull {model}")

        if response.status_code >= 400:
            raise OllamaError(self._explain(response.text, model))

        return response.json()

    @staticmethod
    def _explain(body: str, model: str) -> str:
        """Turn Ollama's terse errors into something a person can act on."""
        lowered = body.lower()
        if "does not support chat" in lowered or "does not support completion" in lowered:
            return (
                f"'{model}' is an embedding model and cannot hold a conversation. "
                "Pick a chat model in the sidebar."
            )
        if "does not support embeddings" in lowered or "does not support embedding" in lowered:
            return (
                f"'{model}' cannot turn text into vectors. Pick an embedding model in the sidebar."
            )
        if "not found" in lowered:
            return f"Model '{model}' is not installed. Run:  ollama pull {model}"
        return f"Ollama returned an error for '{model}': {body[:220]}"

    # ------------------------------------------------------------------ status

    def version(self) -> str:
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=10)
            response.raise_for_status()
            return response.json().get("version", "")
        except Exception:  # noqa: BLE001 - status is advisory, never fatal
            return ""

    def list_models(self) -> list[str]:
        cached = _TAGS_CACHE.get(self.base_url)
        if cached and time.monotonic() - cached[0] < _TTL_SECONDS:
            return cached[1]

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=15)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
        except Exception:  # noqa: BLE001
            return []

        _TAGS_CACHE[self.base_url] = (time.monotonic(), names)
        return names

    def capabilities(self, model: str) -> set[str]:
        """What this model can actually do, e.g. {completion, embedding}.

        Ollama reports this from /api/show. Older servers do not, so we fall back
        to a name heuristic rather than guessing wrong and breaking the app.
        """
        key = (self.base_url, model)
        if key in _INFO_CACHE:
            return _INFO_CACHE[key]

        found: set[str] = set()
        try:
            data = self._post("/api/show", {"model": model})
            raw = data.get("capabilities") or []
            found = {str(c).lower() for c in raw}
        except OllamaError:
            found = set()

        if not found:
            # No capability metadata available - infer from the name.
            if "embed" in model.lower() or "bge" in model.lower():
                found = {EMBED}
            else:
                found = {CHAT}

        _INFO_CACHE[key] = found
        return found

    def can_chat(self, model: str) -> bool:
        return CHAT in self.capabilities(model)

    def can_embed(self, model: str) -> bool:
        return EMBED in self.capabilities(model)

    def chat_models(self) -> list[str]:
        """Installed models that can hold a conversation."""
        return [name for name in self.list_models() if self.can_chat(name)]

    def embed_models(self) -> list[str]:
        """Installed models that can turn text into vectors."""
        return [name for name in self.list_models() if self.can_embed(name)]

    def resolve_chat(self, preferred: str = "") -> str:
        """Best available chat model, preferring `preferred` when it qualifies."""
        options = self.chat_models()
        if not options:
            return ""
        if preferred and preferred in options:
            return preferred
        for fallback in (DEFAULT_CHAT_MODEL, "qwen3:8b", "llama3.2:3b", "gemma3:4b"):
            if fallback in options:
                return fallback
        return options[0]

    def resolve_embed(self, preferred: str = "") -> str:
        """Best available embedding model, preferring `preferred` when it qualifies."""
        options = self.embed_models()
        if not options:
            return ""
        if preferred and preferred in options:
            return preferred
        for fallback in (DEFAULT_EMBED_MODEL, "bge-m3", "embeddinggemma", "all-minilm"):
            if fallback in options:
                return fallback
        return options[0]

    def health(self) -> dict[str, Any]:
        """Everything the sidebar needs to tell the truth about the setup."""
        version = self.version()
        online = bool(version)
        return {
            "online": online,
            "version": version,
            "chat_models": self.chat_models() if online else [],
            "embed_models": self.embed_models() if online else [],
        }

    # ----------------------------------------------------------------- calling

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        think: bool = False,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Call /api/chat. Returns full text, or an iterator of chunks if streaming."""
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": stream,
            "think": think,
            "options": options or {"temperature": 0.3},
        }
        if schema is not None:
            payload["format"] = schema

        if not stream:
            return strip_thinking(self._post("/api/chat", payload)["message"]["content"])

        def _iterator() -> Iterator[str]:
            try:
                with requests.post(
                    f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout
                ) as response:
                    if response.status_code >= 400:
                        raise OllamaError(self._explain(response.text, self.chat_model))
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = event.get("message", {}).get("content", "")
                        if piece:
                            yield piece
            except requests.exceptions.ConnectionError as exc:
                raise OllamaError(
                    f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve`."
                ) from exc

        return _iterator()

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
        retries: int = 1,
    ) -> Any:
        """Call the model and get a parsed JSON object back.

        The schema is enforced server-side. If the model still stumbles we give
        it exactly one more chance with an explicit instruction, because a failed
        grading call mid-demo is the one thing we cannot let happen.
        """
        attempt_messages = list(messages)
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            if attempt:
                attempt_messages = attempt_messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON for the required schema. "
                            "Reply with the JSON object only - no prose, no code fence."
                        ),
                    }
                ]
            raw = self.chat(attempt_messages, schema=schema, options=options)
            assert isinstance(raw, str)  # stream is False here
            try:
                return _extract_json(raw)
            except OllamaError as exc:
                last_error = exc

        raise last_error or OllamaError("The model did not return usable JSON.")

    # --------------------------------------------------------------- embedding

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a list of strings. Uses the batch /api/embed endpoint."""
        items = [t for t in texts if t.strip()]
        if not items:
            return []

        try:
            return self._post("/api/embed", {"model": self.embed_model, "input": items})["embeddings"]
        except (OllamaError, KeyError, TypeError):
            pass

        # Older servers only expose the one-at-a-time endpoint.
        vectors: list[list[float]] = []
        for text in items:
            try:
                vectors.append(
                    self._post("/api/embeddings", {"model": self.embed_model, "prompt": text})["embedding"]
                )
            except (OllamaError, KeyError):
                raise OllamaError(
                    f"Could not embed your notes with '{self.embed_model}'. "
                    f"Pull an embedding model with:  ollama pull {DEFAULT_EMBED_MODEL}"
                )
        return vectors
