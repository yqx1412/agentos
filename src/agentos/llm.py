"""Model abstraction: provider-neutral message types and an Ollama implementation."""

from __future__ import annotations

import contextlib
import uuid
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A tool invocation requested by the model.

    ``arguments`` is normally a dict, but a model may emit a JSON string (valid or not),
    so the raw value is kept and validated later by the tool registry.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # For role="tool": which tool/call this message answers.
    tool_name: str | None = None
    tool_call_id: str | None = None


class ChatResponse(BaseModel):
    message: Message
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    """Anything that can take a conversation plus tool schemas and return one reply."""

    model: str

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse: ...


class LLMError(RuntimeError):
    """The model backend could not produce a response."""


class OllamaLLM:
    """Non-streaming client for Ollama's ``/api/chat`` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        *,
        think: bool = False,
        temperature: float = 0.0,
        num_ctx: int | None = 8192,
        num_predict: int | None = 2048,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.think = think
        self.temperature = temperature
        # Ollama's own default context is 4096 tokens and it truncates longer prompts
        # silently, which multi-step tool use exceeds quickly.
        self.num_ctx = num_ctx
        # Caps one reply. Without it a degenerate generation runs until the HTTP timeout.
        self.num_predict = num_predict
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {"temperature": self.temperature}
        if self.num_ctx is not None:
            opts["num_ctx"] = self.num_ctx
        if self.num_predict is not None:
            opts["num_predict"] = self.num_predict
        return opts

    def warmup(self) -> None:
        """Load the model into memory so the first timed request does not pay for it."""
        self._post(
            "/api/generate",
            {"model": self.model, "keep_alive": "10m", "options": self._options()},
        )

    def unload(self) -> None:
        """Free the model's (V)RAM; errors are ignored because nothing depends on it."""
        with contextlib.suppress(httpx.HTTPError):
            self._client.post("/api/generate", json={"model": self.model, "keep_alive": 0})

    def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            resp = self._client.post(path, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Ollama puts the real reason in the body, e.g. {"error": "..."}.
            detail = exc.response.text.strip()[:500]
            raise LLMError(
                f"Ollama returned HTTP {exc.response.status_code}: {detail or '<empty body>'}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return resp

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_ollama(m) for m in messages],
            "tools": tools,
            "stream": False,
            "think": self.think,
            "options": self._options(),
        }
        resp = self._post("/api/chat", payload)

        data = resp.json()
        raw = data.get("message", {})
        calls = [
            ToolCall(
                name=str(c.get("function", {}).get("name", "")),
                arguments=c.get("function", {}).get("arguments", {}),
            )
            for c in raw.get("tool_calls") or []
        ]
        return ChatResponse(
            message=Message(role="assistant", content=raw.get("content") or "", tool_calls=calls),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )

    def close(self) -> None:
        self._client.close()


def _to_ollama(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
        ]
    if m.role == "tool" and m.tool_name:
        out["tool_name"] = m.tool_name
    return out
