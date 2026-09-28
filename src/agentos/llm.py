"""Model abstraction: provider-neutral message types and an Ollama implementation."""

from __future__ import annotations

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
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.think = think
        self.temperature = temperature
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_ollama(m) for m in messages],
            "tools": tools,
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature},
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

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
