import json

import httpx
import pytest

from agentos.llm import LLMError, Message, OllamaLLM, ToolCall


def make_llm(handler) -> OllamaLLM:
    return OllamaLLM("qwen3:8b", transport=httpx.MockTransport(handler))


def test_request_and_response_mapping() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}
                    ],
                },
                "prompt_eval_count": 120,
                "eval_count": 15,
            },
        )

    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=[ToolCall(name="read_file", arguments={"path": "a"})]),
        Message(role="tool", content="data", tool_name="read_file"),
    ]
    resp = make_llm(handler).chat(history, tools=[{"type": "function"}])

    assert captured["model"] == "qwen3:8b"
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["tools"] == [{"type": "function"}]
    assert captured["messages"][1]["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "a"}}}
    ]
    assert captured["messages"][2] == {"role": "tool", "content": "data", "tool_name": "read_file"}

    assert resp.message.tool_calls[0].name == "calculator"
    assert resp.message.tool_calls[0].arguments == {"expression": "1+1"}
    assert (resp.prompt_tokens, resp.completion_tokens) == (120, 15)


def test_http_error_becomes_llm_error() -> None:
    llm = make_llm(lambda _: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMError, match=r"HTTP 404: .*model not found"):
        llm.chat([Message(role="user", content="hi")], tools=[])
