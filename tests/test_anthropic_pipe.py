"""Regression tests for the Anthropic manifold's provider-independent logic."""

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any


def _install_dependency_stubs() -> None:
    """Install minimal runtime stubs when Open WebUI test dependencies are absent."""
    if "pydantic" not in sys.modules:
        pydantic = types.ModuleType("pydantic")

        class BaseModel:
            """Small subset of Pydantic used by the pipe's valve declarations."""

            def __init__(self, **values: Any) -> None:
                for name, value in self.__class__.__dict__.items():
                    if not name.startswith("_") and not callable(value):
                        setattr(self, name, values.get(name, value))
                for name, value in values.items():
                    setattr(self, name, value)

        def field(default: Any = None, **_: Any) -> Any:
            """Return the declared default value."""
            return default

        pydantic.BaseModel = BaseModel
        pydantic.Field = field
        sys.modules["pydantic"] = pydantic

    if "httpx" not in sys.modules:
        httpx = types.ModuleType("httpx")

        class HTTPError(Exception):
            """Base HTTP error stub."""

        class TransportError(HTTPError):
            """Transport error stub."""

        class HTTPStatusError(HTTPError):
            """HTTP status error stub."""

            def __init__(self, response: Any) -> None:
                self.response = response

        class Timeout:
            """Timeout configuration stub."""

            def __init__(self, *_: Any, **__: Any) -> None:
                pass

        class AsyncClient:
            """Async client annotation stub; HTTP is not exercised by unit tests."""

        httpx.HTTPError = HTTPError
        httpx.TransportError = TransportError
        httpx.HTTPStatusError = HTTPStatusError
        httpx.Timeout = Timeout
        httpx.AsyncClient = AsyncClient
        sys.modules["httpx"] = httpx


_install_dependency_stubs()
MODULE_PATH = Path(__file__).parents[1] / "functions/pipes/anthropic/main.py"
SPEC = importlib.util.spec_from_file_location("anthropic_pipe_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Pipe = MODULE.Pipe


class AnthropicPipeTests(unittest.TestCase):
    """Verify Anthropic payload, thinking, and native-tool translation behavior."""

    def setUp(self) -> None:
        """Create a configured pipe for each test."""
        self.pipe = Pipe()
        self.pipe.valves.ANTHROPIC_API_KEY = "test-key"

    def test_adaptive_capabilities_omit_sampling_and_define_native_tools(self) -> None:
        """Use discovered adaptive-thinking capability data instead of sampling defaults."""
        self.pipe.valves.ENABLE_PROMPT_CACHING = True
        self.pipe.valves.PROMPT_CACHE_TTL = "1h"
        self.pipe.valves.EFFORT = "xhigh"
        self.pipe._model_capabilities = {
            "claude-future": {
                "thinking": {"types": {"adaptive": {"supported": True}}},
                "effort": {"high": {"supported": True}, "low": {"supported": True}},
            }
        }

        def weather(location: str, unit: str = "c") -> str:
            """Look up weather for a location."""
            return f"{location}: 20{unit}"

        payload, tools = self.pipe._build_payload(
            {
                "model": "anthropic.claude-future",
                "messages": [
                    {"role": "system", "content": "Stable instructions"},
                    {"role": "user", "content": "What is the weather?"},
                ],
                "temperature": 0.2,
                "top_p": 0.4,
            },
            {"params": {"function_calling": "native"}},
            {"weather": {"callable": weather}},
        )

        self.assertEqual(payload["model"], "claude-future")
        self.assertEqual(payload["thinking"]["type"], "adaptive")
        self.assertEqual(payload["output_config"]["effort"], "high")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(payload["system"][0]["cache_control"]["ttl"], "1h")
        self.assertEqual(payload["tools"][0]["name"], "weather")
        self.assertEqual(payload["tools"][0]["input_schema"]["required"], ["location"])
        self.assertIn("weather", tools)

    def test_default_function_calling_mode_keeps_openwebui_tools(self) -> None:
        """Treat omitted function-calling mode as native, including generate_image."""

        def generate_image(prompt: str) -> str:
            """Generate an image from a prompt."""
            return prompt

        payload, tools = self.pipe._build_payload(
            {
                "model": "anthropic/claude-sonnet-5",
                "messages": [{"role": "user", "content": "Make an icon."}],
            },
            {},
            {"generate_image": {"callable": generate_image}},
        )

        self.assertEqual(payload["tools"][0]["name"], "generate_image")
        self.assertIn("generate_image", tools)

    def test_manual_thinking_preserves_explicit_sampling(self) -> None:
        """Use manual thinking only for model families that still support it."""
        def calculator(expression: str) -> str:
            """Evaluate an expression."""
            return expression

        payload, _ = self.pipe._build_payload(
            {
                "model": "anthropic/claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Solve this."}],
                "temperature": 0.3,
                "top_k": 10,
                "tool_choice": "required",
            },
            {"params": {"function_calling": "native"}},
            {"calculator": {"callable": calculator}},
        )

        self.assertEqual(payload["thinking"]["type"], "enabled")
        self.assertEqual(payload["thinking"]["budget_tokens"], 4095)
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(payload["top_k"], 10)
        self.assertEqual(payload["tool_choice"], {"type": "auto"})

    def test_manual_thinking_never_exceeds_max_tokens(self) -> None:
        """Keep Haiku's extended-thinking request valid at every requested output size."""
        regular, _ = self.pipe._build_payload(
            {
                "model": "anthropic/claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "Think."}],
                "max_tokens": 4096,
            },
            None,
            None,
        )
        tiny, _ = self.pipe._build_payload(
            {
                "model": "anthropic/claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "Reply briefly."}],
                "max_tokens": 128,
            },
            None,
            None,
        )

        self.assertEqual(regular["thinking"]["budget_tokens"], 4095)
        self.assertNotIn("thinking", tiny)

    def test_openai_tool_history_becomes_anthropic_content_blocks(self) -> None:
        """Translate assistant tool calls and tool-role results into Messages API blocks."""
        _, messages = self.pipe._translate_messages(
            [
                {"role": "user", "content": "Check the weather."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"location":"Boston"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "18 C"},
            ]
        )

        tool_use = messages[1]["content"][-1]
        tool_result = messages[2]["content"][0]
        self.assertEqual(tool_use["type"], "tool_use")
        self.assertEqual(tool_use["input"], {"location": "Boston"})
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "call_1")
        self.assertEqual(messages[2]["role"], "user")

    def test_execute_tools_supports_async_callables_and_errors(self) -> None:
        """Execute injected async callables and return Anthropic-shaped errors."""

        async def add(left: int, right: int) -> int:
            """Add two integers."""
            return left + right

        async def run_test() -> list[dict[str, Any]]:
            return await self.pipe._execute_tools(
                [
                    {"id": "call_1", "name": "add", "input": {"left": 2, "right": 3}},
                    {"id": "call_2", "name": "missing", "input": {}},
                ],
                {"add": add},
                None,
            )

        results = asyncio.run(run_test())
        self.assertEqual(results[0]["content"], "5")
        self.assertTrue(results[1]["is_error"])
        self.assertIn("Unknown tool", results[1]["content"])

    def test_format_response_keeps_thinking_separate_from_text(self) -> None:
        """Render thinking only when the display valve permits it."""
        response = {
            "content": [
                {"type": "thinking", "thinking": "Reasoning"},
                {"type": "text", "text": "Answer"},
            ]
        }
        self.assertEqual(self.pipe._format_response(response), "<think>Reasoning</think>Answer")
        self.pipe.valves.DISPLAY_THINKING = False
        self.assertEqual(self.pipe._format_response(response), "Answer")

    def test_tool_choice_required_is_released_after_the_initial_tool_round(self) -> None:
        """Do not force Claude into another tool call after it has received a tool result."""

        class EmptyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

        responses = iter(
            [
                {
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
                    ]
                },
                {"content": [{"type": "text", "text": "The final answer."}]},
            ]
        )

        async def create_message(_: Any, __: dict[str, Any]) -> dict[str, Any]:
            return next(responses)

        async def lookup() -> str:
            return "result"

        self.pipe._client = lambda: EmptyClient()  # type: ignore[method-assign]
        self.pipe._create_message = create_message  # type: ignore[method-assign]
        payload = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Search."}]}],
            "tool_choice": {"type": "any"},
        }

        result = asyncio.run(self.pipe._run_non_streaming(payload, {"lookup": lookup}, None))

        self.assertEqual(result, "The final answer.")
        self.assertEqual(payload["tool_choice"], {"type": "auto"})

    def test_streaming_thinking_uses_structured_reasoning_deltas(self) -> None:
        """Do not leak raw think XML while streaming into Open WebUI."""
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Reasoning"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Answer"},
            },
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]

        class Response:
            def raise_for_status(self) -> None:
                return None

            async def aread(self) -> bytes:
                return b""

            async def aiter_lines(self):
                for event in events:
                    yield f"data: {MODULE.json.dumps(event)}"

        class Stream:
            async def __aenter__(self) -> Response:
                return Response()

            async def __aexit__(self, *_: Any) -> None:
                return None

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

            def stream(self, *_: Any, **__: Any) -> Stream:
                return Stream()

        self.pipe._client = lambda: Client()  # type: ignore[method-assign]

        async def collect() -> list[str]:
            return [
                chunk
                async for chunk in self.pipe._stream_messages(
                    {"model": "claude-test", "messages": [], "stream": True}, {}, None
                )
            ]

        chunks = asyncio.run(collect())
        self.assertEqual(chunks[1], "Answer")
        self.assertNotIn("<think>", "".join(chunks))
        reasoning_delta = MODULE.json.loads(chunks[0][len("data: ") :])
        self.assertEqual(reasoning_delta["choices"][0]["delta"]["reasoning_content"], "Reasoning")

    def test_streaming_http_error_reads_the_body_before_describing_it(self) -> None:
        """Avoid httpx ResponseNotRead when a streamed Anthropic request is rejected."""

        class Response:
            status_code = 400
            headers = {"request-id": "req_test"}

            def __init__(self) -> None:
                self.was_read = False

            @property
            def text(self) -> str:
                if not self.was_read:
                    raise RuntimeError("Attempted to access streaming response content")
                return '{"error":{"message":"invalid request"}}'

            def json(self) -> dict[str, Any]:
                if not self.was_read:
                    raise RuntimeError("Attempted to access streaming response content")
                return {"error": {"message": "invalid request"}}

            async def aread(self) -> bytes:
                self.was_read = True
                return self.text.encode()

            def raise_for_status(self) -> None:
                try:
                    raise MODULE.httpx.HTTPStatusError("bad request", request=None, response=self)
                except TypeError:
                    # The dependency stub used in this standalone test suite accepts only
                    # the response object, unlike real httpx.
                    raise MODULE.httpx.HTTPStatusError(self)

            async def aiter_lines(self):
                if False:
                    yield ""

        response = Response()

        class Stream:
            async def __aenter__(self) -> Response:
                return response

            async def __aexit__(self, *_: Any) -> None:
                return None

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

            def stream(self, *_: Any, **__: Any) -> Stream:
                return Stream()

        self.pipe._client = lambda: Client()  # type: ignore[method-assign]

        async def collect() -> list[str]:
            return [
                chunk
                async for chunk in self.pipe._stream_messages(
                    {"model": "claude-test", "messages": [], "stream": True}, {}, None
                )
            ]

        self.assertEqual(
            asyncio.run(collect()),
            ["Error: Anthropic stream error: HTTP 400 (request_id=req_test): invalid request"],
        )


if __name__ == "__main__":
    unittest.main()
