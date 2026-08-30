from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main.py"
SPEC = importlib.util.spec_from_file_location("hermes_approval_pipe", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Pipe = MODULE.Pipe


def sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


def capabilities(**overrides: bool) -> dict[str, Any]:
    features = {
        "run_submission": True,
        "run_events_sse": True,
        "run_approval_response": True,
        "approval_events": True,
    }
    features.update(overrides)
    return {"features": features}


async def collect(iterator) -> str:
    return "".join([chunk async for chunk in iterator])


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    def make_pipe(self, handler) -> Pipe:
        pipe = Pipe()
        pipe.valves.HERMES_BASE_URL = "http://hermes.test:8642/v1"
        pipe.valves.HERMES_API_KEY = "test-gateway-key"
        pipe._transport = httpx.MockTransport(handler)
        return pipe

    async def test_pretty_prints_json_approval_payload(self):
        message = MODULE._approval_message(
            {
                "description": "Open an internet session",
                "tool": "browser.create_session",
                "command": json.dumps(
                    {
                        "url": "https://example.com",
                        "options": {"private": True, "timeout": 30},
                    }
                ),
            }
        )

        self.assertIn("**Tool:** `browser.create_session`", message)
        self.assertIn("**Request details**", message)
        self.assertIn("```json", message)
        self.assertIn('\n  "options": {\n', message)
        self.assertIn('\n    "private": true,\n', message)
        self.assertIn('\n  "url": "https://example.com"\n', message)

    async def test_pretty_prints_double_encoded_json_payload(self):
        encoded = json.dumps(json.dumps({"action": "deploy", "count": 2}))

        preview, language = MODULE._approval_preview(encoded)

        self.assertEqual(language, "json")
        self.assertEqual(preview, '{\n  "action": "deploy",\n  "count": 2\n}')

    async def test_pretty_prints_json_inside_hermes_command_envelope(self):
        preview, language = MODULE._approval_preview(
            'POST /v1/sessions Arguments: {"body":{"url":"https://example.com"}}'
        )

        self.assertEqual(language, "text")
        self.assertEqual(
            preview,
            'POST /v1/sessions Arguments:\n{\n  "body": {\n'
            '    "url": "https://example.com"\n  }\n}',
        )

    async def test_markdown_fence_contains_untrusted_backticks(self):
        message = MODULE._approval_message(
            {
                "description": "Review command",
                "command": "printf '```'",
            }
        )

        self.assertIn("````text\nprintf '```'\n````", message)

    async def test_status_preview_summarizes_json(self):
        preview = MODULE._status_preview(
            json.dumps(
                {
                    "url": "https://example.com",
                    "headers": {"Accept": "application/json"},
                    "attempts": [1, 2],
                    "enabled": True,
                    "extra": "hidden",
                }
            )
        )

        self.assertEqual(
            preview,
            "url=https://example.com · headers={…} · attempts=[2 items] · "
            "enabled=true · +1 more",
        )

    async def test_allows_once_and_streams_without_duplicate_final_output(self):
        requests: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/capabilities":
                return httpx.Response(200, json=capabilities())
            if path == "/v1/runs" and request.method == "POST":
                requests.append(
                    {
                        "kind": "run",
                        "body": json.loads(request.content),
                        "session_key": request.headers.get("x-hermes-session-key"),
                    }
                )
                return httpx.Response(
                    202, json={"run_id": "run_abc", "status": "started"}
                )
            if path == "/v1/runs/run_abc/events":
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse(
                        {
                            "event": "approval.request",
                            "run_id": "run_abc",
                            "description": "Delete generated cache",
                            "command": "rm -r build-cache",
                        },
                        {"event": "approval.responded", "choice": "once"},
                        {"event": "message.delta", "delta": "Done"},
                        {"event": "run.completed", "output": "Done"},
                    ),
                )
            if path == "/v1/runs/run_abc/approval":
                requests.append(
                    {"kind": "approval", "body": json.loads(request.content)}
                )
                return httpx.Response(
                    200,
                    json={"run_id": "run_abc", "choice": "once", "resolved": 1},
                )
            raise AssertionError(f"unexpected request: {request.method} {path}")

        pipe = self.make_pipe(handler)
        calls: list[dict[str, Any]] = []
        statuses: list[dict[str, Any]] = []

        async def event_call(event):
            calls.append(event)
            return True

        async def event_emitter(event):
            statuses.append(event)

        output = await collect(
            pipe.pipe(
                {
                    "messages": [
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "Earlier question"},
                        {"role": "assistant", "content": "Earlier answer"},
                        {"role": "user", "content": "Clean the cache"},
                    ]
                },
                __user__={"id": "user-1"},
                __metadata__={"chat_id": "chat-1"},
                __event_call__=event_call,
                __event_emitter__=event_emitter,
            )
        )

        self.assertEqual(output, "Done")
        self.assertEqual(len(calls), 1)
        self.assertIn("rm -r build-cache", calls[0]["data"]["message"])
        approval = next(item for item in requests if item["kind"] == "approval")
        self.assertEqual(approval["body"], {"choice": "once"})
        run = next(item for item in requests if item["kind"] == "run")
        self.assertEqual(run["body"]["input"], "Clean the cache")
        self.assertEqual(run["body"]["instructions"], "Be concise.")
        self.assertEqual(
            run["body"]["conversation_history"],
            [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
            ],
        )
        self.assertTrue(run["body"]["session_id"].startswith("openwebui:"))
        self.assertTrue(run["session_key"].startswith("openwebui:"))
        self.assertTrue(statuses[-1]["data"]["done"])

    async def test_missing_approval_ui_denies(self):
        choices: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/capabilities":
                return httpx.Response(200, json=capabilities())
            if path == "/v1/runs":
                return httpx.Response(202, json={"run_id": "run_denied"})
            if path == "/v1/runs/run_denied/events":
                return httpx.Response(
                    200,
                    content=sse(
                        {
                            "event": "approval.request",
                            "command": "dangerous command",
                            "description": "Dangerous operation",
                        },
                        {"event": "run.completed", "output": "Blocked by user"},
                    ),
                )
            if path == "/v1/runs/run_denied/approval":
                choices.append(json.loads(request.content)["choice"])
                return httpx.Response(200, json={"resolved": 1})
            raise AssertionError(f"unexpected request: {request.method} {path}")

        pipe = self.make_pipe(handler)
        output = await collect(
            pipe.pipe(
                {"messages": [{"role": "user", "content": "Do it"}]},
                __user__={"id": "user-1"},
                __metadata__={"chat_id": "chat-1"},
                __event_call__=None,
            )
        )
        self.assertEqual(choices, ["deny"])
        self.assertEqual(output, "Blocked by user")

    async def test_event_caller_failure_denies(self):
        choices: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/capabilities":
                return httpx.Response(200, json=capabilities())
            if path == "/v1/runs":
                return httpx.Response(202, json={"run_id": "run_disconnect"})
            if path == "/v1/runs/run_disconnect/events":
                return httpx.Response(
                    200,
                    content=sse(
                        {"event": "approval.request", "command": "command"},
                        {"event": "run.completed", "output": "Denied"},
                    ),
                )
            if path == "/v1/runs/run_disconnect/approval":
                choices.append(json.loads(request.content)["choice"])
                return httpx.Response(200, json={"resolved": 1})
            raise AssertionError(f"unexpected request: {request.method} {path}")

        async def disconnected(_event):
            raise RuntimeError("browser disconnected")

        pipe = self.make_pipe(handler)
        output = await collect(
            pipe.pipe(
                {"messages": [{"role": "user", "content": "Do it"}]},
                __event_call__=disconnected,
            )
        )
        self.assertEqual(choices, ["deny"])
        self.assertEqual(output, "Denied")

    async def test_missing_capability_refuses_to_start_run(self):
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path == "/v1/capabilities":
                return httpx.Response(200, json=capabilities(approval_events=False))
            raise AssertionError("run must not start without approval capabilities")

        pipe = self.make_pipe(handler)
        output = await collect(
            pipe.pipe({"messages": [{"role": "user", "content": "Hello"}]})
        )
        self.assertEqual(paths, ["/v1/capabilities"])
        self.assertIn("approval_events", output)

    async def test_malformed_stream_stops_nonterminal_run(self):
        stopped: list[bool] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/capabilities":
                return httpx.Response(200, json=capabilities())
            if path == "/v1/runs":
                return httpx.Response(202, json={"run_id": "run_broken"})
            if path == "/v1/runs/run_broken/events":
                return httpx.Response(200, content=b"data: {not-json}\n\n")
            if path == "/v1/runs/run_broken/stop":
                stopped.append(True)
                return httpx.Response(200, json={"status": "stopping"})
            raise AssertionError(f"unexpected request: {request.method} {path}")

        pipe = self.make_pipe(handler)
        output = await collect(
            pipe.pipe({"messages": [{"role": "user", "content": "Hello"}]})
        )
        self.assertIn("malformed SSE JSON", output)
        self.assertEqual(stopped, [True])

    async def test_failed_approval_resolution_stops_run(self):
        stopped: list[bool] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/capabilities":
                return httpx.Response(200, json=capabilities())
            if path == "/v1/runs":
                return httpx.Response(202, json={"run_id": "run_conflict"})
            if path == "/v1/runs/run_conflict/events":
                return httpx.Response(
                    200,
                    content=sse({"event": "approval.request", "command": "command"}),
                )
            if path == "/v1/runs/run_conflict/approval":
                return httpx.Response(
                    409,
                    json={"error": {"message": "approval not pending"}},
                )
            if path == "/v1/runs/run_conflict/stop":
                stopped.append(True)
                return httpx.Response(200, json={"status": "stopping"})
            raise AssertionError(f"unexpected request: {request.method} {path}")

        pipe = self.make_pipe(handler)

        async def approve(_event):
            return True

        output = await collect(
            pipe.pipe(
                {"messages": [{"role": "user", "content": "Do it"}]},
                __event_call__=approve,
            )
        )
        self.assertIn("approval not pending", output)
        self.assertEqual(stopped, [True])

    def test_session_scopes_are_stable_and_separated(self):
        first = Pipe._scopes({"id": "user-a"}, {"chat_id": "chat-a"})
        repeated = Pipe._scopes({"id": "user-a"}, {"chat_id": "chat-a"})
        other_user = Pipe._scopes({"id": "user-b"}, {"chat_id": "chat-a"})
        other_chat = Pipe._scopes({"id": "user-a"}, {"chat_id": "chat-b"})

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_user)
        self.assertNotEqual(first, other_chat)
        self.assertNotEqual(first[0], first[1])

    async def test_approval_timeout_denies(self):
        choices: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/capabilities":
                return httpx.Response(200, json=capabilities())
            if path == "/v1/runs":
                return httpx.Response(202, json={"run_id": "run_timeout"})
            if path == "/v1/runs/run_timeout/events":
                return httpx.Response(
                    200,
                    content=sse(
                        {"event": "approval.request", "command": "command"},
                        {"event": "run.completed", "output": "Denied"},
                    ),
                )
            if path == "/v1/runs/run_timeout/approval":
                choices.append(json.loads(request.content)["choice"])
                return httpx.Response(200, json={"resolved": 1})
            raise AssertionError(f"unexpected request: {request.method} {path}")

        async def never_returns(_event):
            await asyncio.sleep(60)

        pipe = self.make_pipe(handler)
        pipe.valves.APPROVAL_TIMEOUT_SECONDS = 0.01
        output = await collect(
            pipe.pipe(
                {"messages": [{"role": "user", "content": "Do it"}]},
                __event_call__=never_returns,
            )
        )
        self.assertEqual(choices, ["deny"])
        self.assertEqual(output, "Denied")


if __name__ == "__main__":
    unittest.main()
