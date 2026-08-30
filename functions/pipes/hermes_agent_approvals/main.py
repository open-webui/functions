"""
title: Hermes Agent Approval Bridge
author: wan0net
author_url: https://github.com/wan0net
version: 0.1.2
required_open_webui_version: 0.11.1
license: MIT
requirements: httpx>=0.27.0
tags: hermes-agent, approvals, human-in-the-loop
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]
EventCaller = Callable[[dict[str, Any]], Awaitable[Any]]


class HermesBridgeError(RuntimeError):
    """A controlled, user-displayable bridge failure."""


def _text_content(content: Any) -> str:
    """Flatten OpenAI-style text content without forwarding binary inputs."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "input_text", "output_text"}:
                value = part.get("text", "")
                if isinstance(value, str):
                    text.append(value)
        return "\n".join(value for value in text if value)
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def _conversation_payload(messages: Any) -> tuple[str, list[dict[str, str]], str]:
    """Return last user message, prior history, and layered instructions."""
    if not isinstance(messages, list):
        raise HermesBridgeError("Open WebUI did not provide a message list")

    last_user_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = index
            break
    if last_user_index is None:
        raise HermesBridgeError("No user message was found")

    user_input = _text_content(messages[last_user_index].get("content")).strip()
    if not user_input:
        raise HermesBridgeError(
            "The Hermes approval bridge currently requires a text user message"
        )

    history: list[dict[str, str]] = []
    instructions: list[str] = []
    for message in messages[:last_user_index]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = _text_content(message.get("content")).strip()
        if not content:
            continue
        if role == "system":
            instructions.append(content)
        elif role in {"user", "assistant"}:
            history.append({"role": role, "content": content})

    return user_input, history, "\n\n".join(instructions)


def _confirmed(value: Any) -> bool:
    """Normalize the confirmation return shapes used across Open WebUI releases."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("confirmed", "result", "value", "approved"):
            candidate = value.get(key)
            if isinstance(candidate, bool):
                return candidate
    return False


def _safe_preview(value: Any, limit: int = 4_000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… (truncated by bridge)"


def _json_payload(value: Any) -> dict[str, Any] | list[Any] | None:
    """Return structured JSON from native or encoded approval payloads."""
    candidate = value
    for _ in range(2):
        if isinstance(candidate, (dict, list)):
            return candidate
        if not isinstance(candidate, str):
            return None
        text = candidate.strip()
        if not text or text[0] not in '{["':
            return None
        try:
            candidate = json.loads(text)
        except (json.JSONDecodeError, UnicodeError):
            return None
    return candidate if isinstance(candidate, (dict, list)) else None


def _embedded_json_payload(value: Any) -> tuple[str, dict[str, Any] | list[Any]] | None:
    """Extract a trailing JSON payload from Hermes command envelopes."""
    if not isinstance(value, str):
        return None
    match = re.search(r"\bArguments:\s*(?=[{\[])", value, flags=re.IGNORECASE)
    if not match:
        return None
    payload = _json_payload(value[match.end() :])
    if payload is None:
        return None
    return value[: match.end()].rstrip(), payload


def _markdown_fence(text: str, language: str = "text") -> str:
    """Fence untrusted previews without allowing embedded backticks to escape."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _approval_preview(value: Any, limit: int = 4_000) -> tuple[str, str]:
    """Format a redacted Hermes approval payload for an attended dialog."""
    payload = _json_payload(value)
    if payload is not None:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        return _safe_preview(text, limit), "json"
    embedded = _embedded_json_payload(value)
    if embedded is not None:
        prefix, payload = embedded
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        return _safe_preview(f"{prefix}\n{text}", limit), "text"
    return _safe_preview(value or "(command not provided)", limit), "text"


def _summary_value(value: Any) -> str:
    if isinstance(value, dict):
        return "{…}"
    if isinstance(value, list):
        return f"[{len(value)} item{'s' if len(value) != 1 else ''}]"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).replace("\n", " ").strip()


def _status_preview(value: Any, limit: int = 300) -> str:
    """Keep live tool status useful without dumping dense JSON into the chat."""
    payload = _json_payload(value)
    if isinstance(payload, dict):
        parts = [
            f"{key}={_summary_value(item)}" for key, item in list(payload.items())[:4]
        ]
        if len(payload) > 4:
            parts.append(f"+{len(payload) - 4} more")
        return _safe_preview(" · ".join(parts), limit)
    if isinstance(payload, list):
        return f"{len(payload)} item{'s' if len(payload) != 1 else ''}"
    return _safe_preview(value, limit).replace("\n", " ")


def _approval_message(event: dict[str, Any]) -> str:
    description = _safe_preview(
        event.get("description") or "Hermes requests permission to continue", 1_000
    )
    command, language = _approval_preview(event.get("command"))
    tool = _safe_preview(event.get("tool") or event.get("tool_name"), 160)
    tool_line = f"\n\n**Tool:** `{tool}`" if tool else ""
    label = "Request details" if language == "json" else "Command/code preview"
    return (
        f"{description}{tool_line}\n\n**{label}**\n\n"
        f"{_markdown_fence(command, language)}"
    )


async def _iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Parse JSON data fields from a Server-Sent Events response."""
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                data_lines.clear()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise HermesBridgeError(
                        "Hermes returned malformed SSE JSON"
                    ) from exc
                if isinstance(payload, dict):
                    yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as exc:
            raise HermesBridgeError(
                "Hermes returned malformed trailing SSE JSON"
            ) from exc
        if isinstance(payload, dict):
            yield payload


class Pipe:
    class Valves(BaseModel):
        HERMES_BASE_URL: str = Field(
            default=os.getenv("HERMES_BASE_URL", "http://hermes:8642/v1"),
            description="Hermes API base URL, including /v1.",
        )
        HERMES_API_KEY: str = Field(
            default=os.getenv("HERMES_API_KEY", ""),
            description="Bearer key matching Hermes API_SERVER_KEY.",
        )
        MODEL_ID: str = Field(
            default="hermes-agent",
            description="Hermes model/route passed to the runs API.",
        )
        MODEL_NAME: str = Field(
            default="Hermes Agent (Approvals)",
            description="Model name shown in Open WebUI.",
        )
        APPROVAL_TIMEOUT_SECONDS: float = Field(
            default=300.0,
            ge=1.0,
            le=3_600.0,
            description="Maximum time to wait for Allow/Deny before denying.",
        )
        CONNECT_TIMEOUT_SECONDS: float = Field(default=10.0, ge=1.0, le=120.0)
        REQUEST_TIMEOUT_SECONDS: float = Field(default=30.0, ge=1.0, le=300.0)
        VERIFY_TLS: bool = Field(
            default=True,
            description="Verify HTTPS certificates. Keep enabled outside private HTTP networks.",
        )
        SHOW_TOOL_PROGRESS: bool = Field(default=True)
        REQUIRE_APPROVAL_CAPABILITIES: bool = Field(
            default=True,
            description="Refuse to run unless Hermes advertises run approvals.",
        )

    def __init__(self) -> None:
        self.type = "manifold"
        self.id = "hermes_approvals"
        self.name = ""
        self.valves = self.Valves()
        # Test seam only. Production leaves this as None.
        self._transport: httpx.AsyncBaseTransport | None = None

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": self.valves.MODEL_ID, "name": self.valves.MODEL_NAME}]

    def _base_url(self) -> str:
        value = self.valves.HERMES_BASE_URL.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise HermesBridgeError("HERMES_BASE_URL must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise HermesBridgeError(
                "HERMES_BASE_URL must not contain credentials, a query, or a fragment"
            )
        if not parsed.path.rstrip("/").endswith("/v1"):
            raise HermesBridgeError("HERMES_BASE_URL must include the /v1 suffix")
        return value

    def _headers(self, *, session_key: str | None = None) -> dict[str, str]:
        key = self.valves.HERMES_API_KEY.strip()
        if not key:
            raise HermesBridgeError("Set HERMES_API_KEY in the Function valves")
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if session_key:
            headers["X-Hermes-Session-Key"] = session_key
        return headers

    def _client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            connect=self.valves.CONNECT_TIMEOUT_SECONDS,
            read=None,
            write=self.valves.REQUEST_TIMEOUT_SECONDS,
            pool=self.valves.REQUEST_TIMEOUT_SECONDS,
        )
        return httpx.AsyncClient(
            timeout=timeout,
            verify=self.valves.VERIFY_TLS,
            follow_redirects=False,
            transport=self._transport,
        )

    @staticmethod
    def _scopes(
        user: dict[str, Any] | None, metadata: dict[str, Any] | None
    ) -> tuple[str, str | None]:
        user = user or {}
        metadata = metadata or {}
        user_id = str(user.get("id") or user.get("user_id") or "").strip()
        chat_id = str(
            metadata.get("chat_id") or metadata.get("session_id") or ""
        ).strip()
        if not user_id or not chat_id:
            return f"openwebui:{uuid.uuid4().hex}", None

        digest = hashlib.sha256(f"{user_id}\x00{chat_id}".encode()).hexdigest()
        return f"openwebui:{digest[:32]}", f"openwebui:{digest[32:64]}"

    @staticmethod
    async def _emit(
        emitter: EventEmitter | None, description: str, *, done: bool = False
    ) -> None:
        if emitter is None:
            return
        with suppress(Exception):
            await emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = (
                payload.get("error", payload) if isinstance(payload, dict) else payload
            )
            if isinstance(error, dict):
                detail = (
                    error.get("message") or error.get("detail") or error.get("code")
                )
            else:
                detail = error
        except (ValueError, UnicodeError):
            detail = response.text
        return _safe_preview(detail or response.reason_phrase, 800)

    async def _require_capabilities(
        self, client: httpx.AsyncClient, base_url: str, headers: dict[str, str]
    ) -> None:
        if not self.valves.REQUIRE_APPROVAL_CAPABILITIES:
            return
        response = await client.get(f"{base_url}/capabilities", headers=headers)
        if response.status_code != 200:
            raise HermesBridgeError(
                f"Hermes capability check failed ({response.status_code}): "
                f"{self._error_detail(response)}"
            )
        payload = response.json()
        features = payload.get("features", {}) if isinstance(payload, dict) else {}
        required = {
            "run_submission",
            "run_events_sse",
            "run_approval_response",
            "approval_events",
        }
        missing = sorted(name for name in required if features.get(name) is not True)
        if missing:
            raise HermesBridgeError(
                "Hermes does not advertise required approval capabilities: "
                + ", ".join(missing)
            )

    async def _resolve_approval(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        run_id: str,
        event: dict[str, Any],
        event_call: EventCaller | None,
        event_emitter: EventEmitter | None,
    ) -> None:
        message = _approval_message(event)
        approved = False
        reason = "Approval UI unavailable; denied"

        if event_call is not None:
            try:
                result = await asyncio.wait_for(
                    event_call(
                        {
                            "type": "confirmation",
                            "data": {
                                "title": "Hermes requests approval",
                                "message": message,
                                "confirm": "Allow once",
                                "cancel": "Deny",
                            },
                        }
                    ),
                    timeout=self.valves.APPROVAL_TIMEOUT_SECONDS,
                )
                approved = _confirmed(result)
                reason = "Allowed once" if approved else "Denied by user"
            except asyncio.TimeoutError:
                reason = "Approval timed out; denied"
            # The UI callback is plugin-owned and may raise arbitrary transport
            # or serialization errors. Every such failure is a denial.
            except Exception:  # noqa: BLE001
                reason = "Approval UI disconnected or failed; denied"

        choice = "once" if approved else "deny"
        response = await client.post(
            f"{base_url}/runs/{run_id}/approval",
            headers=headers,
            json={"choice": choice},
        )
        if response.status_code != 200:
            raise HermesBridgeError(
                f"Hermes rejected the {choice} decision ({response.status_code}): "
                f"{self._error_detail(response)}"
            )
        await self._emit(event_emitter, reason)

    async def _stop_run(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        run_id: str,
    ) -> None:
        with suppress(Exception):
            await client.post(
                f"{base_url}/runs/{run_id}/stop",
                headers=headers,
                timeout=self.valves.REQUEST_TIMEOUT_SECONDS,
            )

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __event_emitter__: EventEmitter | None = None,
        __event_call__: EventCaller | None = None,
    ) -> AsyncIterator[str]:
        """Run Hermes through its approval-capable runs API."""
        user_input, history, instructions = _conversation_payload(body.get("messages"))
        base_url = self._base_url()
        session_id, session_key = self._scopes(__user__, __metadata__)
        headers = self._headers(session_key=session_key)
        run_id: str | None = None
        terminal = False
        emitted_delta = False

        await self._emit(__event_emitter__, "Connecting to Hermes")

        async with self._client() as client:
            try:
                await self._require_capabilities(client, base_url, headers)
                payload: dict[str, Any] = {
                    "input": user_input,
                    "session_id": session_id,
                    "model": self.valves.MODEL_ID,
                }
                if history:
                    payload["conversation_history"] = history
                if instructions:
                    payload["instructions"] = instructions

                response = await client.post(
                    f"{base_url}/runs", headers=headers, json=payload
                )
                if response.status_code != 202:
                    raise HermesBridgeError(
                        f"Hermes refused the run ({response.status_code}): "
                        f"{self._error_detail(response)}"
                    )
                run_data = response.json()
                run_id = str(run_data.get("run_id", "")).strip()
                if not re.fullmatch(r"run_[A-Za-z0-9]+", run_id):
                    raise HermesBridgeError("Hermes returned an invalid run identifier")

                await self._emit(__event_emitter__, "Hermes is working")
                async with client.stream(
                    "GET", f"{base_url}/runs/{run_id}/events", headers=headers
                ) as stream:
                    if stream.status_code != 200:
                        detail = await stream.aread()
                        raise HermesBridgeError(
                            f"Hermes event stream failed ({stream.status_code}): "
                            f"{_safe_preview(detail.decode(errors='replace'), 800)}"
                        )

                    async for event in _iter_sse(stream):
                        event_type = str(event.get("event", ""))
                        if event_type == "message.delta":
                            delta = event.get("delta")
                            if isinstance(delta, str) and delta:
                                emitted_delta = True
                                yield delta
                        elif event_type == "approval.request":
                            await self._emit(
                                __event_emitter__, "Waiting for your approval"
                            )
                            await self._resolve_approval(
                                client,
                                base_url,
                                headers,
                                run_id,
                                event,
                                __event_call__,
                                __event_emitter__,
                            )
                        elif (
                            event_type == "tool.started"
                            and self.valves.SHOW_TOOL_PROGRESS
                        ):
                            tool = _safe_preview(event.get("tool") or "tool", 120)
                            preview = _status_preview(event.get("preview"), 300)
                            suffix = f": {preview}" if preview else ""
                            await self._emit(
                                __event_emitter__, f"Hermes started {tool}{suffix}"
                            )
                        elif (
                            event_type == "tool.completed"
                            and self.valves.SHOW_TOOL_PROGRESS
                        ):
                            tool = _safe_preview(event.get("tool") or "tool", 120)
                            state = "failed" if event.get("error") else "completed"
                            await self._emit(
                                __event_emitter__, f"Hermes {state} {tool}"
                            )
                        elif event_type == "run.completed":
                            terminal = True
                            output = event.get("output")
                            if not emitted_delta and isinstance(output, str):
                                yield output
                            await self._emit(
                                __event_emitter__, "Hermes completed", done=True
                            )
                        elif event_type == "run.failed":
                            terminal = True
                            error = _safe_preview(
                                event.get("error") or "Hermes run failed"
                            )
                            yield f"\n\n⚠️ {error}"
                            await self._emit(
                                __event_emitter__, "Hermes failed", done=True
                            )
                        elif event_type == "run.cancelled":
                            terminal = True
                            yield "\n\n⚠️ Hermes run was cancelled."
                            await self._emit(
                                __event_emitter__, "Hermes cancelled", done=True
                            )

                if not terminal:
                    status = await client.get(
                        f"{base_url}/runs/{run_id}", headers=headers
                    )
                    if status.status_code == 200:
                        state = status.json()
                        if state.get("status") == "completed":
                            terminal = True
                            output = state.get("output")
                            if not emitted_delta and isinstance(output, str):
                                yield output
                        elif state.get("status") in {"failed", "cancelled"}:
                            terminal = True
                            yield f"\n\n⚠️ {_safe_preview(state.get('error') or state.get('status'))}"
                    if not terminal:
                        raise HermesBridgeError(
                            "Hermes closed the event stream before reaching a terminal state"
                        )
            except asyncio.CancelledError:
                raise
            except HermesBridgeError as exc:
                yield f"\n\n⚠️ Hermes approval bridge: {exc}"
                await self._emit(__event_emitter__, "Hermes bridge failed", done=True)
            except httpx.HTTPError as exc:
                yield f"\n\n⚠️ Hermes approval bridge transport error: {type(exc).__name__}"
                await self._emit(
                    __event_emitter__, "Hermes connection failed", done=True
                )
            finally:
                if run_id and not terminal:
                    await asyncio.shield(
                        self._stop_run(client, base_url, headers, run_id)
                    )
