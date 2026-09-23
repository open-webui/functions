"""
title: Anthropic Claude Manifold Pipe
authors: justinh-rahb, christian-taillon, jfbloom22, Mark Kazakov, Vincent, NIK-NUB, Snav
author_url: https://github.com/jfbloom22
funding_url: https://github.com/open-webui
version: 0.7.2
required_open_webui_version: 0.4.0
requirements: httpx
license: MIT
description: Async Claude Messages API manifold with native Open WebUI tools, thinking, and caching.
"""

import asyncio
import base64
import binascii
import inspect
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]
ToolRegistry = dict[str, Any]


class Pipe:
    """Translate Open WebUI chat requests to Anthropic's native Messages API."""

    API_URL = "https://api.anthropic.com/v1/messages"
    MODELS_URL = "https://api.anthropic.com/v1/models"
    API_VERSION = "2023-06-01"
    USER_AGENT = "open-webui-anthropic-manifold/0.7.2"
    MAX_IMAGE_SIZE = 10 * 1024 * 1024

    class Valves(BaseModel):
        """Administrator-configurable Anthropic connection and behavior settings."""

        ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key.")
        ANTHROPIC_BASE_URL: str = Field(
            default="https://api.anthropic.com",
            description="Anthropic-compatible API base URL, without the /v1 suffix.",
        )
        REQUEST_TIMEOUT_SECONDS: float = Field(
            default=180.0, description="Per-request timeout in seconds."
        )
        MODEL_CACHE_TTL_SECONDS: int = Field(
            default=600, description="How long to cache the available-model list."
        )
        RETRY_COUNT: int = Field(
            default=2, description="Retries for transient upstream failures."
        )
        ENABLE_THINKING: bool = Field(
            default=True, description="Enable supported Claude thinking modes."
        )
        THINKING_BUDGET: int = Field(
            default=16000,
            description="Manual-thinking budget for Claude 4.5 and earlier supported models.",
        )
        EFFORT: str = Field(
            default="high",
            description="Adaptive-thinking effort: low, medium, high, xhigh, or max.",
        )
        DISPLAY_THINKING: bool = Field(
            default=True, description="Return summarized thinking blocks when supported."
        )
        ENABLE_PROMPT_CACHING: bool = Field(
            default=False,
            description="Cache the stable system prompt at an Anthropic cache breakpoint.",
        )
        PROMPT_CACHE_TTL: str = Field(
            default="5m", description="Prompt-cache TTL: 5m or 1h."
        )
        BETA_FEATURES: str = Field(
            default="", description="Comma-separated Anthropic beta headers, when required."
        )
        MAX_TOOL_ROUNDS: int = Field(
            default=8, description="Maximum native tool-execution rounds per request."
        )
        TOOL_TIMEOUT_SECONDS: float = Field(
            default=30.0, description="Maximum runtime for one injected Open WebUI tool."
        )

    def __init__(self) -> None:
        """Initialize the manifold and its model-discovery cache."""
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
            ANTHROPIC_BASE_URL=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            REQUEST_TIMEOUT_SECONDS=float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "180")),
            MODEL_CACHE_TTL_SECONDS=int(os.getenv("ANTHROPIC_MODEL_CACHE_TTL", "600")),
            RETRY_COUNT=int(os.getenv("ANTHROPIC_RETRY_COUNT", "2")),
            ENABLE_THINKING=os.getenv("ANTHROPIC_ENABLE_THINKING", "true").lower()
            == "true",
            THINKING_BUDGET=int(os.getenv("ANTHROPIC_THINKING_BUDGET", "16000")),
            EFFORT=os.getenv("ANTHROPIC_EFFORT", "high"),
            DISPLAY_THINKING=os.getenv("ANTHROPIC_DISPLAY_THINKING", "true").lower()
            == "true",
            ENABLE_PROMPT_CACHING=os.getenv("ANTHROPIC_ENABLE_PROMPT_CACHING", "false").lower()
            == "true",
            PROMPT_CACHE_TTL=os.getenv("ANTHROPIC_PROMPT_CACHE_TTL", "5m"),
            BETA_FEATURES=os.getenv("ANTHROPIC_BETA_FEATURES", ""),
            MAX_TOOL_ROUNDS=int(os.getenv("ANTHROPIC_MAX_TOOL_ROUNDS", "8")),
            TOOL_TIMEOUT_SECONDS=float(os.getenv("ANTHROPIC_TOOL_TIMEOUT_SECONDS", "30")),
        )
        self._model_cache: list[dict[str, str]] | None = None
        self._model_capabilities: dict[str, dict[str, Any]] = {}
        self._model_cache_time = 0.0

    async def pipes(self) -> list[dict[str, str]]:
        """Return models available to the configured Anthropic account."""
        return await self.get_anthropic_models()

    async def get_anthropic_models(self, force_refresh: bool = False) -> list[dict[str, str]]:
        """Fetch and cache the Anthropic model list."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._model_cache is not None
            and now - self._model_cache_time < self.valves.MODEL_CACHE_TTL_SECONDS
        ):
            return self._model_cache

        if not self.valves.ANTHROPIC_API_KEY:
            return [{"id": "error", "name": "ANTHROPIC_API_KEY is not configured."}]

        try:
            async with self._client() as client:
                response = await client.get(
                    self._api_url("/v1/models"),
                    headers=self._headers(),
                    params={"limit": 1000},
                )
                response.raise_for_status()
                raw_models = response.json().get("data", [])
                models = [
                    {
                        "id": model["id"],
                        "name": model.get("display_name", model["id"]),
                    }
                    for model in raw_models
                    if model.get("id")
                ]
        except httpx.HTTPError as error:
            logger.warning("Unable to fetch Anthropic models: %s", error)
            return [{"id": "error", "name": f"Could not fetch models: {error}"}]

        self._model_cache = models
        self._model_capabilities = {
            model["id"]: model.get("capabilities", {})
            for model in raw_models
            if model.get("id") and isinstance(model.get("capabilities"), dict)
        }
        self._model_cache_time = now
        return models

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: EventEmitter | None = None,
        __metadata__: dict[str, Any] | None = None,
        __tools__: ToolRegistry | None = None,
    ) -> str | AsyncIterator[str]:
        """Run a Claude request, executing injected Open WebUI tools when requested."""
        if not self.valves.ANTHROPIC_API_KEY:
            return "Error: ANTHROPIC_API_KEY is not configured."

        try:
            payload, tool_registry = self._build_payload(body, __metadata__, __tools__)
        except (TypeError, ValueError) as error:
            return f"Error: Invalid Anthropic request: {error}"

        if body.get("stream", False):
            return self._stream_with_status(payload, tool_registry, __event_emitter__)

        await self._emit_status(__event_emitter__, "Contacting Anthropic…", done=False)
        try:
            return await self._run_non_streaming(payload, tool_registry, __event_emitter__)
        except httpx.HTTPError as error:
            logger.exception("Anthropic request failed")
            return f"Error: Anthropic request failed: {self._describe_http_error(error)}"
        except Exception as error:
            logger.exception("Unexpected Anthropic pipe failure")
            return f"Error: {error}"
        finally:
            await self._emit_status(__event_emitter__, "Anthropic request complete.", done=True)

    def _client(self) -> httpx.AsyncClient:
        """Create an HTTP client with a finite timeout and identifiable user agent."""
        timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT_SECONDS, connect=10.0)
        return httpx.AsyncClient(timeout=timeout, headers={"User-Agent": self.USER_AGENT})

    def _api_url(self, path: str) -> str:
        """Build an API URL from the configured base URL and an absolute API path."""
        return f"{self.valves.ANTHROPIC_BASE_URL.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        """Build the required Anthropic API headers."""
        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        if self.valves.BETA_FEATURES.strip():
            headers["anthropic-beta"] = self.valves.BETA_FEATURES.strip()
        return headers

    def _build_payload(
        self,
        body: dict[str, Any],
        metadata: dict[str, Any] | None,
        tool_registry: ToolRegistry | None,
    ) -> tuple[dict[str, Any], ToolRegistry]:
        """Translate an Open WebUI request to Anthropic's Messages payload."""
        model = self._api_model_name(str(body.get("model", "")))
        if not model:
            raise ValueError("A model is required.")

        system, messages = self._translate_messages(body.get("messages", []))
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(body.get("max_tokens", 4096)),
            "stream": bool(body.get("stream", False)),
        }
        if body.get("stop"):
            payload["stop_sequences"] = body["stop"]
        if system:
            payload["system"] = system

        thinking_mode = self._thinking_mode(model)
        self._add_thinking_and_sampling(payload, body, model)
        native_tools, callable_tools = self._native_tools(tool_registry, metadata)
        if native_tools:
            payload["tools"] = native_tools
            if body.get("tool_choice"):
                tool_choice = self._translate_tool_choice(body["tool_choice"])
                if thinking_mode == "manual" and tool_choice["type"] not in {"auto", "none"}:
                    logger.warning(
                        "Manual thinking only supports auto or none tool_choice; using auto."
                    )
                    tool_choice = {"type": "auto"}
                payload["tool_choice"] = tool_choice

        return payload, callable_tools

    def _api_model_name(self, model: str) -> str:
        """Remove the manifold prefix and legacy thinking suffix from a selected model."""
        if model.startswith(f"{self.id}."):
            model = model[len(self.id) + 1 :]
        elif model.startswith(self.name):
            model = model[len(self.name) :]
        return model.removesuffix("-think")

    def _translate_messages(
        self, raw_messages: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Translate Open WebUI, OpenAI-style, and Anthropic content into Messages turns."""
        if not isinstance(raw_messages, list):
            raise ValueError("messages must be a list.")

        system_blocks: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                system_blocks.extend(self._content_blocks(message.get("content", "")))
                continue
            if role == "tool":
                tool_use_id = message.get("tool_call_id") or message.get("id")
                if not tool_use_id:
                    raise ValueError("A tool message requires tool_call_id.")
                result = self._tool_result_block(tool_use_id, message.get("content", ""))
                messages.append({"role": "user", "content": [result]})
                continue
            if role not in {"user", "assistant"}:
                continue

            content = self._content_blocks(message.get("content", ""))
            if role == "assistant":
                content.extend(self._tool_use_blocks(message))
            if content:
                messages.append({"role": role, "content": content})

        if not messages:
            raise ValueError("At least one non-system message is required.")
        if self.valves.ENABLE_PROMPT_CACHING and system_blocks:
            self._add_cache_control(system_blocks[-1])
        return system_blocks, messages

    def _content_blocks(self, content: Any) -> list[dict[str, Any]]:
        """Normalize text, multimodal OpenAI content, and Anthropic blocks."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if content is None:
            return []
        if not isinstance(content, list):
            return [{"type": "text", "text": str(content)}]

        blocks: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                blocks.append({"type": "text", "text": str(item)})
                continue
            item_type = item.get("type")
            if item_type == "text":
                blocks.append({"type": "text", "text": item.get("text", "")})
            elif item_type == "image_url":
                blocks.append(self._image_block(item.get("image_url", {}).get("url", "")))
            elif item_type in {
                "image",
                "document",
                "tool_use",
                "tool_result",
                "thinking",
                "redacted_thinking",
            }:
                blocks.append(dict(item))
            elif "text" in item:
                blocks.append({"type": "text", "text": str(item["text"])})
        return blocks

    def _image_block(self, image_url: str) -> dict[str, Any]:
        """Convert an OpenAI image URL or data URL to an Anthropic image block."""
        if not image_url:
            raise ValueError("An image_url block requires a URL.")
        if not image_url.startswith("data:"):
            return {"type": "image", "source": {"type": "url", "url": image_url}}

        try:
            header, data = image_url.split(",", 1)
            media_type = header.split(";", 1)[0].split(":", 1)[1]
        except (IndexError, ValueError) as error:
            raise ValueError("Invalid image data URL.") from error
        if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise ValueError(f"Unsupported Anthropic image media type: {media_type}.")
        try:
            image_size = len(base64.b64decode(data, validate=True))
        except (ValueError, binascii.Error) as error:
            raise ValueError("Image data URL is not valid base64.") from error
        if image_size > self.MAX_IMAGE_SIZE:
            raise ValueError("Image exceeds Anthropic's 10 MB direct API limit.")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }

    def _tool_use_blocks(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate OpenAI-compatible assistant tool calls to Anthropic tool_use blocks."""
        blocks: list[dict[str, Any]] = []
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            return blocks
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            name = function.get("name") or call.get("name")
            call_id = call.get("id")
            arguments = function.get("arguments", call.get("arguments", {}))
            if not name or not call_id:
                continue
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": arguments if isinstance(arguments, dict) else {},
                }
            )
        return blocks

    def _tool_result_block(
        self, tool_use_id: str, content: Any, is_error: bool = False
    ) -> dict[str, Any]:
        """Build an Anthropic tool_result block from an Open WebUI tool response."""
        result: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id}
        if isinstance(content, list):
            result["content"] = self._content_blocks(content)
        elif isinstance(content, (dict, tuple)):
            result["content"] = json.dumps(content, ensure_ascii=False, default=str)
        else:
            result["content"] = "" if content is None else str(content)
        if is_error:
            result["is_error"] = True
        return result

    def _add_cache_control(self, block: dict[str, Any]) -> None:
        """Add a valid cache breakpoint to the final stable system block."""
        ttl = self.valves.PROMPT_CACHE_TTL if self.valves.PROMPT_CACHE_TTL in {"5m", "1h"} else "5m"
        block["cache_control"] = {"type": "ephemeral", "ttl": ttl}

    def _add_thinking_and_sampling(
        self, payload: dict[str, Any], body: dict[str, Any], model: str
    ) -> None:
        """Apply only model-safe thinking and sampling parameters."""
        thinking_mode = self._thinking_mode(model)
        if self.valves.ENABLE_THINKING and thinking_mode == "adaptive":
            payload["thinking"] = {
                "type": "adaptive",
                "display": "summarized" if self.valves.DISPLAY_THINKING else "omitted",
            }
            payload["output_config"] = {"effort": self._valid_effort(model)}
            return
        if self.valves.ENABLE_THINKING and thinking_mode == "manual":
            # Anthropic requires the thinking budget to be strictly below max_tokens.
            # Keep manual thinking enabled for ordinary chat requests, but skip it for
            # intentionally tiny completions rather than sending an invalid request.
            max_budget = int(payload["max_tokens"]) - 1
            if max_budget >= 1024:
                budget = min(32768, self.valves.THINKING_BUDGET, max_budget)
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            else:
                logger.info("Skipping manual thinking because max_tokens is below 1025.")

        if thinking_mode != "adaptive":
            for parameter in ("temperature", "top_p", "top_k"):
                if parameter in body and body[parameter] is not None:
                    payload[parameter] = body[parameter]

    def _thinking_mode(self, model: str) -> str:
        """Return the documented thinking mode for known current Claude model families."""
        thinking = self._model_capabilities.get(model, {}).get("thinking", {})
        thinking_types = thinking.get("types", {}) if isinstance(thinking, dict) else {}
        if isinstance(thinking_types, dict):
            if thinking_types.get("adaptive", {}).get("supported"):
                return "adaptive"
            if thinking_types.get("enabled", {}).get("supported"):
                return "manual"

        normalized = model.lower()
        if re.match(r"^claude-(?:opus|sonnet)-(?:4-(?:6|7|8)|5)(?:-|$)", normalized):
            return "adaptive"
        if re.match(r"^claude-(?:fable|mythos)-5(?:-|$)", normalized):
            return "adaptive"
        if normalized.startswith("claude-mythos-preview"):
            return "adaptive"
        if re.match(r"^claude-(?:opus|sonnet|haiku)-4-5(?:-|$)", normalized):
            return "manual"
        if re.match(r"^claude-(?:opus|sonnet|haiku)-4(?:-|$)", normalized):
            return "manual"
        return "none"

    def _valid_effort(self, model: str) -> str:
        """Return a supported effort value without sending invalid configuration upstream."""
        effort = self.valves.EFFORT.lower().strip()
        supported = {
            name
            for name, details in self._model_capabilities.get(model, {}).get("effort", {}).items()
            if isinstance(details, dict) and details.get("supported")
        }
        if supported:
            if effort in supported:
                return effort
            if "high" in supported:
                return "high"
            return sorted(supported)[0]
        return effort if effort in {"low", "medium", "high", "xhigh", "max"} else "high"

    def _native_tools(
        self, registry: ToolRegistry | None, metadata: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], ToolRegistry]:
        """Build Anthropic tool definitions from injected Open WebUI callable tools."""
        params = (metadata or {}).get("params", {})
        # Open WebUI's native function-calling mode is the default. Only the
        # explicit legacy mode uses prompt injection instead of native tools.
        if not registry or params.get("function_calling", "native") == "legacy":
            return [], {}

        definitions: list[dict[str, Any]] = []
        callables: ToolRegistry = {}
        for name, definition in registry.items():
            if name.startswith("_"):
                continue
            callable_tool = (
                definition.get("callable") if isinstance(definition, dict) else definition
            )
            if not callable(callable_tool):
                continue
            definitions.append(self._tool_definition(name, definition, callable_tool))
            callables[name] = callable_tool
        return definitions, callables

    def _tool_definition(
        self, name: str, definition: Any, callable_tool: Callable[..., Any]
    ) -> dict[str, Any]:
        """Create an Anthropic client-tool definition from an injected callable."""
        if isinstance(definition, dict):
            schema = definition.get("input_schema") or definition.get("parameters")
            description = definition.get("description")
        else:
            schema = None
            description = None
        if not isinstance(schema, dict):
            schema = self._schema_from_callable(callable_tool)
        return {
            "name": name,
            "description": description or inspect.getdoc(callable_tool) or f"Run {name}.",
            "input_schema": schema,
        }

    def _schema_from_callable(self, callable_tool: Callable[..., Any]) -> dict[str, Any]:
        """Generate a conservative JSON-schema object from a callable signature."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter in inspect.signature(callable_tool).parameters.values():
            if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
                continue
            properties[parameter.name] = self._schema_for_annotation(parameter.annotation)
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def _schema_for_annotation(self, annotation: Any) -> dict[str, str]:
        """Map common Python annotations to the JSON-schema types Anthropic accepts."""
        mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
        if annotation in mapping:
            return {"type": mapping[annotation]}
        if annotation in {list, tuple, set}:
            return {"type": "array"}
        if annotation is dict:
            return {"type": "object"}
        return {"type": "string"}

    def _translate_tool_choice(self, tool_choice: Any) -> dict[str, Any]:
        """Translate OpenAI-style tool_choice values to Anthropic's documented shape."""
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                return {"type": "none"}
            return {"type": "any" if tool_choice == "required" else "auto"}
        if not isinstance(tool_choice, dict):
            return {"type": "auto"}
        if tool_choice.get("type") == "function":
            function = tool_choice.get("function", {})
            if function.get("name"):
                return {"type": "tool", "name": function["name"]}
        if tool_choice.get("type") in {"auto", "any", "none", "tool"}:
            translated = {"type": tool_choice["type"]}
            if tool_choice.get("name"):
                translated["name"] = tool_choice["name"]
            return translated
        return {"type": "auto"}

    async def _run_non_streaming(
        self,
        payload: dict[str, Any],
        tool_registry: ToolRegistry,
        event_emitter: EventEmitter | None,
    ) -> str:
        """Run Messages requests until Claude returns a final response or tool limit is reached."""
        async with self._client() as client:
            for tool_round in range(self.valves.MAX_TOOL_ROUNDS + 1):
                response = await self._create_message(client, payload)
                tool_blocks = [
                    block
                    for block in response.get("content", [])
                    if block.get("type") == "tool_use"
                ]
                if not tool_blocks:
                    return self._format_response(response)
                if not tool_registry:
                    return (
                        "Error: Claude requested a tool, but no Open WebUI tool registry "
                        "was injected."
                    )
                if tool_round >= self.valves.MAX_TOOL_ROUNDS:
                    return "Error: Anthropic tool-call limit reached."

                payload["messages"].append({"role": "assistant", "content": response["content"]})
                results = await self._execute_tools(tool_blocks, tool_registry, event_emitter)
                payload["messages"].append({"role": "user", "content": results})
                self._allow_final_response_after_tool_use(payload)
        return "Error: Anthropic tool-call loop ended unexpectedly."

    async def _create_message(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a message with bounded retries for transient upstream failures."""
        for attempt in range(self.valves.RETRY_COUNT + 1):
            try:
                response = await client.post(
                    self._api_url("/v1/messages"), headers=self._headers(), json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status not in {429, 500, 502, 503, 504} or attempt >= self.valves.RETRY_COUNT:
                    raise
            except httpx.TransportError:
                if attempt >= self.valves.RETRY_COUNT:
                    raise
            await asyncio.sleep(min(2**attempt, 8))
        raise RuntimeError("Anthropic retry loop ended unexpectedly.")

    async def _execute_tools(
        self,
        tool_blocks: list[dict[str, Any]],
        registry: ToolRegistry,
        event_emitter: EventEmitter | None,
    ) -> list[dict[str, Any]]:
        """Execute Claude client-tool calls using Open WebUI's injected callable registry."""
        results: list[dict[str, Any]] = []
        for block in tool_blocks:
            name = block.get("name", "")
            tool_use_id = block.get("id", "")
            callable_tool = registry.get(name)
            if not callable_tool:
                results.append(self._tool_result_block(tool_use_id, f"Unknown tool: {name}", True))
                continue
            await self._emit_status(event_emitter, f"Running tool: {name}", done=False)
            try:
                arguments = block.get("input", {})
                if inspect.iscoroutinefunction(callable_tool):
                    output = callable_tool(**arguments)
                else:
                    output = await asyncio.to_thread(callable_tool, **arguments)
                if inspect.isawaitable(output):
                    output = await asyncio.wait_for(
                        output, timeout=self.valves.TOOL_TIMEOUT_SECONDS
                    )
                results.append(self._tool_result_block(tool_use_id, output))
            except TimeoutError:
                logger.warning(
                    "Anthropic tool %s timed out after %.1f seconds",
                    name,
                    self.valves.TOOL_TIMEOUT_SECONDS,
                )
                results.append(
                    self._tool_result_block(
                        tool_use_id,
                        f"Tool timed out after {self.valves.TOOL_TIMEOUT_SECONDS:g} seconds.",
                        True,
                    )
                )
            except Exception as error:
                logger.exception("Anthropic tool %s failed", name)
                results.append(self._tool_result_block(tool_use_id, str(error), True))
        return results

    def _format_response(self, response: dict[str, Any]) -> str:
        """Render text and optional summarized thinking from a completed Message response."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif self.valves.DISPLAY_THINKING and block.get("type") == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif self.valves.DISPLAY_THINKING and block.get("type") == "redacted_thinking":
                thinking_parts.append("[Redacted thinking content]")
        thinking = "".join(thinking_parts)
        text = "".join(text_parts)
        return f"<think>{thinking}</think>{text}" if thinking else text

    def _allow_final_response_after_tool_use(self, payload: dict[str, Any]) -> None:
        """Release a required initial tool choice so Claude can answer after tool results."""
        if payload.get("tool_choice", {}).get("type") in {"any", "tool"}:
            payload["tool_choice"] = {"type": "auto"}

    def _stream_reasoning_delta(self, model: str, reasoning: str) -> str:
        """Return an OpenAI-shaped SSE reasoning delta understood by current Open WebUI."""
        return "data: " + json.dumps(
            {
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": reasoning},
                        "finish_reason": None,
                    }
                ],
            }
        )

    async def _stream_with_status(
        self,
        payload: dict[str, Any],
        tool_registry: ToolRegistry,
        event_emitter: EventEmitter | None,
    ) -> AsyncIterator[str]:
        """Stream text while completing any injected native-tool rounds."""
        await self._emit_status(event_emitter, "Contacting Anthropic…", done=False)
        try:
            async for chunk in self._stream_messages(payload, tool_registry, event_emitter):
                yield chunk
        except httpx.HTTPError as error:
            logger.exception("Anthropic stream failed")
            yield f"Error: Anthropic request failed: {self._describe_http_error(error)}"
        except Exception as error:
            logger.exception("Unexpected Anthropic stream failure")
            yield f"Error: {error}"
        finally:
            await self._emit_status(event_emitter, "Anthropic request complete.", done=True)

    async def _stream_messages(
        self,
        payload: dict[str, Any],
        tool_registry: ToolRegistry,
        event_emitter: EventEmitter | None,
    ) -> AsyncIterator[str]:
        """Consume Anthropic SSE events and continue after streamed client tool calls."""
        async with self._client() as client:
            for tool_round in range(self.valves.MAX_TOOL_ROUNDS + 1):
                blocks: dict[int, dict[str, Any]] = {}
                stop_reason = ""
                async with client.stream(
                    "POST", self._api_url("/v1/messages"), headers=self._headers(), json=payload
                ) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        # httpx does not expose .json() or .text for streamed responses until
                        # their body has been read. Do that before rendering the provider error.
                        await response.aread()
                        yield f"Error: Anthropic stream error: {self._describe_http_error(error)}"
                        return
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            logger.warning("Ignoring malformed Anthropic SSE data.")
                            continue
                        event_type = event.get("type")
                        if event_type == "error":
                            error = event.get("error", {})
                            message = error.get("message", "unknown error")
                            yield f"Error: Anthropic stream error: {message}"
                            return
                        if event_type == "content_block_start":
                            index = int(event.get("index", 0))
                            block = dict(event.get("content_block", {}))
                            blocks[index] = block
                            continue
                        if event_type == "content_block_delta":
                            index = int(event.get("index", 0))
                            block = blocks.setdefault(index, {})
                            delta = event.get("delta", {})
                            delta_type = delta.get("type")
                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                block["text"] = block.get("text", "") + text
                                yield text
                            elif delta_type == "thinking_delta":
                                thinking = delta.get("thinking", "")
                                block["thinking"] = block.get("thinking", "") + thinking
                                if self.valves.DISPLAY_THINKING:
                                    yield self._stream_reasoning_delta(payload["model"], thinking)
                            elif delta_type == "signature_delta":
                                block["signature"] = delta.get("signature", "")
                            elif delta_type == "input_json_delta":
                                block["_input_json"] = block.get("_input_json", "") + delta.get(
                                    "partial_json", ""
                                )
                            continue
                        if event_type == "content_block_stop":
                            index = int(event.get("index", 0))
                            block = blocks.get(index, {})
                            if block.get("type") == "tool_use":
                                raw_input = block.pop("_input_json", "")
                                try:
                                    block["input"] = json.loads(raw_input) if raw_input else {}
                                except json.JSONDecodeError:
                                    block["input"] = {}
                            continue
                        if event_type == "message_delta":
                            stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
                        if event_type == "message_stop":
                            break

                content = [blocks[index] for index in sorted(blocks)]
                tool_blocks = [block for block in content if block.get("type") == "tool_use"]
                if not tool_blocks or stop_reason != "tool_use":
                    return
                if not tool_registry:
                    yield (
                        "Error: Claude requested a tool, but no Open WebUI tool registry "
                        "was injected."
                    )
                    return
                if tool_round >= self.valves.MAX_TOOL_ROUNDS:
                    yield "Error: Anthropic tool-call limit reached."
                    return
                payload["messages"].append({"role": "assistant", "content": content})
                results = await self._execute_tools(tool_blocks, tool_registry, event_emitter)
                payload["messages"].append({"role": "user", "content": results})
                self._allow_final_response_after_tool_use(payload)

    async def _emit_status(
        self, emitter: EventEmitter | None, description: str, done: bool
    ) -> None:
        """Emit a status event when Open WebUI supplied an event emitter."""
        if emitter is not None:
            await emitter({"type": "status", "data": {"description": description, "done": done}})

    def _describe_http_error(self, error: httpx.HTTPError) -> str:
        """Return an actionable error message including Anthropic's request ID when available."""
        if isinstance(error, httpx.HTTPStatusError):
            response = error.response
            request_id = response.headers.get("request-id", "unknown")
            try:
                message = response.json().get("error", {}).get("message", response.text)
            except (json.JSONDecodeError, ValueError):
                message = response.text
            return f"HTTP {response.status_code} (request_id={request_id}): {message}"
        return str(error)
