"""
title: Langfuse Filter Function v3
author: YetheSamartaka
date: 2025-10-27
version: 1.1.0
license: MIT
description: A filter function that uses Langfuse v3.
required_open_webui_version: 0.6.32
requirements: langfuse>=3.0.0.
Other notes: For local instance of Langfuse, set Open WebUI ENV var: OTEL_EXPORTER_OTLP_ENDPOINT=host:port
"""

import os
import uuid
from typing import Any

from langfuse import Langfuse
from pydantic import BaseModel


def _get_last_assistant_message_obj(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message
    return {}


def _get_last_assistant_message(messages: list[dict[str, Any]]) -> str | None:
    obj = _get_last_assistant_message_obj(messages)
    content = obj.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict):
                v = c.get("text") or c.get("content")
                if isinstance(v, str):
                    parts.append(v)
        return "\n".join(parts) if parts else None
    return None


class Filter:
    class Valves(BaseModel):
        secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here")
        public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here")
        host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        insert_tags: bool = True
        use_model_name_instead_of_id_for_generation: bool = (
            os.getenv("USE_MODEL_NAME", "false").lower() == "true"
        )
        debug: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter"
        self.valves = self.Valves()
        self.langfuse: Langfuse | None = None
        self.chat_traces: dict[str, Any] = {}
        self.suppressed_logs: set[str] = set()
        self.model_names: dict[str, dict[str, str]] = {}
        self._set_langfuse()

    def log(self, message: str, suppress_repeats: bool = False) -> None:
        if self.valves.debug:
            if suppress_repeats:
                if message in self.suppressed_logs:
                    return
                self.suppressed_logs.add(message)
            print(f"[DEBUG] {message}")

    async def on_valves_updated(self) -> None:
        self.log("Valves updated, resetting Langfuse client.")
        self._set_langfuse()

    def _normalize_host(self, raw: str) -> str:
        v = (raw or "").strip().rstrip("/")
        if not v:
            return "https://cloud.langfuse.com"
        if v.startswith("http://") or v.startswith("https://"):
            return v
        return f"https://{v}"

    def _set_langfuse(self) -> None:
        try:
            self.log(f"Initializing Langfuse with host: {self.valves.host}")
            self.log(
                f"Secret key set: {'Yes' if self.valves.secret_key and self.valves.secret_key != 'your-secret-key-here' else 'No'}"
            )
            self.log(
                f"Public key set: {'Yes' if self.valves.public_key and self.valves.public_key != 'your-public-key-here' else 'No'}"
            )
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self._normalize_host(self.valves.host),
                debug=self.valves.debug,
            )
            try:
                self.langfuse.auth_check()
                self.log(
                    f"Langfuse client initialized and authenticated successfully. Connected to host: {self.valves.host}"
                )
            except Exception as e:
                self.log(f"Auth check failed (non-critical, skipping): {e}")
        except Exception as auth_error:
            if (
                "401" in str(auth_error)
                or "unauthorized" in str(auth_error).lower()
                or "credentials" in str(auth_error).lower()
            ):
                self.log(f"Langfuse credentials incorrect: {auth_error}")
                self.langfuse = None
                return
        except Exception as e:
            self.log(f"Langfuse initialization error: {e}")
            self.langfuse = None

    def _build_tags(self, task_name: str) -> list[str]:
        tags_list: list[str] = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name not in ["user_response", "llm_response"]:
                tags_list.append(task_name)
        return tags_list

    async def inlet(
        self,
        body: dict[str, Any],
        __event_emitter__,
        __user__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.log("Langfuse Filter INLET called")
        self._set_langfuse()
        if not self.langfuse:
            self.log("[WARNING] Langfuse client not initialized - Skipped")
            return body
        self.log(f"Inlet function called with body: {body} and user: {__user__}")
        metadata = body.get("metadata", {}) or {}
        chat_id = metadata.get("chat_id", str(uuid.uuid4()))
        if chat_id == "local":
            session_id = metadata.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        metadata["chat_id"] = chat_id
        body["metadata"] = metadata
        model_info = metadata.get("model", {}) or {}
        model_id = body.get("model")
        if chat_id not in self.model_names:
            self.model_names[chat_id] = {
                "id": str(model_id) if model_id is not None else ""
            }
        else:
            self.model_names[chat_id]["id"] = (
                str(model_id) if model_id is not None else ""
            )
        if isinstance(model_info, dict) and "name" in model_info:
            self.model_names[chat_id]["name"] = str(model_info["name"])
            self.log(
                f"Stored model info - name: '{model_info.get('name')}', id: '{model_id}' for chat_id: {chat_id}"
            )
        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]
        if missing_keys:
            error_message = (
                f"Error: Missing keys in the request body: {', '.join(missing_keys)}"
            )
            self.log(error_message)
            raise ValueError(error_message)
        user_email = __user__.get("email") if __user__ else None
        task_name = metadata.get("task", "user_response")
        tags_list = self._build_tags(task_name)
        if chat_id not in self.chat_traces:
            self.log(f"Creating new trace for chat_id: {chat_id}")
            try:
                trace_metadata = {
                    **metadata,
                    "user_id": user_email,
                    "session_id": chat_id,
                    "interface": "open-webui",
                }
                trace = self.langfuse.start_span(
                    name=f"chat:{chat_id}", input=body, metadata=trace_metadata
                )
                trace.update_trace(
                    user_id=user_email,
                    session_id=chat_id,
                    tags=tags_list if tags_list else None,
                    input=body,
                    metadata=trace_metadata,
                )
                self.chat_traces[chat_id] = trace
                self.log(f"Successfully created trace for chat_id: {chat_id}")
            except Exception as e:
                self.log(f"Failed to create trace: {e}")
                return body
        else:
            trace = self.chat_traces[chat_id]
            self.log(f"Reusing existing trace for chat_id: {chat_id}")
            trace_metadata = {
                **metadata,
                "user_id": user_email,
                "session_id": chat_id,
                "interface": "open-webui",
            }
            trace.update_trace(
                tags=tags_list if tags_list else None, metadata=trace_metadata
            )
        metadata["type"] = task_name
        metadata["interface"] = "open-webui"
        try:
            trace = self.chat_traces[chat_id]
            event_metadata = {
                **metadata,
                "type": "user_input",
                "interface": "open-webui",
                "user_id": user_email,
                "session_id": chat_id,
                "event_id": str(uuid.uuid4()),
            }
            event_span = trace.start_span(
                name=f"user_input:{str(uuid.uuid4())}",
                metadata=event_metadata,
                input=body["messages"],
            )
            event_span.end()
            self.log(f"User input event logged for chat_id: {chat_id}")
        except Exception as e:
            self.log(f"Failed to log user input event: {e}")
        return body

    async def outlet(
        self,
        body: dict[str, Any],
        __event_emitter__,
        __user__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.log("Langfuse Filter OUTLET called")
        self._set_langfuse()
        if not self.langfuse:
            self.log("[WARNING] Langfuse client not initialized - Skipped")
            return body
        self.log(f"Outlet function called with body: {body}")
        chat_id: str | None = body.get("chat_id")
        if chat_id == "local":
            session_id = body.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        metadata = body.get("metadata", {}) or {}
        task_name = metadata.get("task", "llm_response")
        tags_list = self._build_tags(task_name)
        if not chat_id or chat_id not in self.chat_traces:
            self.log(
                f"[WARNING] No matching trace found for chat_id: {chat_id}, attempting to re-register."
            )
            return await self.inlet(body, __event_emitter__, __user__)
        assistant_message_text = _get_last_assistant_message(body["messages"])
        assistant_message_obj = _get_last_assistant_message_obj(body["messages"])
        usage_details: dict[str, Any] | None = None
        if assistant_message_obj:
            info = assistant_message_obj.get("usage", {}) or {}
            if isinstance(info, dict):
                input_tokens = (
                    info.get("prompt_eval_count")
                    or info.get("prompt_tokens")
                    or info.get("input_tokens")
                )
                output_tokens = (
                    info.get("eval_count")
                    or info.get("completion_tokens")
                    or info.get("output_tokens")
                )
                if input_tokens is not None and output_tokens is not None:
                    usage_details = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "unit": "TOKENS",
                    }
                    self.log(f"Usage data extracted: {usage_details}")
        trace = self.chat_traces[chat_id]
        metadata["type"] = task_name
        metadata["interface"] = "open-webui"
        complete_trace_metadata = {
            **metadata,
            "user_id": (__user__.get("email") if __user__ else None),
            "session_id": chat_id,
            "interface": "open-webui",
            "task": task_name,
        }
        model_id = self.model_names.get(chat_id, {}).get("id", body.get("model"))
        model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
        model_value = (
            model_name
            if self.valves.use_model_name_instead_of_id_for_generation
            else model_id
        )
        metadata["model_id"] = model_id
        metadata["model_name"] = model_name
        try:
            generation_metadata = {
                **complete_trace_metadata,
                "type": "llm_response",
                "model_id": model_id,
                "model_name": model_name,
                "generation_id": str(uuid.uuid4()),
            }
            generation = trace.start_generation(
                name=f"llm_response:{str(uuid.uuid4())}",
                model=model_value,
                input=body["messages"],
                output=assistant_message_text,
                metadata=generation_metadata,
            )
            if usage_details:
                generation.update(usage_details=usage_details)
            generation.end()
            self.log(f"LLM generation completed for chat_id: {chat_id}")
        except Exception as e:
            self.log(f"Failed to create LLM generation: {e}")

        trace.update_trace(
            output=assistant_message_text,
            metadata=complete_trace_metadata,
            tags=tags_list if tags_list else None,
        )
        trace.end()

        try:
            if self.langfuse:
                self.langfuse.flush()
                self.log("Langfuse data flushed")
        except Exception as e:
            self.log(f"Failed to flush Langfuse data: {e}")
        return body
