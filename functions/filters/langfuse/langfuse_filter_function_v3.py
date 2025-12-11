"""
title: Langfuse Filter Function v3
author: YetheSamartaka
date: 2025-12-10
version: 1.2.0
license: MIT
description: A filter function that uses Langfuse v3.
required_open_webui_version: 0.6.41
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
        self._set_langfuse()

    def log(self, message: str, suppress_repeats: bool = False) -> None:
        if not self.valves.debug:
            return
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

    def _strip_debug_from_string(self, value: str) -> str:
        if "[DEBUG]" not in value:
            return value
        lines = value.splitlines()
        cleaned_lines = [line for line in lines if "[DEBUG]" not in line]
        return "\n".join(cleaned_lines).strip()

    def _sanitize_debug_content(self, obj: Any) -> Any:
        if self.valves.debug:
            return obj
        if isinstance(obj, dict):
            return {k: self._sanitize_debug_content(v) for k, v in obj.items()}
        if isinstance(obj, list):
            sanitized_list = [self._sanitize_debug_content(v) for v in obj]
            return [v for v in sanitized_list if v not in ("", None, [], {})]
        if isinstance(obj, str):
            return self._strip_debug_from_string(obj)
        return obj

    def _build_trace_metadata(
        self, metadata: dict[str, Any], user_email: str | None, chat_id: str
    ) -> dict[str, Any]:
        base_metadata: dict[str, Any] = {
            **metadata,
            "user_id": user_email,
            "session_id": chat_id,
            "interface": "open-webui",
        }
        return self._sanitize_debug_content(base_metadata)

    def _build_safe_input(
        self, body: dict[str, Any], trace_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        safe_body: dict[str, Any] = {
            "model": body.get("model"),
            "messages": body.get("messages"),
        }
        safe_metadata = self._sanitize_debug_content(trace_metadata)
        safe_body["metadata"] = safe_metadata
        return safe_body

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
            msg = str(auth_error)
            if (
                "401" in msg
                or "unauthorized" in msg.lower()
                or "credentials" in msg.lower()
            ):
                self.log(f"Langfuse credentials incorrect: {auth_error}")
                self.langfuse = None
            else:
                self.log(f"Langfuse initialization error: {auth_error}")
                self.langfuse = None

    def _build_tags(self, task_name: str) -> list[str]:
        tags_list: list[str] = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name not in ["user_response", "llm_response"]:
                tags_list.append(task_name)
        return tags_list

    def _extract_model_info(
        self, body: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        self.log("Starting model info extraction")

        model_id: str | None = None
        model_name: str | None = None

        model_item = body.get("model_item")
        self.log(f"Model item block: {model_item}")
        if isinstance(model_item, dict):
            raw_item_id = model_item.get("id")
            raw_item_name = model_item.get("name")
            self.log(f"Model item name: {raw_item_name}, id: {raw_item_id}")
            if isinstance(raw_item_id, str) and raw_item_id:
                model_id = raw_item_id
                self.log(f"Model ID set from model_item: {model_id}")
            if isinstance(raw_item_name, str) and raw_item_name:
                model_name = raw_item_name
                self.log(f"Model name set from model_item: {model_name}")

        raw_model = body.get("model")
        self.log(f"Raw model at root level: {raw_model}")
        if isinstance(raw_model, str) and raw_model and not model_id:
            model_id = raw_model
            self.log(f"Model ID set from root level: {model_id}")

        metadata = body.get("metadata") or {}
        meta_model = metadata.get("model")
        self.log(f"Metadata model block: {meta_model}")

        if isinstance(meta_model, dict):
            raw_name = meta_model.get("name")
            raw_id = meta_model.get("id")
            self.log(f"Metadata model name: {raw_name}, id: {raw_id}")

            if isinstance(raw_name, str) and raw_name and not model_name:
                model_name = raw_name
                self.log(f"Model name set from metadata: {model_name}")

            if isinstance(raw_id, str) and raw_id and not model_id:
                model_id = raw_id
                self.log(f"Model ID set from metadata: {model_id}")
                if not model_name:
                    model_name = raw_id
                    self.log(
                        f"Metadata missing name, falling back to ID for name: {model_name}"
                    )

        if not model_id or not model_name:
            task_body = metadata.get("task_body")
            self.log(f"Looking into metadata.task_body for model info: {task_body}")

            if isinstance(task_body, dict):
                tb_model = task_body.get("model")
                self.log(f"task_body.model: {tb_model}")

                if isinstance(tb_model, str) and tb_model:
                    if not model_id:
                        model_id = tb_model
                        self.log(f"Model ID set from task_body: {model_id}")
                    if not model_name:
                        model_name = tb_model
                        self.log(f"Model name set from task_body: {model_name}")
                elif isinstance(tb_model, dict):
                    tb_id = tb_model.get("id")
                    tb_name = tb_model.get("name")
                    self.log(
                        f"task_body.model.name: {tb_name}, task_body.model.id: {tb_id}"
                    )

                    if isinstance(tb_id, str) and tb_id and not model_id:
                        model_id = tb_id
                        self.log(f"Model ID set from task_body dict: {model_id}")
                    if isinstance(tb_name, str) and tb_name and not model_name:
                        model_name = tb_name
                        self.log(f"Model name set from task_body dict: {model_name}")

        self.log(f"Finished extraction model_id: {model_id}, model_name: {model_name}")
        return model_id, model_name

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

        trace_metadata = self._build_trace_metadata(metadata, user_email, chat_id)
        safe_input = self._build_safe_input(body, trace_metadata)

        if chat_id not in self.chat_traces:
            self.log(f"Creating new trace for chat_id: {chat_id}")
            try:
                trace = self.langfuse.start_span(
                    name=f"chat:{chat_id}", input=safe_input, metadata=trace_metadata
                )
                trace.update_trace(
                    user_id=user_email,
                    session_id=chat_id,
                    tags=tags_list if tags_list else None,
                    input=safe_input,
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
            event_metadata = self._sanitize_debug_content(event_metadata)
            trace.event(
                name=f"user_input:{str(uuid.uuid4())}",
                metadata=event_metadata,
                input=body["messages"],
            )
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
        self.log(f"Task name: {task_name}")
        tags_list = self._build_tags(task_name)

        if not chat_id or chat_id not in self.chat_traces:
            self.log(
                f"[WARNING] No matching trace found for chat_id: {chat_id}, attempting to re-register."
            )
            return await self.inlet(body, __event_emitter__, __user__)

        messages: list[dict[str, Any]] = body.get("messages") or []

        assistant_index: int | None = None
        assistant_message_obj: dict[str, Any] | None = None

        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant_index = i
                assistant_message_obj = message
                break

        assistant_message_text: str | None = None
        if assistant_message_obj is not None:
            content = assistant_message_obj.get("content")
            if isinstance(content, str):
                assistant_message_text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for c in content:
                    if isinstance(c, dict):
                        v = c.get("text") or c.get("content")
                        if isinstance(v, str):
                            parts.append(v)
                if parts:
                    assistant_message_text = "\n".join(parts)

        if assistant_index is not None:
            prompt_messages = messages[:assistant_index]
        else:
            prompt_messages = messages

        usage: dict[str, Any] | None = None
        if assistant_message_obj is not None:
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
                    usage = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "unit": "TOKENS",
                    }
                    self.log(f"Usage data extracted: {usage}")

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
        complete_trace_metadata = self._sanitize_debug_content(complete_trace_metadata)

        trace.update_trace(
            output=assistant_message_text,
            metadata=complete_trace_metadata,
            tags=tags_list if tags_list else None,
        )

        model_id, model_name = self._extract_model_info(body)

        self.log("Beginning model selection block")
        self.log(f"Raw extracted model_id: {model_id}, model_name: {model_name}")

        if self.valves.use_model_name_instead_of_id_for_generation:
            model_value = model_name or model_id or "unknown"
            self.log("Using model name for generation")
        else:
            model_value = model_id or model_name or "unknown"
            self.log("Using model ID for generation")

        self.log(f"Final model_value selected: {model_value}")

        metadata["model_id"] = model_id
        metadata["model_name"] = model_name or model_id

        self.log(
            f"Metadata updated with model_id: {metadata['model_id']}, model_name: {metadata['model_name']}"
        )

        try:
            generation_metadata = {
                **complete_trace_metadata,
                "type": "llm_response",
                "model_id": model_id,
                "model_name": model_name or model_id,
                "generation_id": str(uuid.uuid4()),
            }
            generation_metadata = self._sanitize_debug_content(generation_metadata)

            generation = trace.start_generation(
                name=f"llm_response:{str(uuid.uuid4())}",
                model=model_value,
                input=prompt_messages,
                output=assistant_message_text,
                metadata=generation_metadata,
            )
            if usage:
                generation.update(usage=usage)
            generation.end()
            trace.end()
            self.log(f"LLM generation completed for chat_id: {chat_id}")
        except Exception as e:
            self.log(f"Failed to create LLM generation: {e}")

        try:
            if self.langfuse:
                self.langfuse.flush()
                self.log("Langfuse data flushed")
        except Exception as e:
            self.log(f"Failed to flush Langfuse data: {e}")

        return body
