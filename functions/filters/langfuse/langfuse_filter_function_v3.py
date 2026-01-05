"""
title: Langfuse Filter Function v3
author: YetheSamartaka
date: 2025-12-10
version: 1.4.0
license: MIT
description: A filter function that uses Langfuse v3.
required_open_webui_version: 0.6.41
requirements: langfuse>=3.0.0.
Other notes: For local instance of Langfuse, set Open WebUI ENV var: OTEL_EXPORTER_OTLP_ENDPOINT=host:port
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any

from langfuse import Langfuse
from pydantic import BaseModel, Field


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
        disable_debug_bodies: bool = Field(
            default=False,
            title="Disable debug bodies",
        )
        use_model_name_instead_of_id_for_generation: bool = (
            os.getenv("USE_MODEL_NAME", "false").lower() == "true"
        )
        debug: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter"
        self.valves = self.Valves()
        self.langfuse: Langfuse | None = None
        self.chat_trace_ids: dict[str, str] = {}
        self.chat_tags: dict[str, set[str]] = {}
        self.suppressed_logs: set[str] = set()
        self.session_to_chat_id: dict[str, str] = {}
        self._set_langfuse()

    def _one_line_preview(self, text: str, limit: int = 512) -> str:
        one_line = " ".join(text.split())
        return one_line[:limit]

    def _drop_keys_recursive(self, obj: Any, drop: set[str]) -> Any:
        if isinstance(obj, dict):
            out: dict[Any, Any] = {}
            for k, v in obj.items():
                if isinstance(k, str) and k in drop:
                    continue
                out[k] = self._drop_keys_recursive(v, drop)
            return out
        if isinstance(obj, list):
            return [self._drop_keys_recursive(v, drop) for v in obj]
        return obj

    def _scrub_for_debug(self, obj: Any) -> Any:
        return self._drop_keys_recursive(
            obj, {"knowledge", "profile_image_url", "files"}
        )

    def _extract_json_block(self, text: str) -> Any | None:
        for open_char, close_char in (("{", "}"), ("[", "]")):
            start = text.find(open_char)
            if start == -1:
                continue
            end = text.rfind(close_char)
            if end == -1 or end < start:
                continue
            raw = text[start : end + 1].strip()
            try:
                return json.loads(raw)
            except Exception:
                continue
        return None

    def _scrub_transformed_responses_body(self, message: str) -> str:
        prefix, _, rest = message.partition("Transformed ResponsesBody:")
        parsed = self._extract_json_block(rest)
        if not isinstance(parsed, dict):
            return f"{prefix}Transformed ResponsesBody: <omitted>"

        scrubbed: dict[str, Any] = {}
        for key in ("model", "max_output_tokens", "max_tool_calls", "stream"):
            if key in parsed:
                scrubbed[key] = parsed.get(key)

        return f"{prefix}Transformed ResponsesBody: {scrubbed}"

    def _scrub_event_data(self, message: str) -> str:
        prefix, _, rest = message.partition("Event data:")
        parsed = self._extract_json_block(rest)
        if not isinstance(parsed, dict):
            return f"{prefix}Event data: <omitted>"

        parsed_out: dict[str, Any] = dict(parsed)
        response = parsed_out.get("response")
        if isinstance(response, dict):
            response_out = dict(response)
            response_out.pop("instructions", None)
            response_out.pop("output", None)
            parsed_out["response"] = response_out

        return f"{prefix}Event data: {parsed_out}"

    def _scrub_inlet_outlet_body_line(self, message: str, kind: str) -> str:
        prefix, _, rest = message.partition(f"{kind} function called with body:")
        parsed = self._extract_json_block(rest)
        if not isinstance(parsed, dict):
            return f"{prefix}{kind} function called with body summary: <omitted>"
        return f"{prefix}{kind} function called with body summary: {self._debug_body_summary(parsed)}"

    def _scrub_debug_message(self, message: str) -> str | None:
        if "response.output_text.delta" in message:
            return None

        if self.valves.disable_debug_bodies:
            if "Transformed ResponsesBody:" in message:
                return self._scrub_transformed_responses_body(message)
            if "Event data:" in message:
                return self._scrub_event_data(message)
            if "Inlet function called with body:" in message:
                return self._scrub_inlet_outlet_body_line(message, "Inlet")
            if "Outlet function called with body:" in message:
                return self._scrub_inlet_outlet_body_line(message, "Outlet")

        return message

    def _debug_prefix(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"{ts} | DEBUG    | "

    def log(self, message: str, suppress_repeats: bool = False) -> None:
        if not self.valves.debug:
            return

        scrubbed = self._scrub_debug_message(message)
        if scrubbed is None:
            return

        if suppress_repeats:
            if scrubbed in self.suppressed_logs:
                return
            self.suppressed_logs.add(scrubbed)

        print(f"{self._debug_prefix()}{scrubbed}")

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
        safe_body["metadata"] = self._sanitize_debug_content(trace_metadata)
        return safe_body

    def _set_langfuse(self) -> None:
        try:
            self.log(f"Initializing Langfuse with host: {self.valves.host}")
            self.log(
                "Secret key set: "
                + (
                    "Yes"
                    if self.valves.secret_key
                    and self.valves.secret_key != "your-secret-key-here"
                    else "No"
                )
            )
            self.log(
                "Public key set: "
                + (
                    "Yes"
                    if self.valves.public_key
                    and self.valves.public_key != "your-public-key-here"
                    else "No"
                )
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

    def _extract_model_info(
        self, body: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        self.log("Starting model info extraction")

        model_id: str | None = None
        model_name: str | None = None

        model_item = body.get("model_item")
        self.log(f"Model item block: {self._scrub_for_debug(model_item)}")
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

    def _extract_tags_from_assistant_message(self, message: str | None) -> list[str]:
        preview = (
            self._one_line_preview(message, 128)
            if isinstance(message, str) and message
            else None
        )
        self.log(
            f"Attempting to extract tags from assistant message content: {preview}",
            suppress_repeats=True,
        )
        if not message:
            return []
        text = message.strip()
        try:
            parsed = json.loads(text)
        except Exception as e:
            self.log(
                f"Failed to parse assistant message as JSON for tags: {e}",
                suppress_repeats=True,
            )
            return []
        if not isinstance(parsed, dict):
            return []
        raw_tags = parsed.get("tags")
        if not isinstance(raw_tags, list):
            return []
        tags: list[str] = []
        for t in raw_tags:
            if isinstance(t, str):
                tags.append(t)
        self.log(f"Extracted tags from assistant message: {tags}")
        return tags

    def _get_persistent_tags_for_chat(self, chat_id: str | None) -> list[str]:
        if not chat_id:
            return []
        tags_set = self.chat_tags.get(chat_id)
        if not tags_set:
            return []
        tags_list = sorted(tags_set)
        self.log(f"Loaded persistent tags for chat_id {chat_id}: {tags_list}")
        return tags_list

    def _update_persistent_tags_for_chat(
        self, chat_id: str | None, tags: list[str]
    ) -> None:
        if not chat_id:
            return
        if chat_id not in self.chat_tags:
            self.chat_tags[chat_id] = set()
        for t in tags:
            if isinstance(t, str):
                self.chat_tags[chat_id].add(t)
        self.log(
            f"Updated persistent tags for chat_id {chat_id}: {sorted(self.chat_tags[chat_id])}"
        )

    def _get_or_create_trace_id(self, chat_id: str) -> str:
        cached = self.chat_trace_ids.get(chat_id)
        if cached:
            return cached
        trace_id = Langfuse.create_trace_id(seed=chat_id)
        self.chat_trace_ids[chat_id] = trace_id
        self.log(f"Deterministic trace_id for chat_id {chat_id}: {trace_id}")
        return trace_id

    def _extract_session_id(self, body: dict[str, Any]) -> str | None:
        metadata = body.get("metadata", {}) or {}
        raw_session_id = metadata.get("session_id") or body.get("session_id")
        if isinstance(raw_session_id, str) and raw_session_id:
            return raw_session_id
        return None

    def _extract_chat_id(self, body: dict[str, Any]) -> str:
        metadata = body.get("metadata", {}) or {}
        raw_chat_id = body.get("chat_id") or metadata.get("chat_id")
        session_id = self._extract_session_id(body)

        if isinstance(raw_chat_id, str) and raw_chat_id.startswith("task-"):
            task_body = metadata.get("task_body")
            if isinstance(task_body, dict):
                tb_chat_id = task_body.get("chat_id")
                if (
                    isinstance(tb_chat_id, str)
                    and tb_chat_id
                    and not tb_chat_id.startswith("task-")
                ):
                    raw_chat_id = tb_chat_id

            if session_id:
                mapped = self.session_to_chat_id.get(session_id)
                if mapped:
                    return mapped

        chat_id: str | None = (
            raw_chat_id if isinstance(raw_chat_id, str) and raw_chat_id else None
        )

        if chat_id == "local":
            session_id_for_local = metadata.get("session_id") or body.get("session_id")
            session_str = (
                session_id_for_local
                if isinstance(session_id_for_local, str) and session_id_for_local
                else str(uuid.uuid4())
            )
            chat_id = f"temporary-session-{session_str}"

        if not chat_id:
            session_id_for_missing = metadata.get("session_id") or body.get(
                "session_id"
            )
            if isinstance(session_id_for_missing, str) and session_id_for_missing:
                chat_id = f"temporary-session-{session_id_for_missing}"
            else:
                chat_id = str(uuid.uuid4())

        if session_id and not chat_id.startswith("task-"):
            self.session_to_chat_id[session_id] = chat_id

        return chat_id

    def _debug_body_summary(self, body: dict[str, Any]) -> dict[str, Any]:
        summarized: dict[str, Any] = {}
        for k, v in body.items():
            if k == "messages":
                summarized["messages_count"] = len(v) if isinstance(v, list) else None
                continue
            if k == "metadata":
                if isinstance(v, dict):
                    summarized["metadata_keys"] = sorted(
                        [mk for mk in v.keys() if isinstance(mk, str)]
                    )
                else:
                    summarized["metadata_keys"] = []
                continue
            if k == "model_item":
                summarized["model_item"] = self._scrub_for_debug(v)
                continue
            if k == "files":
                continue
            summarized[k] = self._scrub_for_debug(v)
        return summarized

    def _infer_task_name_from_assistant_message(
        self, message: str | None
    ) -> str | None:
        if not message:
            return None
        text = message.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        if "queries" in parsed:
            return "query_generation"
        if "title" in parsed:
            return "title_generation"
        if isinstance(parsed.get("tags"), list):
            return "tags_generation"
        return None

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

        user_scrubbed = (
            self._scrub_for_debug(__user__) if __user__ is not None else None
        )
        if self.valves.disable_debug_bodies:
            self.log(
                f"Inlet function called with body summary: {self._debug_body_summary(body)} and user: {user_scrubbed}"
            )
        else:
            self.log(
                f"Inlet function called with body: {self._scrub_for_debug(body)} and user: {user_scrubbed}"
            )

        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]
        if missing_keys:
            error_message = (
                f"Error: Missing keys in the request body: {', '.join(missing_keys)}"
            )
            self.log(error_message)
            raise ValueError(error_message)

        metadata = body.get("metadata", {}) or {}
        chat_id = self._extract_chat_id(body)

        user_email = __user__.get("email") if __user__ else None
        tags_list: list[str] = []

        self._update_persistent_tags_for_chat(chat_id, tags_list)
        _ = self._get_or_create_trace_id(chat_id)

        trace_metadata = self._build_trace_metadata(dict(metadata), user_email, chat_id)
        _ = self._build_safe_input(body, trace_metadata)

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

        if self.valves.disable_debug_bodies:
            self.log(
                f"Outlet function called with body summary: {self._debug_body_summary(body)}"
            )
        else:
            self.log(f"Outlet function called with body: {self._scrub_for_debug(body)}")

        chat_id = self._extract_chat_id(body)
        trace_id = self._get_or_create_trace_id(chat_id)

        metadata = body.get("metadata", {}) or {}

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

        task_name_raw = metadata.get("task")
        if isinstance(task_name_raw, str) and task_name_raw:
            task_name = task_name_raw
        else:
            task_name = (
                self._infer_task_name_from_assistant_message(assistant_message_text)
                or "llm_response"
            )

        tags_list: list[str] = []

        classification_tags = self._extract_tags_from_assistant_message(
            assistant_message_text
        )
        persistent_tags = self._get_persistent_tags_for_chat(chat_id)

        outlet_tags_raw = body.get("tags")
        metadata_tags_raw = metadata.get("tags")

        outlet_tags = outlet_tags_raw if isinstance(outlet_tags_raw, list) else []
        metadata_tags = metadata_tags_raw if isinstance(metadata_tags_raw, list) else []

        merged_tags: list[str] = []
        for tag in (
            tags_list
            + outlet_tags
            + metadata_tags
            + classification_tags
            + persistent_tags
        ):
            if isinstance(tag, str) and tag not in merged_tags:
                merged_tags.append(tag)

        if merged_tags:
            tags_list = merged_tags
            self._update_persistent_tags_for_chat(chat_id, merged_tags)

        prompt_messages = (
            messages[:assistant_index] if assistant_index is not None else messages
        )

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

        user_email = __user__.get("email") if __user__ else None
        trace_metadata = self._build_trace_metadata(dict(metadata), user_email, chat_id)
        safe_input = self._build_safe_input(body, trace_metadata)

        complete_trace_metadata = {
            **dict(metadata),
            "user_id": user_email,
            "session_id": chat_id,
            "interface": "open-webui",
            "task": task_name,
            "type": task_name,
        }
        complete_trace_metadata = self._sanitize_debug_content(complete_trace_metadata)

        model_id, model_name = self._extract_model_info(body)

        if self.valves.use_model_name_instead_of_id_for_generation:
            model_value = model_name or model_id or "unknown"
        else:
            model_value = model_id or model_name or "unknown"

        try:
            generation_metadata = {
                **complete_trace_metadata,
                "type": "llm_response",
                "model_id": model_id,
                "model_name": model_name or model_id,
                "generation_id": str(uuid.uuid4()),
            }
            generation_metadata = self._sanitize_debug_content(generation_metadata)

            generation = self.langfuse.start_generation(
                name=f"{task_name}:{str(uuid.uuid4())}",
                trace_context={"trace_id": trace_id},
                model=model_value,
                input=prompt_messages,
                output=assistant_message_text,
                metadata=generation_metadata,
            )

            try:
                generation.update_trace(
                    name=f"chat:{chat_id}",
                    user_id=user_email,
                    session_id=chat_id,
                    tags=tags_list if tags_list else None,
                    input=safe_input,
                    output=assistant_message_text,
                    metadata=complete_trace_metadata,
                )
            except Exception as e:
                self.log(
                    f"Failed to update trace via generation.update_trace (non-critical): {e}"
                )

            if usage:
                try:
                    generation.update(usage=usage)
                except Exception as e:
                    self.log(f"Failed to update generation usage (non-critical): {e}")

            generation.end()
        except Exception as e:
            self.log(f"Failed to create generation: {e}")

        try:
            if self.langfuse:
                self.langfuse.flush()
        except Exception as e:
            self.log(f"Failed to flush Langfuse data: {e}")

        return body
