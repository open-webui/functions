"""
title: Anthropic Manifold Pipe with Extended Thinking and Cache Control
authors: justinh-rahb, christian-taillon, jfbloom22, Mark Kazakov, Vincent, NIK-NUB, cache control added by Snav
author_url: https://github.com/jfbloom22
funding_url: https://github.com/open-webui
version: 0.5.0
required_open_webui_version: 0.3.17
license: MIT
description: An advanced manifold pipe for interacting with Anthropic's Claude models, featuring extended thinking support, cache control, beta features, and sophisticated model handling for Claude 4.5.
"""

import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator, Optional, Dict
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    CACHE_TTL = "1h"

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API Key")
        CLAUDE_USE_TEMPERATURE: bool = Field(
            default=True,
            description="For Claude 4.x models: Use temperature (True) or top_p (False). Claude 4.x models only support one parameter.",
        )
        BETA_FEATURES: str = Field(
            default="",
            description="Enable Anthropic Beta Features. e.g.: context-management-2025-06-27",
        )
        ENABLE_THINKING: bool = Field(
            default=True,
            description="Enable Claude's extended thinking capabilities (Claude 4.5 Sonnet with thinking model only)",
        )
        THINKING_BUDGET: int = Field(
            default=16000,
            description="Maximum number of tokens Claude can use for thinking (min: 1024, max: 32000)",
        )
        DISPLAY_THINKING: bool = Field(
            default=True, description="Display Claude's thinking process in the chat"
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves(
            **{
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
                "CLAUDE_USE_TEMPERATURE": True,  # Use temperature for Claude 4.x models
                "BETA_FEATURES": "",
                "ENABLE_THINKING": True,
                "THINKING_BUDGET": 16000,
                "DISPLAY_THINKING": True,
            }
        )
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image

        # Model cache
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_time: float = 0
        self._cache_ttl = int(os.getenv("ANTHROPIC_MODEL_CACHE_TTL", "600"))

    def get_anthropic_models_from_api(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        """
        Retrieve available Anthropic models from the API.
        Uses caching to reduce API calls.

        Args:
            force_refresh: Whether to force refreshing the model cache

        Returns:
            List of dictionaries containing model id and name.
        """
        # Check cache first
        current_time = time.time()
        if (
            not force_refresh
            and self._model_cache is not None
            and (current_time - self._model_cache_time) < self._cache_ttl
        ):
            return self._model_cache

        if not self.valves.ANTHROPIC_API_KEY:
            return [
                {
                    "id": "error",
                    "name": "ANTHROPIC_API_KEY is not set. Please update the API Key in the valves.",
                }
            ]

        try:
            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            
            response = requests.get(
                "https://api.anthropic.com/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")
            
            data = response.json()
            models = []
            
            for model in data.get("data", []):
                models.append({
                    "id": model["id"],
                    "name": model.get("display_name", model["id"]),
                })
            
            # Update cache
            self._model_cache = models
            self._model_cache_time = current_time
            
            return models
            
        except Exception as e:
            print(f"Error fetching Anthropic models: {e}")
            return [
                {"id": "error", "name": f"Could not fetch models from Anthropic: {str(e)}"}
            ]

    def get_anthropic_models(self) -> List[Dict[str, str]]:
        """
        Get Anthropic models from the API.
        """
        return self.get_anthropic_models_from_api()

    def _attach_cache_control(self, block: dict):
        """Attach cache control to a content block."""
        if not isinstance(block, dict):
            return block

        # Skip block types that cannot be cached directly per Anthropic docs
        if block.get("type") in {"thinking", "redacted_thinking"}:
            return block

        if not block.get("type"):
            block["type"] = "text"
            if "text" not in block:
                block["text"] = ""

        cache_control = dict(block.get("cache_control", {}))
        cache_control["type"] = "ephemeral"
        cache_control["ttl"] = self.CACHE_TTL
        block["cache_control"] = cache_control
        return block

    def _normalize_content_blocks(self, raw_content):
        """Normalize content into proper block format."""
        blocks = []

        if isinstance(raw_content, list):
            items = raw_content
        else:
            items = [raw_content]

        for item in items:
            if isinstance(item, dict) and item.get("type"):
                blocks.append(dict(item))
            elif isinstance(item, dict) and "content" in item:
                # Handle message-style dicts that still wrap content
                blocks.extend(self._normalize_content_blocks(item["content"]))
            elif item is not None:
                blocks.append({"type": "text", "text": str(item)})

        return blocks

    def _prepare_system_blocks(self, system_message):
        """Prepare system message with cache control."""
        if not system_message:
            return None

        # Open WebUI may hand us a raw message dict, list of blocks, or plain text
        content = (
            system_message.get("content")
            if isinstance(system_message, dict) and "content" in system_message
            else system_message
        )

        normalized_blocks = self._normalize_content_blocks(content)
        cached_blocks = [
            self._attach_cache_control(block) for block in normalized_blocks
        ]

        return cached_blocks if cached_blocks else None

    def _apply_cache_control_to_last_message(self, messages):
        """Apply cache control to the last user message."""
        if not messages:
            return

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return

        for block in reversed(last_message.get("content", [])):
            if isinstance(block, dict) and block.get("type") not in {
                "thinking",
                "redacted_thinking",
            }:
                self._attach_cache_control(block)
                break

    def _is_claude_4x_model(self, model_name: str) -> bool:
        """
        Determine if a model is a Claude 4.x generation model that has temperature/top_p constraints.
        Uses a more future-proof approach than simple prefix matching.

        Args:
            model_name: The model name to check

        Returns:
            True if this is a Claude 4.x model with constraints
        """
        import re

        # Pattern to match Claude 4.x models with various version suffixes
        # Examples: claude-opus-4, claude-opus-4-1-20250805, claude-sonnet-4-5, claude-sonnet-4-5-20250929
        # The pattern allows for optional sub-versions (like -1, -5) and dates
        pattern = r"^claude-(opus|sonnet)-4(?:-\d+)?(?:-\d{8})?$"

        return bool(re.match(pattern, model_name))

    def pipes(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_image(self, image_data):
        """Process image data with size validation."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            # Check base64 image size
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # For URL images, perform size check after fetching
            url = image_data["image_url"]["url"]
            response = requests.head(url, allow_redirects=True)
            content_length = int(response.headers.get("content-length", 0))

            if content_length > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        total_image_size = 0.0

        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        processed_image = self.process_image(item)
                        processed_content.append(processed_image)

                        # Track total size for base64 images
                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                            total_image_size += image_size
                            if (
                                total_image_size > 100 * 1024 * 1024
                            ):  # 100MB total limit
                                raise ValueError(
                                    "Total size of images exceeds 100 MB limit"
                                )
                    elif item["type"] == "thinking" and "signature" in item:
                        # Include thinking blocks if present in the message
                        processed_content.append(
                            {
                                "type": "thinking",
                                "thinking": item["thinking"],
                                "signature": item["signature"],
                            }
                        )
                    elif item["type"] == "redacted_thinking" and "data" in item:
                        # Include redacted thinking blocks if present
                        processed_content.append(
                            {"type": "redacted_thinking", "data": item["data"]}
                        )
            else:
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        system_blocks = self._prepare_system_blocks(system_message)
        self._apply_cache_control_to_last_message(processed_messages)

        model_name = body["model"][body["model"].find(".") + 1 :]

        # Check if this is a thinking model
        is_thinking_model = model_name.endswith("-think")

        # Remove the "-think" suffix for API call if present
        api_model_name = (
            model_name.replace("-think", "") if is_thinking_model else model_name
        )

        # Determine if thinking will be enabled
        will_enable_thinking = (
            self.valves.ENABLE_THINKING
            and is_thinking_model
            and "claude-sonnet-4-5" in model_name
        )

        payload = {
            "model": api_model_name,
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "stop_sequences": body.get("stop", []),
            "stream": body.get("stream", False),
        }

        if system_blocks:
            payload["system"] = system_blocks

        # Only add top_k if thinking is NOT enabled
        if not will_enable_thinking:
            payload["top_k"] = body.get("top_k", 40)

        # Add extended thinking for Claude 4.5 Sonnet with thinking
        if will_enable_thinking:
            # Ensure thinking budget is within reasonable limits (1024-32000 tokens)
            thinking_budget = max(1024, min(32000, self.valves.THINKING_BUDGET))
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        # Handle temperature/top_p settings based on model generation
        # Claude 4.x models only support either temperature OR top_p, not both
        is_claude_4x_model = self._is_claude_4x_model(api_model_name)

        if is_claude_4x_model:
            if is_thinking_model:
                # For thinking model, always use temperature = 1.0
                payload["temperature"] = 1.0
            elif self.valves.CLAUDE_USE_TEMPERATURE:
                payload["temperature"] = body.get("temperature", 0.8)
            else:
                payload["top_p"] = body.get("top_p", 0.9)
        else:
            # Other Claude models support both temperature and top_p
            payload["temperature"] = body.get("temperature", 0.8)
            payload["top_p"] = body.get("top_p", 0.9)

        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self.valves.BETA_FEATURES:
            headers["anthropic-beta"] = self.valves.BETA_FEATURES

        url = "https://api.anthropic.com/v1/messages"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        """Handle streaming response with the OpenWebUI thinking tags."""
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=(3.05, 60)
            ) as response:
                if response.status_code != 200:
                    error_text = response.text
                    try:
                        error_json = response.json()
                        if "error" in error_json:
                            error_text = error_json["error"].get("message", error_text)
                    except:
                        pass
                    raise Exception(f"HTTP Error {response.status_code}: {error_text}")

                thinking_content = ""
                is_thinking_block = False
                is_text_block = False
                has_yielded_thinking = False
                has_yielded_think_tag = False

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])

                                # Handle content block starts
                                if data["type"] == "content_block_start":
                                    block_type = data["content_block"].get("type", "")

                                    # Handle thinking block start
                                    if block_type == "thinking":
                                        is_thinking_block = True
                                        # Emit thinking start tag immediately
                                        if (
                                            not has_yielded_think_tag
                                            and self.valves.DISPLAY_THINKING
                                        ):
                                            yield "<think>"
                                            has_yielded_think_tag = True

                                    # Handle transition to text block
                                    elif block_type == "text":
                                        # If we were in a thinking block, close it before starting text
                                        if is_thinking_block and has_yielded_think_tag:
                                            yield "</think>"
                                            has_yielded_thinking = True

                                        is_thinking_block = False
                                        is_text_block = True

                                        # For text blocks, yield the initial text if any
                                        if (
                                            "text" in data["content_block"]
                                            and data["content_block"]["text"]
                                        ):
                                            yield data["content_block"]["text"]

                                    # Handle redacted thinking block
                                    elif (
                                        block_type == "redacted_thinking"
                                        and self.valves.DISPLAY_THINKING
                                    ):
                                        if not has_yielded_think_tag:
                                            yield "<think>"
                                            has_yielded_think_tag = True
                                        yield "[Redacted thinking content]"

                                # Handle block deltas
                                elif data["type"] == "content_block_delta":
                                    delta = data["delta"]

                                    # Stream thinking deltas with the thinking tag
                                    if (
                                        delta["type"] == "thinking_delta"
                                        and is_thinking_block
                                        and self.valves.DISPLAY_THINKING
                                    ):
                                        thinking_content += delta["thinking"]
                                        yield delta["thinking"]

                                    # Stream text deltas normally
                                    elif (
                                        delta["type"] == "text_delta" and is_text_block
                                    ):
                                        yield delta["text"]

                                # Handle block stops
                                elif data["type"] == "content_block_stop":
                                    if is_thinking_block:
                                        is_thinking_block = False
                                        # Close thinking tag at the end of thinking block
                                        if (
                                            has_yielded_think_tag
                                            and not has_yielded_thinking
                                        ):
                                            yield "</think>"
                                            has_yielded_thinking = True
                                    elif is_text_block:
                                        is_text_block = False

                                # Handle message stop
                                elif data["type"] == "message_stop":
                                    # Make sure thinking tag is closed if needed
                                    if (
                                        has_yielded_think_tag
                                        and not has_yielded_thinking
                                    ):
                                        yield "</think>"
                                    break

                                # Handle single message (non-streaming style response in stream)
                                elif data["type"] == "message":
                                    has_thinking = False

                                    # First check if there's thinking content
                                    for content in data.get("content", []):
                                        if (
                                            content["type"] == "thinking"
                                            or content["type"] == "redacted_thinking"
                                        ) and self.valves.DISPLAY_THINKING:
                                            has_thinking = True
                                            break

                                    # If there's thinking, handle it first
                                    if has_thinking:
                                        yield "<think>"

                                        for content in data.get("content", []):
                                            if (
                                                content["type"] == "thinking"
                                                and self.valves.DISPLAY_THINKING
                                            ):
                                                yield content["thinking"]
                                            elif (
                                                content["type"] == "redacted_thinking"
                                                and self.valves.DISPLAY_THINKING
                                            ):
                                                yield "[Redacted thinking content]"

                                        yield "</think>"

                                    # Then yield all text blocks
                                    for content in data.get("content", []):
                                        if content["type"] == "text":
                                            yield content["text"]

                                time.sleep(
                                    0.01
                                )  # Delay to avoid overwhelming the client

                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                print(f"Unexpected data structure: {e}")
                                print(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, headers, payload):
        """Handle non-streaming response from Anthropic API, including thinking blocks."""
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60)
            )
            if response.status_code != 200:
                error_text = response.text
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_text = error_json["error"].get("message", error_text)
                except:
                    pass
                raise Exception(f"HTTP Error {response.status_code}: {error_text}")

            res = response.json()

            if "content" not in res or not res["content"]:
                return ""

            has_thinking = False
            thinking_content = ""
            text_content = ""

            # First organize content by type
            for content_block in res["content"]:
                if content_block["type"] == "thinking" and self.valves.DISPLAY_THINKING:
                    has_thinking = True
                    thinking_content += content_block["thinking"]
                elif (
                    content_block["type"] == "redacted_thinking"
                    and self.valves.DISPLAY_THINKING
                ):
                    has_thinking = True
                    thinking_content += "[Redacted thinking content]"
                elif content_block["type"] == "text":
                    text_content += content_block["text"]

            # Then construct the response with the <think> tags
            result = ""
            if has_thinking:
                result += f"<think>{thinking_content}</think>"

            result += text_content
            return result
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
