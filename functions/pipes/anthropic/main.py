"""
title: Anthropic Manifold Pipe
authors: justinh-rahb, christian-taillon, jfbloom22, aaronchan0
author_url: https://github.com/justinh-rahb
funding_url: https://github.com/open-webui
version: 0.4.0
required_open_webui_version: 0.3.17
license: MIT
"""

import os
import requests
import time
import json
from typing import List, Union, AsyncGenerator, Iterator, Optional, Dict
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")
        ANTHROPIC_ENABLE_WEB_SEARCH: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = ""  # anthropic/"
        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""), "ANTHROPIC_ENABLE_WEB_SEARCH": os.getenv("ANTHROPIC_ENABLE_WEB_SEARCH", False)}
        )
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
        self.client: AsyncAnthropic = AsyncAnthropic(api_key=self.valves.ANTHROPIC_API_KEY)

        # Model cache
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_time: float = 0
        self._cache_ttl = int(os.getenv("ANTHROPIC_MODEL_CACHE_TTL", "600"))

    def get_client(self) -> AsyncAnthropic:
        if self.client.api_key != self.valves.ANTHROPIC_API_KEY:
            self.client: AsyncAnthropic = AsyncAnthropic(api_key=self.valves.ANTHROPIC_API_KEY)
        return self.client

    async def get_anthropic_models_from_api(self, force_refresh: bool = False) -> List[Dict[str, str]]:
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
        if not force_refresh and self._model_cache is not None and (current_time - self._model_cache_time) < self._cache_ttl:
            return self._model_cache

        if not self.valves.ANTHROPIC_API_KEY:
            return [
                {
                    "id": "error",
                    "name": "ANTHROPIC_API_KEY is not set. Please update the API Key in the valves.",
                }
            ]

        try:
            anthropic_models = await self.get_client().models.list()
            models = [{"id": model.id, "name": model.display_name} for model in anthropic_models.data]

            # Update cache
            self._model_cache = models
            self._model_cache_time = current_time

            return models

        except Exception as e:
            print(f"Error fetching Anthropic models: {e}")
            return [
                {
                    "id": "error",
                    "name": f"Could not fetch models from Anthropic: {str(e)}",
                }
            ]

    async def get_anthropic_models(self) -> List[Dict[str, str]]:
        """
        Get Anthropic models from the API.
        """
        return await self.get_anthropic_models_from_api()

    async def pipes(self) -> List[dict]:
        return await self.get_anthropic_models()

    def process_image(self, image_data):
        """Process image data with size validation."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            # Check base64 image size
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB")

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
                raise ValueError(f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB")

            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }

    async def pipe(self, body: dict) -> Union[str, AsyncGenerator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        total_image_size = 0

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
                            if total_image_size > 100 * 1024 * 1024:  # 100MB total limit
                                raise ValueError("Total size of images exceeds 100 MB limit")
            else:
                processed_content = [{"type": "text", "text": message.get("content", "")}]

            processed_messages.append({"role": message["role"], "content": processed_content})

        payload = {
            "model": body["model"][body["model"].find(".") + 1 :],
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 0.8),
            "top_k": body.get("top_k", 40),
            # "top_p": body.get("top_p", 0.9),
            "stop_sequences": body.get("stop", []),
            "system": str(system_message) if system_message else "",
            "stream": body.get("stream", False),
        }
        payload["tools"] = (
            [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                    "user_location": {
                        "type": "approximate",
                        "city": "San Ramon",
                        "region": "California",
                        "country": "US",
                        "timezone": "America/Los_Angeles",
                    },
                }
            ]
            if self.valves.ANTHROPIC_ENABLE_WEB_SEARCH
            else []
        )
        try:
            if body.get("stream", False):
                return self.stream_response(payload)
            else:
                return await self.non_stream_response(payload)
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    async def stream_response(self, payload):
        try:
            async with self.get_client().messages.stream(
                model=payload["model"], max_tokens=payload["max_tokens"], system=payload["system"], messages=payload["messages"], tools=payload["tools"]
            ) as stream:
                input_json: str = ""
                is_thinking: bool = False
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "server_tool_use" or event.content_block.type == "tool_use":
                            if not is_thinking:
                                is_thinking = True
                                yield "<think>"
                            input_json = ""
                        elif event.content_block.type == "text":
                            if is_thinking:
                                is_thinking = False
                                yield "</think>"
                    elif event.type == "content_block_stop":
                        if event.content_block.type == "server_tool_use" or event.content_block.type == "tool_use":
                            input_params = ", ".join([f"{key}: {value}" for key, value in json.loads(input_json).items()])
                            yield f"calling {event.content_block.name} with {input_params}\n"
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield event.delta.text
                        elif event.delta.type == "input_json_delta":
                            input_json += event.delta.partial_json
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    async def non_stream_response(self, payload):
        try:
            resp = await self.get_client().messages.create(
                model=payload["model"], max_tokens=payload["max_tokens"], system=payload["system"], messages=payload["messages"], tools=payload["tools"]
            )
            return "\n".join([r.text if isinstance(r, TextBlock) else "" for r in resp.content])
        except Exception as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
