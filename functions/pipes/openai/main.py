"""
title: OpenAI Manifold Pipe
authors: aaronchan0
version: 0.1.0
required_open_webui_version: 0.6.41
license: MIT
"""

import os
import requests
import time
import re
import json
from typing import List, Union, AsyncGenerator, Iterator, Optional, Dict
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
from openai import AsyncOpenAI


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="")
        OPENAI_ENABLE_WEB_SEARCH: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.id = "openai"
        self.name = ""  # openai/"
        self.valves = self.Valves(
            **{"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""), "OPENAI_ENABLE_WEB_SEARCH": os.getenv("OPENAI_ENABLE_WEB_SEARCH", False)}
        )
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=self.valves.OPENAI_API_KEY)

        # Model cache
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_time: float = 0
        self._cache_ttl = int(os.getenv("OPENAI_MODEL_CACHE_TTL", "600"))

    def get_client(self) -> AsyncOpenAI:
        if self.client.api_key != self.valves.OPENAI_API_KEY:
            self.client: AsyncOpenAI = AsyncOpenAI(api_key=self.valves.OPENAI_API_KEY)
        return self.client

    async def get_openai_models_from_api(self, force_refresh: bool = False) -> List[Dict[str, str]]:
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

        if not self.valves.OPENAI_API_KEY:
            return [{"id": "error", "name": "OPENAI_API_KEY is not set. Please update the API Key in the valves."}]

        try:
            openai_models = await self.get_client().models.list()
            models = [{"id": model.id, "name": model.id} for model in openai_models.data if re.search(r"gpt-(4\.1|5\.1)", model.id)]
            models.sort(key=lambda x: x["name"])

            # Update cache
            self._model_cache = models
            self._model_cache_time = current_time

            return models

        except Exception as e:
            print(f"Error fetching OpenAI models: {e}")
            return [{"id": "error", "name": f"Could not fetch models from OpenAI: {str(e)}"}]

    async def get_openai_models(self) -> List[Dict[str, str]]:
        """
        Get OpenAI models from the API.
        """
        return await self.get_openai_models_from_api()

    async def pipes(self) -> List[dict]:
        return await self.get_openai_models()

    def process_image(self, image_data):
        """Process image data with size validation."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            # Check base64 image size
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB")

            return {"type": "input_image", "image_url": f"{mime_type},{base64_data}"}
        else:
            # For URL images, perform size check after fetching
            url = image_data["image_url"]["url"]
            response = requests.head(url, allow_redirects=True)
            content_length = int(response.headers.get("content-length", 0))

            if content_length > self.MAX_IMAGE_SIZE:
                raise ValueError(f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB")

            return {
                "type": "input_image",
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
                        processed_content.append({"type": "input_text" if message["role"] == "user" else "output_text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        processed_image = self.process_image(item)
                        processed_content.append(processed_image)

                        # Track total size for base64 images
                        image_size = len(processed_image["image_url"]) * 3 / 4
                        total_image_size += image_size
                        if total_image_size > 100 * 1024 * 1024:  # 100MB total limit
                            raise ValueError("Total size of images exceeds 100 MB limit")
            else:
                processed_content = message.get("content", "")

            processed_messages.append({"role": message["role"], "content": processed_content})

        payload = {
            "model": body["model"][body["model"].find(".") + 1 :],
            "input": processed_messages,
            "max_output_tokens": body.get("max_tokens", 4096),
            # "temperature": body.get("temperature", 0.8),
            "instructions": str(system_message) if system_message else "",
            "tools": (
                [
                    {
                        "type": "web_search",
                        "user_location": {
                            "type": "approximate",
                            "country": "US",
                            "city": "San Ramon",
                            "region": "CA",
                        },
                    }
                ]
                if self.valves.OPENAI_ENABLE_WEB_SEARCH
                else []
            ),
        }
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
            async with self.get_client().responses.stream(**payload) as stream:
                input_json: str = ""
                is_thinking: bool = False
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                    elif event.type == "response.web_search_call.in_progress":
                        yield "<think>performing web search..."
                    elif event.type == "response.web_search_call.completed":
                        yield "</think>"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    async def non_stream_response(self, payload):
        try:
            resp = await self.get_client().responses.create(**payload)
            return resp.output_text
        except Exception as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
