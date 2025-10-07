"""
title: Anthropic Manifold Pipe
authors: justinh-rahb, christian-taillon, jfbloom22
author_url: https://github.com/justinh-rahb
funding_url: https://github.com/open-webui
version: 0.4.0
required_open_webui_version: 0.3.17
license: MIT
"""

import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator, Optional, Dict
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "")}
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
            else:
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        # Handle sampling parameters - Claude 4.5 doesn't allow both temperature and top_p
        temperature = body.get("temperature", 0.8)
        top_p = body.get("top_p", 0.9)

        # If both are provided, prioritize temperature and ignore top_p
        # This follows Anthropic's recommendation to use either temperature OR top_p, not both
        if temperature != 0.8 and top_p != 0.9:
            # Both parameters were explicitly provided, use temperature and ignore top_p
            print("Warning: Both temperature and top_p provided for Claude model. Using temperature only.")
        elif temperature == 0.8 and top_p == 0.9:
            # Neither parameter was explicitly provided, use temperature as default
            pass
        elif temperature != 0.8:
            # Only temperature was explicitly provided
            pass
        else:
            # Only top_p was explicitly provided, use it instead of temperature
            temperature = None

        payload = {
            "model": body["model"][body["model"].find(".") + 1 :],
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "top_k": body.get("top_k", 40),
            "stop_sequences": body.get("stop", []),
            **({"system": str(system_message)} if system_message else {}),
            "stream": body.get("stream", False),
        }

        # Only add temperature or top_p, never both
        if temperature is not None:
            payload["temperature"] = temperature
        else:
            payload["top_p"] = top_p

        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

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
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=(3.05, 60)
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data["type"] == "content_block_start":
                                    yield data["content_block"]["text"]
                                elif data["type"] == "content_block_delta":
                                    yield data["delta"]["text"]
                                elif data["type"] == "message_stop":
                                    break
                                elif data["type"] == "message":
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
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60)
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            return (
                res["content"][0]["text"] if "content" in res and res["content"] else ""
            )
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
