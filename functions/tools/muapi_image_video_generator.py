"""
title: MuAPI Image & Video Generator
author: muapi
author_url: https://muapi.ai
funding_url: https://muapi.ai
version: 1.0.0
license: MIT
requirements: requests
"""

import asyncio
import time
import requests
from typing import Optional
from pydantic import BaseModel, Field

BASE_URL = "https://api.muapi.ai/api/v1"

IMAGE_MODELS = [
    "flux-schnell", "flux-dev", "flux-kontext-pro", "hidream-fast",
    "midjourney", "gpt4o", "imagen4", "imagen4-fast", "seedream",
    "reve", "ideogram", "gpt-image-2",
]

VIDEO_MODELS = [
    "veo3-fast", "veo3", "kling-master", "wan2.1", "wan2.2",
    "seedance-pro", "runway", "pixverse", "sora",
]


class Tools:
    class Valves(BaseModel):
        MUAPI_API_KEY: str = Field(
            default="",
            description="Your MuAPI API key. Get one at https://muapi.ai/dashboard/api-keys",
            json_schema_extra={"ui": {"label": "MuAPI API Key", "type": "password"}},
        )
        DEFAULT_IMAGE_MODEL: str = Field(
            default="flux-schnell",
            description=f"Default image model. Options: {', '.join(IMAGE_MODELS)}",
        )
        DEFAULT_VIDEO_MODEL: str = Field(
            default="veo3-fast",
            description=f"Default video model. Options: {', '.join(VIDEO_MODELS)}",
        )
        POLL_INTERVAL: float = Field(default=3.0, description="Seconds between status polls")
        MAX_WAIT: int = Field(default=300, description="Max seconds to wait for generation")

    def __init__(self):
        self.valves = self.Valves()

    def _submit_and_poll(self, endpoint: str, payload: dict) -> str:
        """Submit a muapi.ai job and poll until completion."""
        headers = {"x-api-key": self.valves.MUAPI_API_KEY, "Content-Type": "application/json"}

        resp = requests.post(f"{BASE_URL}/{endpoint}", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        request_id = resp.json()["request_id"]

        deadline = time.time() + self.valves.MAX_WAIT
        while time.time() < deadline:
            time.sleep(self.valves.POLL_INTERVAL)
            poll = requests.get(
                f"{BASE_URL}/predictions/{request_id}/result",
                headers={"x-api-key": self.valves.MUAPI_API_KEY},
                timeout=15,
            )
            poll.raise_for_status()
            data = poll.json()
            if data["status"] == "completed":
                outputs = data.get("outputs", [])
                if not outputs:
                    raise RuntimeError("Generation completed but returned no outputs")
                return outputs[0]
            if data["status"] in ("failed", "cancelled"):
                raise RuntimeError(f"Generation {data['status']}: {data.get('error', '')}")

        raise TimeoutError(f"Generation timed out after {self.valves.MAX_WAIT}s")

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """
        Generate an image from a text description using MuAPI.
        Supports Flux, Midjourney, GPT-4o Image, Google Imagen 4, Seedream, and more.
        Returns a markdown image with the result URL.

        :param prompt: Text description of the image to generate
        :param model: Model ID (e.g. flux-schnell, midjourney, gpt4o, imagen4). Uses default if not set.
        :param width: Image width in pixels (optional)
        :param height: Image height in pixels (optional)
        """
        if not self.valves.MUAPI_API_KEY:
            return "Error: MUAPI_API_KEY is not set. Go to Admin > Tools and add your key."

        endpoint = model or self.valves.DEFAULT_IMAGE_MODEL
        payload: dict = {"prompt": prompt}
        if width:
            payload["width"] = width
        if height:
            payload["height"] = height

        try:
            url = self._submit_and_poll(endpoint, payload)
            return f"![Generated image]({url})\n\n**Model:** {endpoint} | **Prompt:** {prompt}"
        except Exception as e:
            return f"Image generation failed: {e}"

    def generate_video(
        self,
        prompt: str,
        model: Optional[str] = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
    ) -> str:
        """
        Generate a short video from a text description using MuAPI.
        Supports Veo3, Kling, Wan, Seedance, Runway, Pixverse, Sora, and more.
        Returns the video URL.

        :param prompt: Text description of the video to generate
        :param model: Model ID (e.g. veo3-fast, kling-master, wan2.1, runway). Uses default if not set.
        :param duration: Duration in seconds (3-60, default 5)
        :param aspect_ratio: Aspect ratio (16:9, 9:16, 1:1), default 16:9
        """
        if not self.valves.MUAPI_API_KEY:
            return "Error: MUAPI_API_KEY is not set. Go to Admin > Tools and add your key."

        endpoint = model or self.valves.DEFAULT_VIDEO_MODEL
        payload: dict = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        try:
            url = self._submit_and_poll(endpoint, payload)
            return f"**Generated video:** {url}\n\n**Model:** {endpoint} | **Prompt:** {prompt}"
        except Exception as e:
            return f"Video generation failed: {e}"
