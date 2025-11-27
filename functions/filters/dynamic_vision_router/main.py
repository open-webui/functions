"""
title: Dynamic Vision Router
author: open-webui, atgehrhardt,
    credits to @iamg30 for v0.1.5-v0.1.7 updates
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1.8
required_open_webui_version: 0.3.8
"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Literal
import json

from open_webui.utils.misc import get_last_user_message_item


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        vision_model_id: str = Field(
            default="",
            description="The identifier of the vision model to be used for processing images. Note: Compatibility is provider-specific; ollama models can only route to ollama models, and OpenAI models to OpenAI models respectively.",
        )
        skip_reroute_models: list[str] = Field(
            default_factory=list,
            description="A list of model identifiers that should not be re-routed to the chosen vision model.",
        )
        enabled_for_admins: bool = Field(
            default=False,
            description="Whether dynamic vision routing is enabled for admin users.",
        )
        enabled_for_users: bool = Field(
            default=True,
            description="Whether dynamic vision routing is enabled for regular users.",
        )
        status: bool = Field(
            default=False,
            description="A flag to enable or disable the status indicator. Set to True to enable status updates.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if __model__["id"] in self.valves.skip_reroute_models:
            return body
        if __model__["id"] == self.valves.vision_model_id:
            return body
        if __user__ is not None:
            if __user__.get("role") == "admin" and not self.valves.enabled_for_admins:
                return body
            elif __user__.get("role") == "user" and not self.valves.enabled_for_users:
                return body

        messages = body.get("messages")
        if messages is None:
            # Handle the case where messages is None
            return body

        user_message = get_last_user_message_item(messages)
        if user_message is None:
            # Handle the case where user_message is None
            return body

        has_images = user_message.get("images") is not None
        if not has_images:
            user_message_content = user_message.get("content")
            if user_message_content is not None and isinstance(
                user_message_content, list
            ):
                has_images = any(
                    item.get("type") == "image_url" for item in user_message_content
                )

        # check for all history
        if not has_images:
            for m in messages:
                user_message_content = m.get("content")
                if user_message_content is not None and isinstance(
                    user_message_content, list
                ):
                    has_images = any(
                        item.get("type") == "image_url" for item in user_message_content
                    )

        if has_images:
            if self.valves.vision_model_id:
                body["model"] = self.valves.vision_model_id
                if self.valves.status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Request routed to {self.valves.vision_model_id}",
                                "done": True,
                            },
                        }
                    )
            else:
                if self.valves.status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "No vision model ID provided, routing could not be completed.",
                                "done": True,
                            },
                        }
                    )
        return body
