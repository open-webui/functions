import aiohttp
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        SCANNER_URL: str = Field(
            "http://localhost:8000/scan",
            description="URL of the external prompt scan service",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _get_last_user_prompt(self, body: dict) -> str:
        """Safely extract the most recent user prompt."""
        messages = body.get("messages", [])
        if not messages:
            return ""
        # Find the last message from the user (in reverse)
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return str(msg["content"])
        # Fallback: use the last message content if exists
        return str(messages[-1].get("content", ""))

    async def inlet(self, body):
        """Intercept the user prompt before sending to model."""
        prompt = self._get_last_user_prompt(body)
        if not prompt.strip():
            return body

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.SCANNER_URL,
                    json={"prompt": prompt},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
        except Exception as e:
            print(f"[Scan] Warning: {e}")
            return body  # fail open on network error

        if not data.get("is_valid", True):
            raise Exception(f"Input blocked (score={data.get('risk_score', 0):.2f})")

        sanitized = data.get("sanitized_prompt", prompt)
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                msg["content"] = sanitized
                break

        return body
