"""
title: Bilig WorkPaper Formula Readback
author: Greg Konush
author_url: https://github.com/gregkonush
git_url: https://github.com/proompteng/bilig
description: Write an input cell to a Bilig WorkPaper endpoint, recalculate formulas, and return verified JSON readback.
required_open_webui_version: 0.4.0
version: 0.1.0
license: MIT
"""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        base_url: str = Field(
            default="https://bilig.proompteng.ai",
            description="Base URL for the Bilig WorkPaper service.",
        )
        formula_path: str = Field(
            default="/api/workpaper/n8n/forecast",
            description="Path for the formula readback endpoint.",
        )
        timeout_seconds: int = Field(
            default=20,
            ge=1,
            le=120,
            description="HTTP timeout for formula readback requests.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def verify_formula_readback(
        self,
        sheet_name: str = "Inputs",
        address: str = "B3",
        value: Any = "0.5",
    ) -> str:
        """
        Write one input cell to a Bilig WorkPaper endpoint and return verified
        formula readback.

        :param sheet_name: Worksheet name, for example "Inputs".
        :param address: A1-style cell address, for example "B3".
        :param value: Cell value. JSON literals such as 0.5, true, or "SKU-1"
            are accepted. Native JSON values from tool calls are also accepted.
        """

        payload = {
            "sheetName": sheet_name,
            "address": address,
            "value": self._parse_value(value),
        }
        endpoint = urljoin(
            self.valves.base_url.rstrip("/") + "/",
            self.valves.formula_path.lstrip("/"),
        )

        try:
            result = self._post_json(endpoint, payload)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return json.dumps(
                {
                    "ok": False,
                    "error": "bilig_http_error",
                    "status": exc.code,
                    "body": body[:1200],
                },
                indent=2,
                sort_keys=True,
            )
        except URLError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": "bilig_network_error",
                    "reason": str(exc.reason),
                },
                indent=2,
                sort_keys=True,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": "bilig_unexpected_error",
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )

        return json.dumps(self._summarize_result(result), indent=2, sort_keys=True)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "accept": "application/json",
                "user-agent": "open-webui-bilig-workpaper-tool/0.1.0",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.valves.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _summarize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        after = result.get("after") if isinstance(result.get("after"), dict) else {}
        before = result.get("before") if isinstance(result.get("before"), dict) else {}

        return {
            "ok": bool(result.get("verified")),
            "verified": result.get("verified"),
            "editedCell": result.get("editedCell"),
            "persisted": result.get("persisted"),
            "restoredReadbackMatches": result.get("restoredReadbackMatches"),
            "before": {
                "input": before.get("input"),
                "expectedArr": before.get("expectedArr"),
                "expectedCustomers": before.get("expectedCustomers"),
            },
            "after": {
                "input": after.get("input"),
                "expectedArr": after.get("expectedArr"),
                "expectedCustomers": after.get("expectedCustomers"),
            },
        }

    def _parse_value(self, raw_value: Any) -> Any:
        if not isinstance(raw_value, str):
            return raw_value

        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value
