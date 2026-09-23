"""
title: Progress Ledger
author: Workplace Labs
description: Lightweight per-user workflow state for assistants and scheduled automations.
version: 1.0.0
license: MIT
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        database_path: str = Field(
            default="/app/backend/data/progress_ledger.sqlite3",
            description="Server-local SQLite path for Progress Ledger state.",
        )
        max_recent_items: int = Field(
            default=25,
            ge=1,
            le=100,
            description="Maximum recent completed items retained for one workflow.",
        )
        max_state_chars: int = Field(
            default=12000,
            ge=1000,
            le=50000,
            description="Maximum serialized size of one workflow state.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _key(self, value: str, label: str) -> str:
        value = (value or "").strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value):
            raise ValueError(f"{label} must use lowercase letters, numbers, hyphens, or underscores.")
        return value

    def _user_id(self, user: Optional[dict]) -> str:
        user_id = (user or {}).get("id")
        if not user_id:
            raise ValueError("An authenticated Open WebUI user is required.")
        return str(user_id)

    def _db(self) -> sqlite3.Connection:
        path = self.valves.database_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS progress_ledger (
                user_id TEXT NOT NULL,
                project_key TEXT NOT NULL,
                workflow_key TEXT NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL,
                cursor_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                recent_items_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, project_key, workflow_key)
            )
        """)
        return conn

    def _record(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "workflow_status": row["status"],
            "version": row["version"],
            "cursor": json.loads(row["cursor_json"]),
            "state": json.loads(row["state_json"]),
            "recent_items": json.loads(row["recent_items_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _validate_size(self, cursor: dict, state: dict, history: list):
        if len(json.dumps({"cursor": cursor, "state": state, "recent_items": history}, ensure_ascii=False)) > self.valves.max_state_chars:
            raise ValueError("State is too large; keep summaries and history items concise.")

    def get_state(self, project_key: str, workflow_key: str, __user__: Optional[dict] = None) -> Dict[str, Any]:
        """Read current workflow state for the calling user. Never changes state."""
        try:
            user_id = self._user_id(__user__)
            project_key, workflow_key = self._key(project_key, "project_key"), self._key(workflow_key, "workflow_key")
            with self._db() as conn:
                row = conn.execute("SELECT * FROM progress_ledger WHERE user_id=? AND project_key=? AND workflow_key=?", (user_id, project_key, workflow_key)).fetchone()
            if not row:
                return {"status": "not_initialized", "project_key": project_key, "workflow_key": workflow_key, "message": "No state exists. Initialize it before recording progress."}
            result = self._record(row)
            result.update({"status": "ok", "project_key": project_key, "workflow_key": workflow_key})
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def initialize_state(self, project_key: str, workflow_key: str, cursor: Dict[str, Any], state: Dict[str, Any], __user__: Optional[dict] = None) -> Dict[str, Any]:
        """Create a workflow once. Existing state is preserved and never overwritten."""
        try:
            user_id = self._user_id(__user__)
            project_key, workflow_key = self._key(project_key, "project_key"), self._key(workflow_key, "workflow_key")
            if not isinstance(cursor, dict) or not isinstance(state, dict):
                raise ValueError("cursor and state must be JSON objects.")
            self._validate_size(cursor, state, [])
            now = self._now()
            with self._db() as conn:
                old = conn.execute("SELECT version FROM progress_ledger WHERE user_id=? AND project_key=? AND workflow_key=?", (user_id, project_key, workflow_key)).fetchone()
                if old:
                    return {"status": "already_initialized", "version": old["version"], "message": "Existing state was preserved."}
                conn.execute("INSERT INTO progress_ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (user_id, project_key, workflow_key, "active", 1, json.dumps(cursor), json.dumps(state), "[]", now, now))
            return {"status": "initialized", "version": 1, "cursor": cursor, "state": state}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def record_completion(self, project_key: str, workflow_key: str, expected_version: int, completed_item: Dict[str, Any], next_cursor: Dict[str, Any], state_updates: Dict[str, Any], __user__: Optional[dict] = None) -> Dict[str, Any]:
        """After successful work, append a compact record and advance state exactly once."""
        try:
            user_id = self._user_id(__user__)
            project_key, workflow_key = self._key(project_key, "project_key"), self._key(workflow_key, "workflow_key")
            if not all(isinstance(item, dict) for item in (completed_item, next_cursor, state_updates)):
                raise ValueError("completed_item, next_cursor, and state_updates must be JSON objects.")
            now = self._now()
            with self._db() as conn:
                row = conn.execute("SELECT * FROM progress_ledger WHERE user_id=? AND project_key=? AND workflow_key=?", (user_id, project_key, workflow_key)).fetchone()
                if not row:
                    return {"status": "not_initialized", "message": "Initialize state before recording completion."}
                if row["status"] != "active":
                    return {"status": row["status"], "message": "Workflow is not active; no completion recorded."}
                if row["version"] != expected_version:
                    return {"status": "conflict", "current_version": row["version"], "message": "State changed; re-read before recording."}
                state = json.loads(row["state_json"])
                state.update(state_updates)
                history = json.loads(row["recent_items_json"])
                item = dict(completed_item)
                item.setdefault("completed_at", now)
                history = (history + [item])[-self.valves.max_recent_items:]
                self._validate_size(next_cursor, state, history)
                new_version = expected_version + 1
                changed = conn.execute("UPDATE progress_ledger SET version=?, cursor_json=?, state_json=?, recent_items_json=?, updated_at=? WHERE user_id=? AND project_key=? AND workflow_key=? AND version=?", (new_version, json.dumps(next_cursor), json.dumps(state), json.dumps(history), now, user_id, project_key, workflow_key, expected_version)).rowcount
                if changed != 1:
                    return {"status": "conflict", "message": "Concurrent state change; no completion recorded."}
            return {"status": "recorded", "version": new_version, "cursor": next_cursor, "state": state}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def update_state(self, project_key: str, workflow_key: str, expected_version: int, state_updates: Dict[str, Any], cursor: Optional[Dict[str, Any]] = None, workflow_status: Optional[str] = None, __user__: Optional[dict] = None) -> Dict[str, Any]:
        """Make a deliberate correction, pause, resume, or redirect without recording completion."""
        try:
            user_id = self._user_id(__user__)
            project_key, workflow_key = self._key(project_key, "project_key"), self._key(workflow_key, "workflow_key")
            if not isinstance(state_updates, dict):
                raise ValueError("state_updates must be a JSON object.")
            if cursor is not None and not isinstance(cursor, dict):
                raise ValueError("cursor must be a JSON object.")
            if workflow_status is not None and workflow_status not in {"active", "paused", "completed"}:
                raise ValueError("workflow_status must be active, paused, or completed.")
            now = self._now()
            with self._db() as conn:
                row = conn.execute("SELECT * FROM progress_ledger WHERE user_id=? AND project_key=? AND workflow_key=?", (user_id, project_key, workflow_key)).fetchone()
                if not row:
                    return {"status": "not_initialized", "message": "Initialize state before updating."}
                if row["version"] != expected_version:
                    return {"status": "conflict", "current_version": row["version"], "message": "State changed; re-read before updating."}
                state = json.loads(row["state_json"])
                state.update(state_updates)
                new_cursor = cursor if cursor is not None else json.loads(row["cursor_json"])
                history = json.loads(row["recent_items_json"])
                self._validate_size(new_cursor, state, history)
                new_status = workflow_status or row["status"]
                new_version = expected_version + 1
                changed = conn.execute("UPDATE progress_ledger SET status=?, version=?, cursor_json=?, state_json=?, updated_at=? WHERE user_id=? AND project_key=? AND workflow_key=? AND version=?", (new_status, new_version, json.dumps(new_cursor), json.dumps(state), now, user_id, project_key, workflow_key, expected_version)).rowcount
                if changed != 1:
                    return {"status": "conflict", "message": "Concurrent state change; no update made."}
            return {"status": "updated", "version": new_version, "workflow_status": new_status, "cursor": new_cursor, "state": state}
        except Exception as e:
            return {"status": "error", "message": str(e)}
