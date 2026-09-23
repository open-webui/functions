# Progress Ledger

Reusable Open WebUI Workspace Tool for durable, per-user workflow state across
independent chats and scheduled automations.

## Install

Paste [`main.py`](./main.py) into Open WebUI Workspace Tools. The deployed
tool uses the caller's `__user__` context to scope records and stores runtime
SQLite state at `/app/backend/data/progress_ledger.sqlite3`.

Supported operations:

- `get_state`
- `initialize_state`
- `record_completion`
- `update_state`

Do not commit the SQLite database or other runtime exports.

## Light-production operating contract

The production Open WebUI service mounts its persistent `openwebui_data`
volume at `/app/backend/data`. The infrastructure backup job detects volumes
whose names contain `openwebui`, uploads them to remote object storage, and
retains remote backups for 30 days. This covers the ledger database alongside
Open WebUI's other application data.

The backup job snapshots a live read-only volume. That is an intentional
light-production tradeoff for this low-write ledger, not a substitute for an
application-aware SQLite backup. Keep ledger writes short and infrequent.

Callers should fail closed on state errors:

- `recorded`: completion was persisted.
- `conflict`, `error`, or `not_initialized`: do not claim completion or advance
  the devotional series; surface the state problem and retry or request repair.
