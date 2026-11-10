# 🛡️ Uninterrupted Tool Runs

> **Local models finish tool-using tasks without being derailed by cutoffs, hallucinated tool calls, or context blowouts**

[![Version](https://img.shields.io/badge/version-2.8.21-blue.svg)](https://github.com/skalidd/unified-filter)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-Compatible-green.svg)](https://github.com/open-webui/open-webui)
[![License](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://unlicense.org)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

---

## 🌟 Overview

**Uninterrupted Tool Runs** is an OpenWebUI filter that keeps long, tool-heavy
conversations with local models alive and finishable. Small and mid-size local
models derail in four repeatable ways mid-task. This filter detects each one,
repairs it inside the same turn, and the model completes the job as if nothing
happened. Every fix is an independent valve, so you can switch any of them off.

## ✨ Key Features

- 🩹 **Cutoff healing** — a reply that dies mid-reasoning or mid-code-block is
  auto-continued by a sub-call to the same model and stitched onto the message;
  detection is anchored to the structured `output[]` items' own status, not guesses
- 🚫 **Hallucinated tool-call rejection** — plain-text tool-call blocks naming tools
  that are not active in the session are stripped, replaced with a short note, and
  the model is told to continue in prose (allowlist valve available); bare and
  spaced tag shapes are all detected
- 🔁 **Echo collapse** — the model replaying its own bridge narration is deduped,
  seam-anchored so only the *copy* is ever removed, never the original
- ✂️ **Context-blowout defense** — oversized `tool` results and `tool_call`
  arguments are budgeted before the prompt is sent; trimmed text spills to disk and
  the model sees an excerpt + path so it can re-read in slices. Nothing destroyed,
  just filed away. Runs on follow-up tool-loop requests too, not just turn start
- 🧹 **Spill-dir garbage collection** — rolling spill-file retention with live-message
  protection; the spill dir never grows forever
- 🧪 **Regression-tested** — 8 standalone test suites that caught real bugs during
  development all pass against this file

## 🚀 Installation

1. OpenWebUI → Admin Panel → Functions → **+** → *Pipeline - Filter*
2. Paste `main.py`, Save
3. Set the `owui_api_key` valve to your own API key (Admin Panel → Settings →
   API Keys) — required for the cutoff-healing sub-calls; everything else works without it
4. Point `tool_history_spill_dir` at a container-writable path (default
   `/app/backend/data/uf_spill`)
5. Enable the function; optionally add as a global filter

## ⚙️ Valves

All defaults are sensible; each feature has an on/off switch:
`heal_truncated_streams`, `reject_hallucinated_tools`, `collapse_echoed_bridge_lines`,
`trim_oversized_tool_history`, `trim_live_tool_history` — plus budgets, spill
retention, timeouts, and a `debug_halluc_audit` log valve. Full table in the
[project README](https://github.com/skalidd/unified-filter#valves-settings-at-a-glance).

## 📚 More

- Full documentation, valve table, and a "parts you can steal" section:
  [skalidd/unified-filter](https://github.com/skalidd/unified-filter)
- Every version has a written root-cause story: [CHANGELOG.md](https://github.com/skalidd/unified-filter/blob/main/CHANGELOG.md)
- License: public domain (Unlicense) — take any part of it for anything
