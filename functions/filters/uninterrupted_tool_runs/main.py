"""
title: Uninterrupted Tool Runs
author: skalid
author_url: https://github.com/skalidd
project_url: https://github.com/skalidd/unified-filter
version: 2.8.21
license: Unlicense (public domain)
description: Lets a local model finish a tool-using task without derailment: auto-continues replies whose stream died mid-reasoning or mid-code-block, rejects plain-text tool calls naming tools that don't exist, stops double-printed narration, and caps oversized tool output before it blows the context window (full text spilled to disk, nothing destroyed). Every behaviour is a separate valve.
"""

import asyncio
import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
import time
import uuid
import httpx
from typing import Optional, Callable, Any, Awaitable, Tuple, List
from pydantic import BaseModel, Field

try:
    from open_webui.models.chats import Chats
except ImportError:
    Chats = None

ZWSP = "\u200b"
SELF_CALL_SENTINEL = "\u200b\u200c\u200b"

# Strong references to in-flight chat:reload tasks (see outlet()).
_PENDING_RELOAD_TASKS: set = set()


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=-100,
            description="Negative priority runs BEFORE other pipeline filters.",
        )
        owui_api_url: str = Field(
            default="http://localhost:8080/api/chat/completions",
            description="Open WebUI internal endpoint (falls back to :3000 if bare-metal).",
        )
        owui_api_key: str = Field(
            default="PASTE_YOUR_OPEN_WEBUI_API_KEY_HERE",
            description="Open WebUI API Key (sk-... from api_key table).",
        )
        command_regex: str = Field(
            default=r"```(?:bash|sh|shell|zsh)[ \t]*\n(.*?)\n```",
            description="Regex pattern for executable plaintext bash blocks.",
        )
        execute_plaintext_tool_tags: bool = Field(
            default=True,
            description="Also execute plaintext < tool_call>{...}< /tool_call> JSON blocks outside thoughts.",
        )
        max_loops: int = Field(
            default=3,
            description="Maximum auto-continue cycles per turn.",
        )
        command_timeout: int = Field(
            default=30,
            description="Timeout in seconds for command execution.",
        )
        show_command_output: bool = Field(
            default=True,
            description="Display executed command output inside the chat message.",
        )
        heal_truncated_streams: bool = Field(
            default=True,
            description="Auto-continue if the stream dies mid-reasoning, mid-thought, or mid-backtick.",
        )
        heal_context_messages: int = Field(
            default=12,
            description="Max prior messages sent to the continuation sub-call.",
        )
        reject_hallucinated_tools: bool = Field(
            default=True,
            description="Reject plain-text tool calls naming tools that are not active in this session, and let the model continue.",
        )
        hallucinated_tool_allowlist: str = Field(
            default="",
            description="Comma/space-separated extra tool names to treat as available (for inlet-provided or MCP tools not present in metadata['tools']).",
        )
        debug_halluc_audit: bool = Field(
            default=False,
            description="Diagnostic only: log the resolved active-tool set and detected dead-block count to /app/backend/data/uf_audit.log.",
        )
        post_heal_reload_seconds: float = Field(
            default=1.5,
            description="v2.8.19: after a heal actually modifies a turn, fire a chat:reload event this many seconds after outlet() returns, so the client refetches the chat and the healed body paints without a tab switch. The delay clears OWU's own persist of our body (middleware persists after outlet returns). 0 disables.",
        )
        collapse_echoed_bridge_lines: bool = Field(
            default=True,
            description="v2.8.8: when the model replays its own interim narration lines verbatim at the top of its final answer, drop the replayed copy. The originals survive as their own output items, so nothing is lost.",
        )
        trim_oversized_tool_history: bool = Field(
            default=True,
            description="v2.8.14: cap oversized tool outputs / tool-call arguments coming from HISTORY in the inlet, spilling the full text to disk so the turn does not blow the context window. Does not touch the stored message; only the outbound prompt.",
        )
        trim_live_tool_history: bool = Field(
            default=True,
            description="v2.8.21: also run the trim pass on the tool loop's mid-turn follow-up requests (filter_type='request'), which carry freshly produced tool results that history trimming never sees. Same caps, same spill-to-disk, same idempotent TRIM_MARK.",
        )
        tool_history_char_budget: int = Field(
            default=12000,
            description="Per-block character cap for replayed tool history (older blocks). 0 disables the per-block pass.",
        )
        tool_history_newest_char_budget: int = Field(
            default=40000,
            description="Character cap for the most recent replayed tool result, which the model most likely still needs.",
        )
        tool_history_total_char_budget: int = Field(
            default=60000,
            description="Second pass: if ALL replayed tool blocks together exceed this, shrink the largest ones until they fit. 0 disables the pass.",
        )
        tool_history_keep_head: int = Field(
            default=4000,
            description="Leading characters kept when a block is trimmed.",
        )
        tool_history_keep_tail: int = Field(
            default=2000,
            description="Trailing characters kept when a block is trimmed.",
        )
        tool_history_spill_dir: str = Field(
            default="/app/backend/data/uf_spill",
            description="Where full trimmed text is written. Must be readable by both the OWU container and the assistant's own filesystem tools.",
        )
        tool_history_spill_keep: int = Field(
            default=10,
            description="v2.8.17: rolling inventory - how many spilled files the directory keeps before the oldest are deleted. 0 disables cleanup entirely. A spill that is re-used gets its mtime refreshed, so an actively referenced block does not age out.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _is_self_call(self, messages: list) -> bool:
        for m in messages[-2:]:
            content = m.get("content", "")
            if isinstance(content, str) and SELF_CALL_SENTINEL in content:
                return True
        return False

    def _zwsp_defang(
        self, text: str, defang_fences: bool = True, defang_think_tags: bool = False
    ) -> str:
        tag_names = "tool_call|function_call|cmd|command|run_command|shell_call|execute_code"
        if defang_think_tags:
            tag_names += "|think|thought"

        text = re.sub(
            rf"<(/?\s*(?:{tag_names})\b)",
            lambda m: f"<{ZWSP}{m.group(1)}",
            text,
            flags=re.IGNORECASE,
        )
        if defang_fences:
            text = re.sub(
                r"```(bash|sh|shell|zsh|python|json|tool_code)\b",
                lambda m: f"```{ZWSP}{m.group(1)}",
                text,
                flags=re.IGNORECASE,
            )
        return text

    TRIM_MARK = "[unified_filter: tool history trimmed"

    def _spill_text(self, text: str, key: str) -> Optional[str]:
        """v2.8.14: write the full original of a trimmed block to disk so nothing is
        destroyed by a context trim. Content-addressed name means the same bytes are
        never written twice, and the path is stable across turns."""
        spill_dir = (self.valves.tool_history_spill_dir or "").strip()
        if not spill_dir:
            return None
        try:
            os.makedirs(spill_dir, exist_ok=True)
            digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]
            safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key or "tool"))[:48]
            path = os.path.join(spill_dir, f"{safe_key}_{digest}.txt")
            if os.path.exists(path):
                # v2.8.17: the name is content-addressed, so a re-spill of the same bytes
                # would otherwise leave mtime frozen at first write and let the GC retire
                # a block the current prompt is still pointing at. Touch it: recency now
                # means "still in use", not "first seen".
                try:
                    os.utime(path, None)
                except Exception:
                    pass
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            return path
        except Exception:
            return None

    def _gc_spill(self, spill_dir: str, protect: Optional[set] = None) -> int:
        """v2.8.17: keep the spill directory a rolling inventory instead of an append-only
        ledger. Without this the directory grows forever: every trimmed block writes a file
        and nothing ever removes it.

        Policy is by mtime, newest N survive (re-spills refresh mtime, so "newest" tracks
        what is actually in use). Anything that is not a .txt we wrote is left alone, and a
        cap of 0 disables cleanup. Returns the number of files deleted; never raises, since
        a failed cleanup must not take the trim pass down with it.

        `protect` is the set of paths the current prompt still quotes. The cap alone is not
        safe: a stub can name a spill the cap would evict, leaving the model pointed at a
        file that no longer exists. Caught by the GC suite (G6), not by eye. Protected files
        are never deleted however old they are, so the inventory is "newest N, plus anything
        still referenced" rather than a hard N.
        """
        try:
            keep = max(0, int(self.valves.tool_history_spill_keep))
        except Exception:
            keep = 10
        if keep == 0:
            return 0
        protect = {os.path.realpath(p) for p in (protect or set()) if p}
        try:
            entries = []
            for name in os.listdir(spill_dir):
                if not name.endswith(".txt"):
                    continue
                full = os.path.join(spill_dir, name)
                try:
                    entries.append((os.path.getmtime(full), full))
                except OSError:
                    continue
            deleted = 0
            for _, full in sorted(entries, key=lambda e: e[0], reverse=True)[keep:]:
                if os.path.realpath(full) in protect:
                    continue
                try:
                    os.unlink(full)
                    deleted += 1
                except OSError:
                    pass
            if deleted and self.valves.debug_halluc_audit:
                self._audit_line(f"spill gc: kept newest {keep}, deleted {deleted}")
            return deleted
        except Exception:
            return 0

    def _head_tail(self, text: str, head: int, tail: int) -> str:
        if head + tail >= len(text):
            return text
        dropped = len(text) - head - tail
        return (
            text[:head]
            + f"\n... [{dropped} characters omitted by unified_filter] ...\n"
            + (text[-tail:] if tail > 0 else "")
        )

    def _trim_tool_history(self, messages: list) -> List[dict]:
        """v2.8.14: cap replayed tool history in the outbound prompt.

        Why this is the filter-only half of the fix, and where its edge is:
        OWU flattens stored `output` items into OpenAI-format messages at
        middleware.py:2177 (process_messages_with_output), BEFORE the inlet runs at
        2319. So at inlet time history is already `role: 'tool'` messages plus
        assistant `tool_calls`. That means this pass can see and shrink replayed
        history from previous turns.

        CORRECTION (v2.8.21) - the claim in this docstring from 2.8.14 through 2.8.20
        was wrong. It asserted the loop 're-issues the follow-up through
        generate_chat_completion without re-entering this inlet', which is true of the
        INLET but not of the FILTER: middleware.py:6030 calls process_filter_functions
        with filter_type='request' right before that generate_chat_completion at 6037,
        and utils/filter.py:180 dispatches by getattr(function_module, filter_type),
        i.e. a method literally named `request`. There was none, so the dispatch
        silently returned form_data unchanged. The mid-turn payload is likewise flat:
        middleware.py:5905-5950 rebuilds new_form_data['messages'] via
        convert_output_to_messages(..., flatten_tool_images=True), so fresh tool
        results arrive as role:'tool' messages this same pass can shrink. That is why
        this method is shared by inlet and request instead of being duplicated.

        Rules that must not be broken:
          * Never drop a tool message outright - `sanitize_tool_pairs` and provider
            templates require every tool_call to have a matching tool_call_id reply.
            Shrink content in place.
          * tool_calls[].arguments must stay parseable JSON.
          * Image parts in a list-shaped tool content are left untouched.
          * Blocks already carrying TRIM_MARK are skipped, so re-sending a trimmed
            history costs nothing and never double-spills.
        """
        v = self.valves
        trimmed: List[dict] = []
        if not getattr(v, "trim_oversized_tool_history", True):
            return trimmed

        budget = max(0, int(v.tool_history_char_budget))
        newest_budget = max(budget, int(v.tool_history_newest_char_budget))
        total_budget = max(0, int(v.tool_history_total_char_budget))
        head = max(0, int(v.tool_history_keep_head))
        tail = max(0, int(v.tool_history_keep_tail))
        if budget <= 0 and total_budget <= 0:
            return trimmed

        last_tool_idx = -1
        for idx, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                last_tool_idx = idx

        def text_parts(content: Any) -> List[Tuple[int, str]]:
            """(part_index, text) for the text-bearing parts of a tool content."""
            if isinstance(content, str):
                return [(0, content)]
            if isinstance(content, list):
                out = []
                for i, part in enumerate(content):
                    if isinstance(part, dict):
                        t = part.get("text") or part.get("content")
                        if isinstance(t, str):
                            out.append((i, t))
                return out
            return []

        def record(kind: str, where: Any, original: str, kept: str, spill: Optional[str]):
            trimmed.append(
                {
                    "kind": kind,
                    "where": where,
                    "before": len(original),
                    "after": len(kept),
                    "spill": spill,
                }
            )

        # pass 1: per-block caps
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")

            if role == "tool":
                cap = newest_budget if idx == last_tool_idx else budget
                if cap <= 0:
                    continue
                content = msg.get("content")
                call_id = str(msg.get("tool_call_id") or f"tool_{idx}")
                if isinstance(content, str):
                    if self.TRIM_MARK in content or len(content) <= cap:
                        continue
                    spill = self._spill_text(content, call_id)
                    kept = self._stub(content, cap, head, tail, spill, call_id)
                    msg["content"] = kept
                    record("tool_output", idx, content, kept, spill)
                else:
                    for part_idx, text in text_parts(content):
                        if self.TRIM_MARK in text or len(text) <= cap:
                            continue
                        spill = self._spill_text(text, call_id)
                        kept = self._stub(text, cap, head, tail, spill, call_id)
                        content[part_idx]["text"] = kept
                        record("tool_output", (idx, part_idx), text, kept, spill)

            elif role == "assistant":
                calls = msg.get("tool_calls")
                if not isinstance(calls, list):
                    continue
                for c_idx, call in enumerate(calls):
                    if not isinstance(call, dict):
                        continue
                    args = call.get("function", {}).get("arguments") if isinstance(
                        call.get("function"), dict
                    ) else call.get("arguments")
                    if not isinstance(args, str) or self.TRIM_MARK in args:
                        continue
                    if len(args) <= max(budget, 4000):
                        continue
                    name = (
                        call.get("function", {}).get("name")
                        if isinstance(call.get("function"), dict)
                        else None
                    ) or call.get("name") or f"call_{c_idx}"
                    spill = self._spill_text(args, name)
                    kept = self._args_stub(args, head, tail, spill)
                    if isinstance(call.get("function"), dict):
                        call["function"]["arguments"] = kept
                    else:
                        call["arguments"] = kept
                    record("tool_args", (idx, c_idx), args, kept, spill)

        # pass 2: total cap, biggest first, only on tool outputs still in play
        if total_budget > 0:
            for _ in range(12):
                live = []
                for idx, msg in enumerate(messages):
                    if isinstance(msg, dict) and msg.get("role") == "tool":
                        for part_idx, text in text_parts(msg.get("content")):
                            if self.TRIM_MARK not in text:
                                live.append((len(text), idx, part_idx, text))
                total = sum(n for n, *_ in live)
                if not live or total <= total_budget:
                    break
                live.sort(reverse=True)
                n, idx, part_idx, text = live[0]
                target = max(head + tail, 1500)
                if n <= target:
                    break
                spill = self._spill_text(text, f"tool_{idx}")
                kept = self._stub(text, target, head, tail, spill, f"tool_{idx}")
                content = messages[idx].get("content")
                if isinstance(content, str):
                    messages[idx]["content"] = kept
                else:
                    content[part_idx]["text"] = kept
                record("tool_output_total", (idx, part_idx), text, kept, spill)

        # v2.8.17: cleanup runs once per trim pass, not per spill write, and it is told
        # which files this prompt still quotes. A stub embeds its spill path in message
        # content, so anything still named by a live message is protected from the cap.
        spill_dir = (v.tool_history_spill_dir or "").strip()
        if trimmed and spill_dir:
            protect = {t["spill"] for t in trimmed if t.get("spill")}
            for msg in messages:
                if not isinstance(msg, dict) or msg.get("role") != "tool":
                    continue
                content = msg.get("content")
                parts = [content] if isinstance(content, str) else [
                    p.get("text") or p.get("content")
                    for p in content
                    if isinstance(p, dict)
                ] if isinstance(content, list) else []
                for ptext in parts:
                    if isinstance(ptext, str) and self.TRIM_MARK in ptext:
                        for tok in ptext.split():
                            tok = tok.strip("-().,")
                            if tok.startswith(spill_dir.rstrip("/") + "/"):
                                protect.add(tok)
            self._gc_spill(spill_dir, protect=protect)

        return trimmed

    def _stub(
        self,
        original: str,
        cap: int,
        head: int,
        tail: int,
        spill: Optional[str],
        call_id: str,
    ) -> str:
        # v2.8.16: the shape here used to be `keep = min(cap, head + tail)`, which made
        # every cap above head+tail (4000+2000 by default) a dead knob: a block with a
        # 12000 budget and one with a 40000 budget both floored to the same ~6000 chars,
        # so the newest-gets-more-headroom rule never had any effect and only keep_head /
        # keep_tail did anything. Caught by the trim suite, not by eye. The cap is now the
        # budget: tail keeps its configured amount, the head absorbs the remaining room
        # (the start of a tool dump is the part worth reading first), and the mark plus
        # spill pointer is reserved so the result stays inside the cap.
        reserve = 400 if spill else 200
        keep = max(200, cap - reserve)
        t = min(max(0, tail), keep // 2)
        h = keep - t
        body = self._head_tail(original, h, t)
        where = (
            f" Full text saved to {spill} - re-read it in slices with read_file."
            if spill
            else " (spill failed; only the excerpt survives.)"
        )
        return (
            f"{self.TRIM_MARK}: {len(original)} chars -> {len(body)} kept for context "
            f"budget]{where}\n{body}"
        )

    def _args_stub(
        self, original: str, head: int, tail: int, spill: Optional[str]
    ) -> str:
        """Keep tool_calls.arguments parseable: the trimmed form is a real JSON object."""
        preview = self._head_tail(original, min(head, 1200), min(tail, 600))
        payload = {
            "unified_filter_trimmed": True,
            "original_chars": len(original),
            "full_text_file": spill,
            "preview": preview,
        }
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return json.dumps({"unified_filter_trimmed": True, "original_chars": len(original)})

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> dict:
        messages = body.get("messages", [])
        if not messages:
            return body

        is_subcall = self._is_self_call(messages)

        if messages[-1].get("role") == "assistant" and isinstance(
            messages[-1].get("content"), str
        ):
            last_content = messages[-1]["content"]
            if re.search(r"</details>\s*$", last_content, flags=re.IGNORECASE):
                messages[-1]["content"] = self._unwrap_reasoning_for_api(last_content)

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                content = re.sub(
                    r"<\u200b?(/?)\s*(tool_call|function_call|run_command|cmd|command|shell_call|execute_code)\b",
                    r"< \1\2",
                    content,
                    flags=re.IGNORECASE,
                )
                content = re.sub(
                    r"`<\u200b?(/?)(think|thought)>`",
                    r"`< \1\2>`",
                    content,
                    flags=re.IGNORECASE,
                )
                msg["content"] = content

        if is_subcall:
            return body

        # v2.8.12: the tag-hygiene note that used to be appended here to the last user
        # message is GONE. Reasons, both measured:
        #   * It was redundant. The same instruction already lives in OWU's per-model
        #     system prompt (chat model row `system`, 664 chars), which is the editable,
        #     discoverable place for it. Two injections of one rule is prompt bloat
        #     asked to stop accumulating.
        #   * It was actively harmful to non-chat models. one small local model (a live chat /
        #     an assistant row) read the note as CALL syntax, not as a caution about
        #     prose mentions: its stored reasoning says 'Formatting: Specific syntax for
        #     tool calls (space after "<"))' and it then invented a `google_search` tool
        #     that has never been registered here. A note written for one model's chat
        #     profile was being pushed into every model's turns.
        # The defanging regex above still normalizes inbound history; only the extra
        # prompt text is removed. `is_subcall` still guards the early return.

        # v2.8.14: context-blowout defense for REPLAYED history. Runs after the
        # sub-call guard on purpose: heal sub-calls already get clipped tool data via
        # _extract_tool_outputs_from_output, so trimming them again would only lose
        # evidence the continuation needs.
        await self._trim_pass(messages, "history", __event_emitter__)
        return body

    async def request(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> dict:
        """v2.8.21: the hook OWU's TOOL LOOP looks for.

        utils/filter.py:180 does getattr(function_module, filter_type, None) and bails
        silently if it is not callable. filter_type is 'request' at middleware.py:6030
        (inside the tool loop, right before the follow-up generate_chat_completion at
        6037) and at 3005/3174. Without a method named `request` every mid-turn
        follow-up skipped the filter, which is how a single 742,993-char live tool
        result reached the engine untrimmed and produced
        'prompt (245393 tokens) leaves no room to answer in the context (131072)'.

        Unlike inlet, an exception here is NOT swallowed: utils/filter.py re-raises for
        every filter_type except 'inlet', so a bug in this method would take down the
        whole turn. Everything defensive therefore lives inside _trim_pass.
        """
        messages = body.get("messages") if isinstance(body, dict) else None
        if not messages or not isinstance(messages, list):
            return body
        if not getattr(self.valves, "trim_live_tool_history", True):
            return body
        await self._trim_pass(messages, "mid-turn", __event_emitter__)
        return body

    async def _trim_pass(
        self,
        messages: list,
        source: str,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> None:
        try:
            trimmed = self._trim_tool_history(messages)
        except Exception:
            trimmed = []
        if trimmed:
            off = sum(t["before"] - t["after"] for t in trimmed)
            if self.valves.debug_halluc_audit:
                self._audit_line(
                    f"{source} trim: "
                    + "; ".join(
                        f"{t['kind']}@{t['where']} {t['before']}->{t['after']} "
                        f"spill={t['spill']}"
                        for t in trimmed[:8]
                    )
                )
            if __event_emitter__ is not None:
                try:
                    await __event_emitter__(
                        {
                            "type": "notice",
                            "data": {
                                "level": "info",
                                "title": "unified_filter: context trim",
                                "description": (
                                    f"{len(trimmed)} {source} tool block(s) capped, "
                                    f"{off} chars off the prompt. Full text spilled to "
                                    f"{self.valves.tool_history_spill_dir}"
                                ),
                                "content": (
                                    f"{len(trimmed)} {source} tool block(s) capped, "
                                    f"{off} chars off the prompt. Full text spilled to "
                                    f"{self.valves.tool_history_spill_dir}"
                                ),
                            },
                        }
                    )
                except Exception:
                    pass

    async def _get_output_items(
        self, last_msg: dict, __metadata__: Optional[dict]
    ) -> List[dict]:
        output = last_msg.get("output")
        if isinstance(output, list) and len(output) > 0:
            return output

        if Chats and isinstance(__metadata__, dict):
            chat_id = __metadata__.get("chat_id")
            message_id = __metadata__.get("message_id") or last_msg.get("id")
            if chat_id and message_id and not str(chat_id).startswith("local:"):
                try:
                    res = Chats.get_message_by_id_and_message_id(chat_id, message_id)
                    if inspect.isawaitable(res):
                        res = await res
                    if res:
                        db_output = (
                            res.get("output")
                            if isinstance(res, dict)
                            else getattr(res, "output", None)
                        )
                        if isinstance(db_output, list):
                            return db_output
                except Exception:
                    pass
        return []

    def _extract_text_from_parts(self, parts: Any) -> str:
        if isinstance(parts, str):
            return parts
        if isinstance(parts, list):
            chunks = []
            for p in parts:
                if isinstance(p, str):
                    chunks.append(p)
                elif isinstance(p, dict):
                    t = p.get("text") or p.get("content") or ""
                    if isinstance(t, str):
                        chunks.append(t)
            return "".join(chunks)
        return ""

    def _extract_reasoning_from_output(self, output_items: List[dict]) -> str:
        reasoning_blocks = []
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            text = (
                self._extract_text_from_parts(item.get("content"))
                or self._extract_text_from_parts(item.get("summary"))
                or (item.get("text") if isinstance(item.get("text"), str) else "")
            )
            if text.strip():
                reasoning_blocks.append(text.strip())
        return "\n\n".join(reasoning_blocks)

    def _extract_tool_outputs_from_output(self, output_items: List[dict]) -> str:
        """
        v2.8.1: Preserves in-turn answered tool results (function_call + function_call_output)
        so a heal sub-call sees the data its tools already fetched before the reasoning death.
        """
        calls_by_id = {}
        for item in output_items:
            if isinstance(item, dict) and item.get("type") in self.TOOLISH_TYPES:
                cid = item.get("call_id") or item.get("id")
                name = item.get("name") or "tool"
                args = item.get("arguments") or ""
                if cid:
                    calls_by_id[cid] = f"{name}({str(args)[:200]})"

        results = []
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                cid = item.get("call_id")
                raw_out = self._extract_text_from_parts(item.get("output")) or str(
                    item.get("output") or ""
                )
                if raw_out.strip():
                    header = calls_by_id.get(cid, f"call_{cid}")
                    results.append(
                        f"[{header} -> {self._tail_clip(raw_out.strip(), 1500)}]"
                    )
        return "\n".join(results[-6:])

    def _extract_visible_from_output(self, output_items: List[dict]) -> str:
        msg_blocks = []
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            text = self._extract_text_from_parts(item.get("content"))
            if text.strip():
                msg_blocks.append(text.strip())
        return "\n".join(msg_blocks)

    def _collapse_echoed_bridge(self, final_text: str, interim_texts: List[str]) -> str:
        """v2.8.8: drop interim 'bridge' lines the model restates verbatim inside
        its own final answer.

        Why this is safe when most of this filter refuses to delete: every span
        removed here still exists verbatim as its own output `message` item, so
        the span being cut is a copy, never the only copy. That is the opposite
        of the heal-path risk, where a removed span is the only copy.

        The visible symptom: OWU stores `content` as '\\n'.join() of every
        message item (utils/misc.py get_output_text), so an answer that opens by
        replaying the interim narration lines gets them twice - once from the
        interim items, once inside the final item.

        Guard: only full interim items are candidates, and only when reasonably
        long, so a legitimately repeated short line is left alone.
        """
        if not final_text or not interim_texts:
            return final_text

        collapsed = final_text
        # Fenced / backticked / thought regions are off limits: a line that
        # happens to match the interim narration but sits inside a code block is
        # the model showing code, not repeating itself.
        code_spans = self._code_region_spans(collapsed)

        def _in_code(pos: int) -> bool:
            return any(start <= pos < end for start, end in code_spans)

        for text in interim_texts:
            probe = text.strip()
            if len(probe) < 40:
                continue
            # Line-anchored only. An unanchored cut could splice two unrelated
            # sentences together if the model quotes the line inside prose, and
            # spliced prose is corruption while a repeated line is only ugly.
            pattern = re.compile(
                r"(?:^|(?<=\n))" + re.escape(probe) + r"(?=\n|$)"
            )
            hit = next(
                (m for m in pattern.finditer(collapsed) if not _in_code(m.start())),
                None,
            )
            if hit is None:
                continue
            collapsed = collapsed[: hit.start()] + collapsed[hit.end() :]
            code_spans = self._code_region_spans(collapsed)

        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed).strip()
        if not collapsed:
            return final_text
        return collapsed

    TOOLISH_TYPES = (
        "function_call",
        "tool_call",
        "tool_calls",
        "custom_tool_call",
        "computer_call",
        "local_shell_call",
    )

    def _has_unanswered_tool_call(
        self, last_msg: dict, output_items: List[dict]
    ) -> bool:
        if last_msg.get("tool_calls") or last_msg.get("function_call"):
            return True

        answered = {
            item.get("call_id")
            for item in output_items
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id")
        }
        for item in reversed(output_items):
            if not isinstance(item, dict):
                continue
            if item.get("type") in self.TOOLISH_TYPES:
                cid = item.get("call_id") or item.get("id")
                return cid not in answered
            if item.get("type") in ("message", "reasoning"):
                if self._extract_text_from_parts(item.get("content")).strip():
                    return False
                continue
        return False

    def _last_index_of(self, output_items: List[dict], types: tuple) -> int:
        for i in range(len(output_items) - 1, -1, -1):
            item = output_items[i]
            if isinstance(item, dict) and item.get("type") in types:
                return i
        return -1

    def _inspect_output_for_cutoff(
        self, last_msg: dict, output_items: List[dict], latest_chunk: str
    ) -> Tuple[bool, str]:
        if not output_items:
            return False, ""

        if self._has_unanswered_tool_call(last_msg, output_items):
            return False, ""

        reasoning_idx = self._last_index_of(output_items, ("reasoning",))
        if reasoning_idx < 0:
            return False, ""

        message_idx = -1
        for i, item in enumerate(output_items):
            if (
                isinstance(item, dict)
                and item.get("type") == "message"
                and self._extract_text_from_parts(item.get("content")).strip()
            ):
                message_idx = i
        if message_idx > reasoning_idx:
            return False, ""

        reasoning_text = self._extract_reasoning_from_output(output_items)
        if not reasoning_text.strip():
            return False, ""

        visible_from_output = self._extract_visible_from_output(output_items)
        visible_from_content = self._strip_thoughts_only(latest_chunk)
        has_any_visible = bool(visible_from_output or visible_from_content)

        status = output_items[reasoning_idx].get("status")
        reasoning_is_last_item = (
            isinstance(output_items[-1], dict)
            and output_items[-1].get("type") == "reasoning"
        )

        if not has_any_visible:
            if self._is_truncated_mid_backtick(reasoning_text):
                return True, self._clean_dangling_backtick(reasoning_text)
            return True, reasoning_text

        if reasoning_is_last_item and status in ("incomplete", "in_progress", "failed"):
            return True, reasoning_text

        return False, reasoning_text

    def _sync_structured_output(
        self,
        existing_output: List[dict],
        full_visible_content: str,
        healed_delta: str = "",
        final_item_text: Optional[str] = None,
    ) -> List[dict]:
        """
        v2.8.9 root-cause fix for the duplicated-narration turns.
        OWU stores `content` as the join of every message item, and it already
        keeps each interim bridge line as its own item. When the outlet rewrote
        the FINAL item with `full_visible_content` (the whole cumulative
        display), that item became a copy of everything that came before it and
        the join rendered the bridge lines a second time. So the final item now
        receives `final_item_text`: that item's own text plus whatever the
        filter's heal/execute loops produced. `full_visible_content` remains
        only as the fallback for turns that have no message item to hold the
        visible text (reasoning-only turns), which is the legacy case.

        v2.8.1 fix for Gate-3 timeline ordering:
        1. Marks any trailing incomplete/in_progress reasoning blocks as 'completed'.
        2. Only updates an existing message item in-place if it already sits AFTER the last
           reasoning item (or is an empty message stub at the end).
        3. On Gate-3 (where an earlier message item like "Let me probe for it." sits BEFORE
           tool calls and final reasoning), keeps that earlier message item intact and appends
           a new message item at the end containing `healed_delta`.
        """
        new_output = copy.deepcopy(existing_output) if existing_output else []

        for item in new_output:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                if item.get("status") in ("incomplete", "in_progress", "failed"):
                    item["status"] = "completed"

        last_reasoning_idx = self._last_index_of(new_output, ("reasoning",))

        for i in range(len(new_output) - 1, -1, -1):
            item = new_output[i]
            if isinstance(item, dict) and item.get("type") == "message":
                existing_txt = self._extract_text_from_parts(
                    item.get("content")
                ).strip()
                if i > last_reasoning_idx or not existing_txt:
                    item["status"] = "completed"
                    if final_item_text is not None:
                        text_for_item = final_item_text
                    elif (
                        last_reasoning_idx >= 0
                        and healed_delta.strip()
                        and i > last_reasoning_idx
                    ):
                        text_for_item = healed_delta.strip()
                    else:
                        text_for_item = full_visible_content
                    item["content"] = [
                        {"type": "output_text", "text": text_for_item}
                    ]
                    return new_output
                break

        text_to_append = (
            healed_delta.strip()
            if (last_reasoning_idx >= 0 and healed_delta.strip())
            else full_visible_content
        )
        if not text_to_append.strip():
            return new_output
        if not healed_delta.strip() and any(
            isinstance(item, dict)
            and item.get("type") == "message"
            and self._extract_text_from_parts(item.get("content")).strip()
            for item in new_output
        ):
            return new_output
        new_output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text_to_append}],
            }
        )
        return new_output

    def _strip_Echoed_pseudo_tags(self, text: str) -> str:
        """Strips leaked [Thought] / [Tool Call] narration prefixes if the sub-call echoes them."""
        cleaned = re.sub(
            r"^\s*\[(?:Thought|Thought So Far|Reasoning)\]\s*:?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\n*\[Tool Call\]\s*[^\n]+$",
            "",
            cleaned.strip(),
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    @staticmethod
    def _dedupe_continuation(displayed: str, new_text: str) -> str:
        """
        v2.8.4 rewrite. v2.8.3 searched the WHOLE turn for the continuation's
        leading lines and blanket-stripped both ends. an external reviewer model's adversarial review
        found that deletes genuinely new text, and all 6 of its cases reproduced
        as failures against the real 2.8.3 function: repeated table/code headers
        in a new section, closing fences, indented code bodies, coincidental
        >=20-char prefixes. Duplicated text is ugly; deleted text is corrupting.

        This version anchors to the SEAM only. A continuation echo is by
        definition a re-emission of the TAIL of what was just streamed, so a
        block counts as an echo only when it matches the END of `displayed`.
        Leading whitespace is preserved so indented code survives; only trailing
        whitespace is trimmed. When the evidence is ambiguous it duplicates
        rather than deletes.
        """
        new_text = new_text or ""
        displayed = displayed or ""
        if not new_text.strip() or not displayed.strip():
            return new_text

        d_tail = [ln.strip() for ln in displayed.splitlines()]
        n_lines = new_text.split("\n")
        n_head = [ln.strip() for ln in n_lines]

        # 1) Line-anchored echo: the continuation's first i lines equal
        #    displayed's LAST i lines. Longest such i wins. 40-char floor so
        #    short boilerplate lines survive.
        best = 0
        for i in range(min(len(d_tail), len(n_head)), 1, -1):
            if len("\n".join(n_head[:i])) < 40:
                continue
            if n_head[:i] == d_tail[-i:]:
                best = i
                break
        if best:
            remainder = "\n".join(n_lines[best:])
            if not remainder.strip():
                return ""
            new_text = remainder

        # 2) Character-anchored echo (mid-sentence cutoff): displayed ends with
        #    the continuation's first k chars. Trusted only when the seam lands
        #    on a word boundary (next char is whitespace) or the continuation is
        #    fully consumed -- that guard is what stops a coincidental shared
        #    prefix like "assert response.json()" from being eaten. Floor is 20
        #    chars so short legitimate repeats survive.
        probe = new_text.rstrip()
        max_k = min(len(displayed), len(probe))
        for k in range(max_k, 19, -1):
            if displayed.endswith(probe[:k]):
                nxt = probe[k] if k < len(probe) else ""
                if not nxt or nxt.isspace():
                    new_text = new_text[k:]
                break

        return new_text.rstrip()

    def _strip_thoughts_only(self, text: str) -> str:
        cleaned = re.sub(
            r"<details\b[^>]*type=[\"']reasoning[\"'][^>]*>[\s\S]*?</details>",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(?<!`)<(?:think|thought)>(?!`)[\s\S]*?(?<!`)</(?:think|thought)>(?!`)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if re.search(r"(?<!`)</(?:think|thought)>(?!`)", cleaned, flags=re.IGNORECASE):
            cleaned = re.split(
                r"(?<!`)</(?:think|thought)>(?!`)", cleaned, flags=re.IGNORECASE
            )[-1]
        cleaned = re.split(
            r"(?<!`)<(?:think|thought)>(?!`)", cleaned, maxsplit=1, flags=re.IGNORECASE
        )[0]
        return cleaned.strip()

    def _strip_thoughts_and_backticks(self, text: str) -> str:
        cleaned = self._strip_thoughts_only(text)
        cleaned = re.sub(r"(?<!`)`(?!`)[^`\n]+?(?<!`)`(?!`)", "", cleaned)
        return cleaned.strip()

    def _extract_plaintext_command(self, actionable_text: str) -> Optional[str]:
        match = re.search(self.valves.command_regex, actionable_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        if self.valves.execute_plaintext_tool_tags:
            tc_match = re.search(
                r"<\u200b?\s*tool_call>\s*(\{[\s\S]*?\})\s*<\u200b?/\s*tool_call>",
                actionable_text,
                re.IGNORECASE,
            )
            if tc_match:
                try:
                    data = json.loads(tc_match.group(1))
                    args = data.get("arguments") or data.get("parameters") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    if isinstance(args, dict):
                        for key in ("cmd", "command", "script", "code"):
                            if key in args and isinstance(args[key], str):
                                return args[key].strip()
                except Exception:
                    pass
        return None

    def _defang_visible_text(self, text: str) -> str:
        try:
            text = re.sub(
                r"```[\s\S]*?```",
                lambda m: self._zwsp_defang(
                    m.group(0), defang_fences=False, defang_think_tags=True
                ),
                text,
            )
            text = re.sub(
                r"(?<!`)`(?!`)[^`\n]+?(?<!`)`(?!`)",
                lambda m: self._zwsp_defang(
                    m.group(0), defang_fences=False, defang_think_tags=True
                ),
                text,
            )
            text = re.sub(
                r"((?<!`)<(?:think|thought)>(?!`))([\s\S]*?)((?<!`)</(?:think|thought)>(?!`))",
                lambda m: f"{m.group(1)}{self._zwsp_defang(m.group(2), defang_fences=True)}{m.group(3)}",
                text,
                flags=re.IGNORECASE,
            )
            if not re.search(
                r"(?<!`)<(?:think|thought)>(?!`)", text, re.IGNORECASE
            ) and re.search(r"(?<!`)</(?:think|thought)>(?!`)", text, re.IGNORECASE):
                parts = re.split(
                    r"((?<!`)</(?:think|thought)>(?!`))",
                    text,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )
                if len(parts) >= 3:
                    parts[0] = self._zwsp_defang(parts[0], defang_fences=True)
                    text = "".join(parts)
            if re.search(
                r"(?<!`)<(?:think|thought)>(?!`)", text, re.IGNORECASE
            ) and not re.search(
                r"(?<!`)</(?:think|thought)>(?!`)", text, re.IGNORECASE
            ):
                parts = re.split(
                    r"((?<!`)<(?:think|thought)>(?!`))",
                    text,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )
                if len(parts) == 3:
                    parts[2] = self._zwsp_defang(parts[2], defang_fences=True)
                    text = "".join(parts)
            text = self._zwsp_defang(
                text, defang_fences=False, defang_think_tags=False
            )
        except Exception:
            pass
        return text

    def _is_thought_only_cutoff(self, text: str) -> bool:
        has_reasoning = bool(
            re.search(
                r"<details\b[^>]*type=[\"']reasoning[\"']|<(?:think|thought)>",
                text,
                re.IGNORECASE,
            )
        )
        if not has_reasoning:
            return False
        visible_outside_thought = self._strip_thoughts_only(text)
        return len(visible_outside_thought) == 0

    def _is_truncated_mid_backtick(self, text: str) -> bool:
        stripped = re.sub(
            r"(?:\s*</(?:details|think|thought)>\s*)+$",
            "",
            text.rstrip(),
            flags=re.IGNORECASE,
        ).rstrip()
        if not stripped:
            return False
        if re.search(r"(?:^|[\s(\[{])`<?$", stripped):
            return True
        last_line = stripped.splitlines()[-1]
        last_line_clean = re.sub(r"^\s*>\s?", "", last_line)
        if (
            not last_line_clean.strip().startswith("```")
            and last_line_clean.count("`") % 2 == 1
            and stripped.endswith("`")
        ):
            return True
        return False

    def _clean_dangling_backtick(self, display_text: str) -> str:
        if re.search(r"</details>\s*$", display_text, flags=re.IGNORECASE):
            return re.sub(
                r"`<?(\s*</details>\s*)$",
                r"\1",
                display_text,
                count=1,
                flags=re.IGNORECASE,
            )
        return re.sub(r"`<?\s*$", "", display_text)

    def _unwrap_reasoning_for_api(self, text: str, fallback_reasoning: str = "") -> str:
        unwrapped = re.sub(
            r"<details\b[^>]*type=[\"']reasoning[\"'][^>]*>\s*<summary>.*?</summary>\s*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        unwrapped = re.sub(r"</details>", "\n", unwrapped, flags=re.IGNORECASE)
        unwrapped = re.sub(
            r"</?(?:think|thought)>", "\n", unwrapped, flags=re.IGNORECASE
        )
        unwrapped = re.sub(r"(?m)^>\s?", "", unwrapped).strip()
        if not unwrapped and fallback_reasoning.strip():
            return self._clean_dangling_backtick(fallback_reasoning.strip())
        return unwrapped

    def _run_command(self, cmd: str) -> str:
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.valves.command_timeout,
            )
            output = (res.stdout + res.stderr).strip()
            return output if output else "(Command completed with no output)"
        except subprocess.TimeoutExpired:
            return f"(Error: Command timed out after {self.valves.command_timeout}s)"
        except Exception as e:
            return f"(Error executing command: {str(e)})"

    def _tail_clip(self, text: str, limit: int) -> str:
        if not isinstance(text, str) or len(text) <= limit:
            return text or ""
        return "[...truncated...]\n" + text[-limit:]

    def _trim_for_subcall(self, messages: List[dict]) -> List[dict]:
        keep = max(2, int(self.valves.heal_context_messages or 12))
        trimmed = []
        for idx, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                continue
            content = m.get("content")
            if not isinstance(content, str):
                text = self._extract_text_from_parts(content) or ""
            else:
                text = content
            if not text.strip() and m.get("output"):
                text = self._extract_visible_from_output(m["output"])
            is_last = idx == len(messages) - 1
            if not text.strip():
                if not is_last:
                    continue
                text = ""
            trimmed.append({"role": role, "content": self._tail_clip(text, 8000)})
        if len(trimmed) > keep:
            trimmed = trimmed[:1] + trimmed[-keep:]
        return trimmed


    # ------------------------------------------------------------ v2.8.5 message cutoff
    CONTENT_ITEM_TYPES = ("message", "reasoning", "function_call", "function_call_output")

    @classmethod
    def _last_content_index(cls, items: List[dict]) -> int:
        """Index of the last real content item, skipping trailing non-content items
        (e.g. open_webui:context_compaction) that OWU appends to ~some turns."""
        for i in range(len(items) - 1, -1, -1):
            item = items[i]
            if isinstance(item, dict) and item.get("type") in cls.CONTENT_ITEM_TYPES:
                return i
        return -1

    @staticmethod
    def _has_unclosed_fence(text: str) -> bool:
        """True when the last code fence is left open.

        A real scanner, not a line count: parity counting misfires when a longer fence
        (````) wraps a bare ``` line, since that inner line is content, not a delimiter.
        A fence opens with >=3 of its marker at indent <=3 and only closes on a line of
        the same marker with >= the opening run and no info string.
        """
        for marker in ("`", "~"):
            open_run = None
            for line in text.splitlines():
                stripped = line.lstrip()
                if len(line) - len(stripped) > 3 or not stripped.startswith(marker * 3):
                    continue
                run = len(stripped) - len(stripped.lstrip(marker))
                if open_run is None:
                    open_run = run
                elif run >= open_run and stripped.lstrip(marker).strip() == "":
                    open_run = None
            if open_run is not None:
                return True
        return False

    @staticmethod
    def _ends_non_terminal(text: str) -> bool:
        """Definitively non-terminal endings only. Deliberately excludes ';', '-', ':'
        and em-dash: turns legitimately end on those (`kill -9 $PID;`, `---` rules,
        'paste your trace below:'), and a false positive appends rambles to a complete
        answer, which is worse than leaving one comma-truncated turn alone."""
        t = text.rstrip()
        return bool(t) and t.endswith((",", "[", "(", "{"))

    def _is_message_cutoff(
        self, last_msg: dict, output_items: List[dict], visible_text: str
    ) -> bool:
        """True when the turn's FINAL visible prose was cut off mid-thought.

        v2.8.4 healed only reasoning-shaped cutoffs: _inspect_output_for_cutoff
        explicitly returns False once a non-empty message item sits after the last
        reasoning item, so 'reasoning completed, message prose died mid-sentence'
        was never healed. There is no upstream signal for this class (observed real
        case: done=1, error=null, status [Turn complete], item status 'completed',
        89 completion tokens), so detection is heuristic by necessity.
        """
        if not output_items:
            return False
        if self._has_unanswered_tool_call(last_msg, output_items):
            return False
        idx = self._last_content_index(output_items)
        if idx < 0:
            return False
        item = output_items[idx]
        if item.get("type") != "message":
            return False
        if not self._extract_text_from_parts(item.get("content")).strip():
            return False
        if item.get("status") in ("incomplete", "in_progress", "failed"):
            return True
        if not visible_text.strip():
            return False
        return self._has_unclosed_fence(visible_text) or self._ends_non_terminal(
            visible_text
        )

    # ------------------------------------------------------- v2.8.6 hallucinated tool calls
    # Real observed shapes (from chat_message.content), shown defanged here:
    #   < tool_call> then <function=replace_file_path_note> with <parameter=x>...< /parameter>,
    #   then < /function> and < /tool_call>
    #   < tool_call> then <function=read> then < /function> < /tool_call>   (no params at all)
    #   < tool_call> then <function=ask_user_question> with <parameter=questions>[...]
    #   JSON shape: < tool_call>{"name": "run_terminal", "arguments": {...}}< /tool_call>
    # Stored rows may carry the ZWSP we inserted during defang, so every tag match allows one.
    # v2.8.7: horizontal whitespace after '<' and '</' is allowed now. The system prompt tells
    # the model to write '< tool_call>' with a space, and a block spaced on BOTH the wrapper and
    # the inner tag was previously invisible to the detector (measured miss, not hypothetical).
    # A reviewer proposed \s*, which crosses newlines and would let a stray '<' at a line end pair
    # with a 'tool_call>' far below, so this stays [ \t]*.
    OPEN_TAG = r"<(?:\u200b)?[ \t]*"
    CLOSE_TAG = r"<(?:\u200b)?[ \t]*/[ \t]*"

    TOOL_BLOCK_RE = re.compile(
        OPEN_TAG + r"tool_call>[\s\S]*?(?:" + CLOSE_TAG + r"tool_call>|\Z)"
        r"|" + OPEN_TAG + r"function\b[^>]*>[\s\S]*?(?:" + CLOSE_TAG + r"function>|\Z)"
        r"|" + OPEN_TAG + r"tool_use\b[^>]*>[\s\S]*?(?:" + CLOSE_TAG + r"tool_use>|\Z)"
        r"|" + OPEN_TAG + r"invoke\b[^>]*>[\s\S]*?(?:" + CLOSE_TAG + r"invoke>|\Z)",
        re.IGNORECASE,
    )

    # v2.8.10: attribute/JSON shapes were covered, XML-ELEMENT shapes were not.
    # Measured against the live 2.8.9 filter: a block like
    #   < tool_call>< function>ghost_tool</ function>< /tool_call>
    # matched TOOL_BLOCK_RE, extracted an EMPTY name, and `if not name: continue`
    # let it through untouched (audit evidence: poststrip raw_names=['', '', ''] in
    # the 'Test Email Feature' chat). Element style is what several chat templates
    # emit natively, so this was the likeliest real hallucination shape, not a
    # hypothetical one.
    #
    # The new alternatives demand that the identifier is the FIRST token inside the
    # tag and is immediately followed (space/tab only) by '<'. That is what keeps
    # prose safe: '< function>please call the ghost tool and see' cannot match,
    # because 'please' is not followed by '<'. Discussion of tags normally sits in a
    # code span anyway (_code_region_spans excludes those before this ever runs).
    # Accepted trade-off: prose naming an inactive tool in this exact tag form is now
    # stripped. That is the hallucination we are hunting, so it is the right call.
    TOOL_NAME_RE = re.compile(
        OPEN_TAG + r"function\s*=\s*([A-Za-z0-9_.:\[\]-]+)>"
        r"|" + OPEN_TAG + r"(?:function|invoke|tool_use)\s+name\s*=\s*[\"']([^\"']+)[\"']"
        r"|\"(?:name|tool|function)\"\s*:\s*\"([A-Za-z0-9_.:\[\]-]+)\""
        r"|" + OPEN_TAG
        + r"(?:function|invoke|tool_use|tool)[ \t]*>[ \t]*([A-Za-z0-9_.:\[\]-]+)[ \t]*(?=<)"
        r"|" + OPEN_TAG + r"name[ \t]*>[ \t]*([A-Za-z0-9_.:\[\]-]+)[ \t]*(?=<)"
        # v2.8.13: BARE-NAME shape. The 2.8.10 alternatives all require one of
        # function|invoke|tool_use|tool immediately before '>', so the tag text
        # 'tool_call>' fails at '_call>' and the whole set falls through. Measured
        # on the real emission from the abliterated model (a live chat,
        # originalContent): the block matched TOOL_BLOCK_RE, extracted '', and
        # `if not name: continue` passed it through untouched - so the invented
        # tool call was never rejected and no audit line was written.
        #
        # The guard is the lookahead at the end: after the name (plus optional
        # space/tab) the next char must be a PAYLOAD OPENER - '{', '(', or '<' -
        # or the end of the block. Prose cannot produce that, because the name
        # charset excludes whitespace, so 'a bare < tool_call>NAME is the shape'
        # stays intact: 'NAME' is followed by ' is', not by a payload opener.
        # Every real call form has one: one small local model wrote NAME{...}, element/no-argument
        # style is followed by '<', and a paren-wrapped argument list by '('. The
        # '$' branch catches the truncated-stream shape where the name is all that
        # got out. Measured on 2.8.13 with the class narrowed to [{(]: the
        # no-argument form '< tool_call>ghost_tool< /tool_call>' extracted '' and
        # fell through, which is why '<' is in the class.
        # Accepted trade-off, same as 2.8.10: unbackticked prose that contains the
        # full '< tool_call>NAME<payload>' block is stripped. That is the shape we
        # are hunting.
        r"|" + OPEN_TAG
        + r"(?!/)[a-z_][a-z0-9_]*[ \t]*>[ \t]*([A-Za-z0-9_.:\[\]-]+)[ \t]*(?=[{(<]|$)",
        re.IGNORECASE,
    )

    @staticmethod
    def _audit_line(text: str) -> None:
        """Append one diagnostic line. Best-effort; never raises."""
        try:
            with open("/app/backend/data/uf_audit.log", "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:
            pass

    def _active_tool_names(self, __metadata__: Optional[dict]) -> set:
        """Names of tools actually resolved for this turn.

        OWU writes metadata['tools'] = tools_dict during process_chat_payload
        (middleware ~line 2910), keyed by tool name and including deferred
        tool-search entries, so it is the authoritative 'active' set. The outlet
        runs after that, and __metadata__ is the same dict, so it is visible here.
        Inlet-provided tools and MCP tools outside that map are covered by the
        hallucinated_tool_allowlist valve. Empty set => feature disables itself.
        """
        names = set()
        if isinstance(__metadata__, dict):
            tools = __metadata__.get("tools")
            if isinstance(tools, dict):
                names.update(str(k) for k in tools.keys())
            elif isinstance(tools, list):
                for entry in tools:
                    if not isinstance(entry, dict):
                        continue
                    spec = entry.get("spec") or entry.get("function") or entry
                    name = spec.get("name") if isinstance(spec, dict) else None
                    if name:
                        names.add(str(name))
        for part in (self.valves.hallucinated_tool_allowlist or "").replace(",", " ").split():
            names.add(part.strip())
        return {n for n in names if n}

    def _code_region_spans(self, text: str) -> List[tuple]:
        """Spans that hold code / quoted markup / thoughts: never scan there.

        Prose ABOUT tags (backticked or fenced) is normal in this chat and must
        not be mistaken for a call, so exclusion is mandatory, not cosmetic.
        """
        spans = []
        for m in re.finditer(r"```[\s\S]*?```|~~~[\s\S]*?~~~", text):
            spans.append(m.span())
        for m in re.finditer(r"(?<!`)`(?!`)[^`\n]+?(?<!`)`(?!`)", text):
            spans.append(m.span())
        for m in re.finditer(
            r"<details\b[^>]*>[\s\S]*?</details>|<(?:\u200b)?(?:think|thought)>[\s\S]*?</(?:\u200b)?(?:think|thought)>",
            text,
            re.IGNORECASE,
        ):
            spans.append(m.span())
        # An unclosed fence means everything after it is code.
        last_fence = max((text.rfind("```"), text.rfind("~~~")))
        if last_fence != -1 and self._has_unclosed_fence(text[: last_fence + 3]):
            spans.append((last_fence, len(text)))
        return spans

    def _tool_name_from_block(self, block: str) -> str:
        for m in self.TOOL_NAME_RE.finditer(block):
            for group in m.groups():
                if group and group.strip():
                    return group.strip()
        return ""

    def _find_hallucinated_blocks(
        self, text: str, allowed: set
    ) -> List[Tuple[int, int, str]]:
        """Tool-call blocks outside code regions whose tool name is not active."""
        if not text:
            return []
        code_spans = self._code_region_spans(text)
        allowed_lower = {n.lower() for n in allowed}
        found = []
        for m in self.TOOL_BLOCK_RE.finditer(text):
            start, end = m.span()
            if any(start < ce and end > cs for cs, ce in code_spans):
                continue
            name = self._tool_name_from_block(m.group(0))
            if not name:
                continue
            if name.lower() in allowed_lower:
                continue
            found.append((start, end, name))
        return found

    def _strip_hallucinated_blocks(
        self, text: str, allowed: set
    ) -> Tuple[str, List[str]]:
        blocks = self._find_hallucinated_blocks(text, allowed)
        if not blocks:
            return text, []
        names = []
        cleaned = text
        for start, end, name in reversed(blocks):
            names.append(name)
            cleaned = cleaned[:start] + cleaned[end:]
        names.reverse()
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, names

    @staticmethod
    def _push_sentinel_user_turn(
        loop_messages: List[dict], content: str
    ) -> None:
        """Append an internal instruction turn, replacing the last one if it is already an
        internal instruction turn.

        Measured: with two rejections in a row the payload ended up
        ['user','assistant','user','user','user']. Once the first reject prompt popped the
        empty assistant turn it became the last message, so the next reject prompt stacked.
        Replacing (rather than appending) keeps the newest instruction and a bounded prompt.
        The command-output append deliberately does NOT use this: a tool-result turn is real
        context, and dropping it would be the corrupting direction.
        """
        if (
            loop_messages
            and loop_messages[-1].get("role") == "user"
            and SELF_CALL_SENTINEL in (loop_messages[-1].get("content") or "")
        ):
            loop_messages[-1]["content"] = content
            return
        loop_messages.append({"role": "user", "content": content})

    def _merge_sentinel_user_turn(
        self, loop_messages: List[dict], content: str
    ) -> None:
        """v2.8.15 fix A: concatenate instead of replace.

        _push_sentinel_user_turn replaces, which is right when one instruction simply
        supersedes another. It is wrong when a turn needs BOTH a reject notice and a
        heal instruction: replacing loses one of them, and appending stacks two user
        turns so the model has to guess which is live.
        """
        if (
            loop_messages
            and loop_messages[-1].get("role") == "user"
            and (loop_messages[-1].get("content") or "").startswith(SELF_CALL_SENTINEL)
        ):
            prior = (loop_messages[-1]["content"] or "")[len(SELF_CALL_SENTINEL):].strip()
            extra = content
            if extra.startswith(SELF_CALL_SENTINEL):
                extra = extra[len(SELF_CALL_SENTINEL):].strip()
            loop_messages[-1]["content"] = (
                SELF_CALL_SENTINEL + prior + "\n\n" + extra
            )
            return
        loop_messages.append({"role": "user", "content": content})

    def _strip_all_tool_blocks(self, text: str) -> Tuple[str, List[str]]:
        """Strip every tool-call-shaped block regardless of the tool it names.

        Used only on the outlet continuation sub-call's reply: that request carries no
        tool_ids, so nothing is bound to it and ANY call shape in it is dead markup, even
        when it names a tool that is active in the parent turn. Code regions stay excluded.
        Names come back so the caller decides which of them are genuinely unavailable.
        """
        return self._strip_hallucinated_blocks(text, set())

    def _scrub_hallucinated_in_output(
        self, output_items: List[dict], allowed: set
    ) -> List[str]:
        """Remove dead markup from every message item, not just the last one.

        Gate-3 keeps earlier message items intact, so a hallucinated block sitting
        in mid-turn narration would survive the final-content rewrite otherwise.
        """
        removed = []
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            parts = item.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                for key in ("text", "content"):
                    val = part.get(key)
                    if isinstance(val, str) and val.strip():
                        cleaned, names = self._strip_hallucinated_blocks(val, allowed)
                        if names:
                            removed.extend(names)
                            part[key] = cleaned
        return removed

    async def _call_owui_async(self, payload: dict, headers: dict) -> dict:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    self.valves.owui_api_url, json=payload, headers=headers
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.ConnectError:
                alt_url = (
                    self.valves.owui_api_url.replace(":8080", ":3000")
                    if ":8080" in self.valves.owui_api_url
                    else self.valves.owui_api_url.replace(":3000", ":8080")
                )
                resp = await client.post(alt_url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> dict:
        messages = body.get("messages", [])
        if (
            not messages
            or messages[-1].get("role") != "assistant"
            or self._is_self_call(messages)
        ):
            return body

        last_msg = messages[-1]
        model_id = body.get("model", "default")

        # v2.8.11: carry the turn's tool binding into the sub-call.
        # process_chat_payload pops tool_ids/terminal_id out of form_data early
        # (middleware ~2407-2408) and re-publishes them on metadata (~2635). The
        # outlet runs after that point, so body no longer has them but
        # __metadata__ does. Without this, every heal/reject sub-call is a bare
        # completion with zero tools bound, which is why the reject prompt had to
        # forbid call markup - any markup it produced was inert text. With the
        # ids forwarded, OWU's tool handler binds the same tools to the retry and
        # a corrected call actually executes.
        retry_tool_ids: List[Any] = []
        retry_terminal_id = None
        if isinstance(__metadata__, dict):
            ids = __metadata__.get("tool_ids")
            if isinstance(ids, list):
                retry_tool_ids = ids
            retry_terminal_id = __metadata__.get("terminal_id")
        latest_chunk = last_msg.get("content") or ""
        output_items = await self._get_output_items(last_msg, __metadata__)

        accumulated_display = self._defang_visible_text(latest_chunk)
        healed_deltas: List[str] = []
        loop_messages = self._trim_for_subcall(messages)
        in_turn_tool_summary = self._extract_tool_outputs_from_output(output_items)
        did_modify = False

        # v2.8.9: OWU rebuilds `content` as the join of every message item, so
        # defanging has to happen on the ITEMS, not only on the cumulative
        # string. Otherwise plaintext tool markup in an interim bridge item can
        # resurface at save time and trip the client's stream parser.
        def _defang_items_in_place(items: List[dict]) -> None:
            for item in items:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "message"
                    and item.get("content")
                ):
                    txt = self._extract_text_from_parts(item.get("content"))
                    if txt:
                        defanged = self._defang_visible_text(txt)
                        if defanged != txt:
                            item["content"] = [
                                {"type": "output_text", "text": defanged}
                            ]

        _defang_items_in_place(output_items)

        # v2.8.8: the model sometimes opens its final answer by replaying the
        # interim narration lines it already said between tool calls. OWU stores
        # `content` as '\\n'.join() over every message item, so the replayed copy
        # shows up as the same paragraphs printed twice. Strip the replay from
        # both the visible string and the final item - each removed span still
        # exists verbatim as its own item, so the original is never lost.
        if self.valves.collapse_echoed_bridge_lines:
            msg_indices = [
                i
                for i, item in enumerate(output_items)
                if isinstance(item, dict)
                and item.get("type") == "message"
                and self._extract_text_from_parts(item.get("content")).strip()
            ]
            if len(msg_indices) > 1:
                interim_texts = [
                    self._extract_text_from_parts(output_items[i].get("content")).strip()
                    for i in msg_indices[:-1]
                ]

                collapsed = self._collapse_echoed_bridge(
                    accumulated_display, interim_texts
                )
                if collapsed != accumulated_display:
                    accumulated_display = collapsed
                    did_modify = True

                final_idx = msg_indices[-1]
                final_text = self._extract_text_from_parts(
                    output_items[final_idx].get("content")
                )
                collapsed_final = self._collapse_echoed_bridge(
                    final_text, interim_texts
                )
                if collapsed_final != final_text.strip():
                    output_items[final_idx]["content"] = [
                        {"type": "output_text", "text": collapsed_final}
                    ]
                    did_modify = True

        # --- v2.8.6: reject tool calls naming tools that are not active this turn.
        # If the active set cannot be resolved, stay inert rather than strip everything.
        allowed_tools = (
            self._active_tool_names(__metadata__)
            if self.valves.reject_hallucinated_tools
            else set()
        )
        halluc_names: List[str] = []
        if self.valves.debug_halluc_audit:
            cid = "?"
            if isinstance(__metadata__, dict):
                cid = str(__metadata__.get("chat_id", "?"))[:16]
            # v2.8.11: prove the retry-binding assumption instead of trusting it.
            # If tool_ids is empty at outlet time, the reject-with-tools path can
            # never arm and the prose fallback is the only live branch.
            self._audit_line(
                "binding: chat=%s tool_ids=%d terminal=%r" % (
                    cid, len(retry_tool_ids), retry_terminal_id
                )
            )
            self._audit_line(
                "chat=%s allowed=%r raw_blocks=%d code_spans=%d chunk_len=%d" % (
                    cid,
                    sorted(allowed_tools),
                    len(self.TOOL_BLOCK_RE.findall(latest_chunk)),
                    len(self._code_region_spans(latest_chunk)),
                    len(latest_chunk),
                )
            )
        if allowed_tools:
            latest_chunk, halluc_names = self._strip_hallucinated_blocks(
                latest_chunk, allowed_tools
            )
            accumulated_display, display_names = self._strip_hallucinated_blocks(
                accumulated_display, allowed_tools
            )
            halluc_names = sorted(set(halluc_names) | set(display_names))
            if halluc_names and output_items:
                self._scrub_hallucinated_in_output(output_items, allowed_tools)
        if self.valves.debug_halluc_audit:
            raw_names = [self._tool_name_from_block(m.group(0)) for m in self.TOOL_BLOCK_RE.finditer(latest_chunk)]
            self._audit_line(
                "poststrip: raw_names=%r halluc=%r chunk_len=%d" % (
                    raw_names, halluc_names, len(latest_chunk)
                )
            )

        for step in range(1, self.valves.max_loops + 1):
            is_hallucinated_tool = bool(allowed_tools and halluc_names)
            is_mid_backtick = (
                self.valves.heal_truncated_streams
                and self._is_truncated_mid_backtick(latest_chunk)
            )
            is_inline_thought_cutoff = (
                self.valves.heal_truncated_streams
                and self._is_thought_only_cutoff(latest_chunk)
            )
            is_message_cutoff = bool(
                self.valves.heal_truncated_streams
                and step == 1
                and self._is_message_cutoff(
                    last_msg,
                    output_items,
                    self._strip_thoughts_only(latest_chunk),
                )
            )
            if step == 1:
                is_output_reasoning_cutoff, separate_reasoning_text = (
                    self._inspect_output_for_cutoff(
                        last_msg, output_items, latest_chunk
                    )
                    if self.valves.heal_truncated_streams
                    else (False, "")
                )
            else:
                is_output_reasoning_cutoff, separate_reasoning_text = (
                    self.valves.heal_truncated_streams and not latest_chunk.strip(),
                    "",
                )

            # v2.8.15 fix A: reject and cutoff-heal used to be `if / elif`, which made
            # the two failure modes mutually exclusive. Proven failure (a live chat,
            # a live message): a plaintext call naming `replace_file` AND a truncated
            # reasoning item (status: incomplete) in the same turn. Reject won the `if`
            # and the heal never ran. They are now evaluated independently, and when
            # both fire the two instructions are merged into one sentinel turn.
            needs_heal = bool(
                is_mid_backtick
                or is_inline_thought_cutoff
                or is_output_reasoning_cutoff
                or is_message_cutoff
            )
            sentinel_pushed = False
            cmd_to_run = None
            if is_hallucinated_tool:
                # v2.8.6: dead markup already stripped from content and output items.
                # Tell the model which tool was rejected so it can pick a real one and
                # keep the turn going, instead of ending on an unparsed call.
                did_modify = True
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": (
                                    f"Rejected unavailable tool call (Step {step}): "
                                    f"{', '.join(halluc_names[:4])}"
                                ),
                                "done": False,
                            },
                        }
                    )
                note = (
                    f"_(Tool call rejected: {', '.join(halluc_names)} "
                    "is not available in this session.)_"
                )
                # v2.8.7: a follow-up plaintext call naming an ACTIVE tool is also dead
                # markup, but naming it in this note would be a lie, so the visible text
                # stays generic whenever the rejected names are all active ones.
                _allowed_lower = {n.lower() for n in allowed_tools}
                note_names = sorted(
                    {n for n in halluc_names if n.lower() not in _allowed_lower}
                )
                if not note_names:
                    note = (
                        "_(Tool call rejected: it was emitted as plain text, "
                        "so nothing was executed.)_"
                    )
                if accumulated_display.strip():
                    accumulated_display += "\n\n" + note
                else:
                    # Dead block was the entire message: keep a visible trace in the
                    # final content, since the transient status event alone vanishes.
                    accumulated_display = note
                healed_deltas.append(note)
                # Keep the model's own prose (minus dead markup) as its last turn, so the
                # correction prompt continues from what it actually said.
                if loop_messages and loop_messages[-1].get("role") == "assistant":
                    visible_only = self._strip_thoughts_only(latest_chunk)
                    if visible_only.strip():
                        loop_messages[-1]["content"] = self._tail_clip(
                            visible_only.strip(), 1200
                        )
                    else:
                        loop_messages.pop()
                # v2.8.11: the old wording ("finish in prose, do NOT emit any call
                # markup") was self-sabotage. Measured: it made the model announce "no
                # tools in this turn, my manifest isn't in front of me" and abandon the
                # task. That wording was only correct while the sub-call carried no
                # tool_ids. When the binding IS forwarded, the right move is the obvious
                # one: name the mistake, name the valid alternatives, re-issue.
                # v2.8.15 fix B: v2.8.11 forwarded the session's tool_ids into this
                # retry and told the model "the tools are bound and available to you
                # right now - emit the call again". That claim was false, and it is what
                # killed the turn. OWU's tool loop lives in its own websocket chat
                # handler, so a REST /api/chat/completions sub-call returns one
                # completion that nobody executes. Proven by grep: the literal name
                # `tool_calls` appears at exactly two lines in this whole filter, both
                # about `last_msg`; the sub-call handler reads only choice_msg["content"].
                # A native call in the reply is discarded on the floor, so that prompt
                # instructed the model straight into a dead end. Binding is gone; the
                # prompt now names only paths this filter can actually honour: prose, or
                # a fenced ```bash command the else-branch below really does run.
                self._push_sentinel_user_turn(
                    loop_messages,
                    (
                        f"{SELF_CALL_SENTINEL}[System: tool call rejected - correct it and continue]\n"
                        f"You emitted a plain-text tool call for "
                        f"{', '.join(halluc_names)}, and it did nothing, because plain XML "
                        "tags in message text are never executed. Nothing ran, so no result "
                        "exists and nothing on disk changed.\n"
                        "Pick one of these, and be honest about what each can do:\n"
                        "1. Answer in plain prose from what you already know.\n"
                        "2. If a step genuinely needs a shell command, put it in a ```bash "
                        "fenced block. That block WILL be run and its output handed back to "
                        "you in the next message. Shell only - it cannot edit files, write "
                        "code, search the web, or invoke any other tool, so never use it to "
                        "pretend an unrunnable step ran.\n"
                        "3. If the step needs something that is not a shell command, say so "
                        "plainly and stop. Naming a tool in prose does not run it.\n"
                        "Do not write tool-call XML in this reply, do not restate the "
                        "rejected call, and do not apologize.\n"
                        "Do NOT output `[Thought]` or `[Tool Call]` headers in your visible message, "
                        "and ALWAYS put a space after '<' (write `< tool_call>` and `< /think>`) when referring to tags as text."
                    ),
                )
                sentinel_pushed = True
                halluc_names = []
            if needs_heal:
                did_modify = True
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": (
                                    f"Healing truncated message (Step {step})..."
                                    if is_message_cutoff
                                    else f"Healing mid-reasoning/stream cutoff (Step {step})..."
                                ),
                                "done": False,
                            },
                        }
                    )

                if is_mid_backtick:
                    accumulated_display = self._clean_dangling_backtick(
                        accumulated_display
                    )
                    latest_chunk_cleaned = self._clean_dangling_backtick(latest_chunk)
                else:
                    latest_chunk_cleaned = latest_chunk

                visible_only = self._strip_thoughts_only(latest_chunk_cleaned)
                if is_output_reasoning_cutoff and separate_reasoning_text.strip():
                    resume_from = separate_reasoning_text
                elif is_message_cutoff:
                    resume_from = latest_chunk_cleaned
                else:
                    resume_from = self._unwrap_reasoning_for_api(
                        latest_chunk_cleaned,
                        fallback_reasoning=separate_reasoning_text,
                    )

                # v2.8.1: Do not put "[Thought So Far]:" inside the assistant message content,
                # as that primes the model to output "[Thought] ... [Tool Call]" in visible text.
                if loop_messages and loop_messages[-1].get("role") == "assistant":
                    if visible_only.strip():
                        loop_messages[-1]["content"] = self._tail_clip(
                            visible_only.strip(), 1200
                        )
                    else:
                        loop_messages.pop()

                tail_context = self._tail_clip(resume_from, 1000)
                tool_ctx_block = (
                    f"\n\nTool results already gathered in this turn:\n{in_turn_tool_summary}\n"
                    if in_turn_tool_summary
                    else ""
                )

                if is_message_cutoff:
                    heal_instruction = (
                        f"{SELF_CALL_SENTINEL}Your final message was cut off mid-sentence by the server, "
                        f"not by choice. Here is exactly how your visible message ended:\n"
                        f"---\n{tail_context}\n---"
                        f"{tool_ctx_block}\n"
                        "Continue that same message with ONLY the remaining text, starting so it splices "
                        "seamlessly onto the tail above (finish the interrupted sentence first). "
                        "Do not repeat, restate, rephrase, or summarize anything in the tail. "
                        "Do not add a greeting, apology, heading, or new section. "
                        "Do NOT output `[Thought]` or `[Tool Call]` headers in your visible message, "
                        "and ALWAYS put a space after '<' (write `< tool_call>` and `< /think>`) when referring to tags as text."
                    )
                else:
                    heal_instruction = (
                        f"{SELF_CALL_SENTINEL}Your stream was cut off mid-reasoning. "
                        f"Here is the tail of your internal reasoning before the cutoff:\n"
                        f"---\n{tail_context}\n---"
                        f"{tool_ctx_block}\n"
                        "Now write ONLY the final user-facing response directly answering the user. "
                        "Continue with ONLY NEW text: do not repeat, restate, or re-quote anything you "
                        "already wrote before the cutoff. "
                        "Do NOT output `[Thought]` or `[Tool Call]` headers in your visible message, "
                        "and ALWAYS put a space after '<' (write `< tool_call>` and `< /think>`) when referring to tags as text."
                    )

                # fix A: when reject already armed an instruction turn this iteration,
                # merge rather than append. Stacked sentinel turns made the model lose
                # track of which instruction was live - visible in the failed turn's own
                # reasoning: "the tool call for replace_file_content was rejected ... that
                # appeared in my previous turn's output apparently".
                if sentinel_pushed:
                    self._merge_sentinel_user_turn(loop_messages, heal_instruction)
                else:
                    loop_messages.append(
                        {"role": "user", "content": heal_instruction}
                    )
            else:
                actionable = self._strip_thoughts_and_backticks(latest_chunk)
                cmd_to_run = self._extract_plaintext_command(actionable)
                if not cmd_to_run and not sentinel_pushed:
                    break
                # fix A/B: a reject with no cutoff to heal still has a live instruction
                # turn waiting, so falling through to the sub-call is what lets the turn
                # continue instead of ending on the rejection note. The whole command
                # block sits under `if cmd_to_run:` - an unguarded fall-through would
                # otherwise reach _run_command(None).
                if cmd_to_run:
                    did_modify = True
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Running command (Step {step}): {cmd_to_run[:40]}...",
                                    "done": False,
                                },
                            }
                        )

                    cmd_output = self._run_command(cmd_to_run)
                    if self.valves.show_command_output:
                        sep = "\n\n" if accumulated_display.strip() else ""
                        cmd_block = f"{sep}---\n**Executed Output:**\n```text\n{cmd_output}\n```\n---\n\n"
                        accumulated_display += cmd_block
                        healed_deltas.append(cmd_block.strip())
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "replace",
                                    "data": {"content": accumulated_display},
                                }
                            )

                    loop_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{SELF_CALL_SENTINEL}[Command Output]:\n{cmd_output}\n\n"
                                "Use this output to continue and finish answering the original request."
                            ),
                        }
                    )

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.valves.owui_api_key}",
                }
                payload = {
                    "model": model_id,
                    "messages": loop_messages,
                    "stream": False,
                }
                # v2.8.15 fix B: no tool_ids / terminal_id forwarding any more. A REST
                # /api/chat/completions sub-call returns one completion and nothing
                # downstream executes message.tool_calls, so binding only produced a
                # reply whose only useful field we threw away.
                data = await self._call_owui_async(payload, headers)
                choice = (data.get("choices") or [{}])[0]
                choice_msg = choice.get("message") or {}

                # -------------------------------------------------- v2.8.15 fix C
                # A continuation that came back empty used to burn a loop iteration in
                # total silence (max_loops=3, so up to three rounds with nothing on
                # screen - what was actually watched happen). Every continuation is now
                # logged, and both failure signatures get a standing WARNING line.
                resp_content = choice_msg.get("content") or ""
                resp_reasoning = choice_msg.get("reasoning_content") or ""
                resp_tool_calls = choice_msg.get("tool_calls") or []
                if self.valves.debug_halluc_audit:
                    self._audit_line(
                        "subcall: step=%d content=%d reasoning=%d tool_calls=%d finish=%r msgs=%d"
                        % (
                            step,
                            len(resp_content),
                            len(resp_reasoning),
                            len(resp_tool_calls),
                            choice.get("finish_reason"),
                            len(loop_messages),
                        )
                    )
                    if resp_tool_calls:
                        self._audit_line(
                            "subcall: step=%d WARNING response carried %d tool_calls this "
                            "filter cannot execute; discarded (fix B)"
                            % (step, len(resp_tool_calls))
                        )
                    if not resp_content.strip():
                        self._audit_line(
                            "subcall: step=%d WARNING empty continuation, reasoning=%d "
                            "finish=%r - burned a loop step"
                            % (step, len(resp_reasoning), choice.get("finish_reason"))
                        )

                latest_chunk = self._strip_Echoed_pseudo_tags(resp_content)
                # v2.8.6: the continuation itself may hallucinate again; re-arm the
                # detector so the next loop iteration rejects it (bounded by max_loops).
                # v2.8.7: this reply comes from OUR sub-call, where no tools are bound, so
                # a plaintext call naming an ACTIVE tool is equally dead. Measured leak: the
                # reject prompt used to say "use one of the available tools", the model
                # obeyed with <function=read_file> markup, and it landed in the final message.
                # So strip every call shape here unless it is a runnable plaintext command
                # (which the else-branch would execute), and re-arm on dead names only, never
                # on active ones - the reviewer's version set halluc_names from a blanket strip and
                # would have announced a live tool as unavailable.
                # v2.8.15 fix B: the old `if reject_with_tools` case existed only because
                # tools were bound into the retry. They are not any more, so that branch
                # is gone and this general handling applies to every continuation.
                if allowed_tools:
                    if self._extract_plaintext_command(
                        self._strip_thoughts_and_backticks(latest_chunk)
                    ):
                        latest_chunk, halluc_names = self._strip_hallucinated_blocks(
                            latest_chunk, allowed_tools
                        )
                    else:
                        latest_chunk, dead_any = self._strip_all_tool_blocks(
                            latest_chunk
                        )
                        _allowed_lower = {n.lower() for n in allowed_tools}
                        halluc_names = [
                            n for n in dead_any if n.lower() not in _allowed_lower
                        ]
                if not latest_chunk.strip() and choice_msg.get("reasoning_content"):
                    separate_reasoning_text = choice_msg.get("reasoning_content", "")
            except Exception as e:
                reason = str(e).strip() or type(e).__name__
                err_note = f"\n*(Auto-continue failed: {reason})*"
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "replace",
                            "data": {"content": accumulated_display + err_note},
                        }
                    )
                break

            defanged_chunk = self._dedupe_continuation(
                accumulated_display, self._defang_visible_text(latest_chunk)
            )
            if defanged_chunk.strip():
                healed_deltas.append(defanged_chunk.strip())
                if accumulated_display.strip():
                    accumulated_display += "\n\n" + defanged_chunk
                else:
                    accumulated_display = defanged_chunk

            loop_messages.append(
                {
                    "role": "assistant",
                    "content": defanged_chunk.strip() or separate_reasoning_text[-800:],
                }
            )

            if __event_emitter__ and accumulated_display.strip():
                await __event_emitter__(
                    {"type": "replace", "data": {"content": accumulated_display}}
                )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Turn complete", "done": True},
                }
            )

        body["messages"][-1]["content"] = accumulated_display
        needs_status_flip = any(
            isinstance(item, dict)
            and item.get("status") in ("incomplete", "in_progress", "failed")
            for item in output_items
        )
        if (did_modify or output_items) and (
            accumulated_display.strip() or needs_status_flip
        ):
            # v2.8.9: the final item gets its OWN text plus this turn's filter
            # deltas - never the cumulative display. Interim bridge lines are
            # already separate items; writing the whole display here is what
            # made every bridge line render twice.
            final_item_text: Optional[str] = None
            final_msg_idx = self._last_index_of(output_items, ("message",))
            if final_msg_idx >= 0:
                own = self._extract_text_from_parts(
                    output_items[final_msg_idx].get("content")
                ).strip()
                parts = [own] if own else []
                parts.extend(healed_deltas)
                final_item_text = "\n\n".join(parts).strip()
            body["messages"][-1]["output"] = self._sync_structured_output(
                output_items,
                accumulated_display,
                healed_delta="\n\n".join(healed_deltas),
                final_item_text=final_item_text,
            )

        # HEAL VISIBILITY FIX (v2.8.19): the healed body reaches the DB but
        # never repaints a mounted view. Two client facts from the bundle:
        #   * type:"replace" mutates a local message object with no store
        #     reassignment -> inert once the message is finalized.
        #   * chat:outlet repaints, but is guarded on the message id being
        #     present in the local store and on content differing; it has
        #     not been observed to land here.
        # chat:reload, by contrast, dispatches BEFORE any message-id lookup:
        # the handler is `await Rs(); return` - a full chat refetch, which is
        # exactly the tab-out/tab-back action that makes text appear. OWU
        # itself uses this shape (utils/subagents.py:232, utils/timers.py:374).
        # Ordering constraint: middleware persists our healed body AFTER
        # outlet() returns (middleware.py ~3874), immediately before its own
        # chat:outlet emit. So the reload must not fire inline - it would
        # refetch stale content. It runs as a detached task with a short
        # delay so the persist has landed.
        _reload_delay = float(
            getattr(self.valves, "post_heal_reload_seconds", 1.5) or 0.0
        )
        if did_modify and __event_emitter__ is not None and _reload_delay > 0:

            async def _post_heal_reload(delay: float) -> None:
                try:
                    await asyncio.sleep(delay)
                    await __event_emitter__({"type": "chat:reload"})
                    self._audit_line(
                        "RELOAD_EMIT t=%.3f: fired chat:reload %.1fs after "
                        "outlet return (healed body should refetch)"
                        % (time.time(), delay)
                    )
                except asyncio.CancelledError:
                    self._audit_line(
                        "RELOAD_EMIT: task cancelled before firing - request "
                        "task group tore down first"
                    )
                    raise
                except Exception as exc:  # never break the turn
                    self._audit_line(
                        "RELOAD_EMIT failed: %s: %s" % (type(exc).__name__, exc)
                    )

            # The event loop keeps only a WEAK reference to tasks, so a task
            # sleeping 1.5s can be garbage-collected mid-sleep and never fire.
            # Module-level strong set + discard-on-done is the documented fix.
            task = asyncio.create_task(_post_heal_reload(_reload_delay))
            _PENDING_RELOAD_TASKS.add(task)
            task.add_done_callback(_PENDING_RELOAD_TASKS.discard)
        # --------------------------------------------------------------

        return body
