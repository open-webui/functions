"""
title: Token Price Checker
description: Check OpenRouter model token prices (per 1M), detect price changes (⬆/⬇), and find comparable models. Commands: /price, /price comparable <model>, /balance, /cheapest.
author: Your Name
version: 0.3.2
requirements: requests
"""

import os, json, asyncio, difflib, requests
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field

OPENROUTER_SINGLE = "https://openrouter.ai/api/v1/model/{}"
OPENROUTER_ALL = "https://openrouter.ai/api/v1/models"
OPENROUTER_KEY = "https://openrouter.ai/api/v1/key"
CACHE_FILE = os.path.join(os.getcwd(), ".cache", "token_prices_v2.json")
FULL_CACHE = os.path.join(os.getcwd(), ".cache", "openrouter_models_cache.json")
CACHE_HOURS = 6
FUZZY_CUTOFF = 0.6


def _load(p):
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except:
        return {}


def _save(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f, indent=2)


def _per_1m(s):
    if not s:
        return None
    try:
        v = float(s)
        return None if v < 0 else round(v * 1_000_000, 4)
    except:
        return None


def _fmt(v, arrow=""):
    if v is None:
        return "—"
    if v == 0:
        return "**Free**"
    return f"{arrow}${v:.4f}"


def _change_tag(old, new):
    """Returns (arrow_char, change_text)."""
    if old is None or new is None:
        return ("", "")
    if old == 0 and new > 0:
        return ("⬆", " **⬆ FORMERLY FREE**")
    d = new - old
    if d > 0.001:
        return ("⬆", f" **⬆ +${d:.4f}**")
    if d < -0.001:
        return ("⬇", f" **⬇ {d:.4f}**")
    return ("", "")


def _fetch_one(mid):
    try:
        r = requests.get(OPENROUTER_SINGLE.format(mid), timeout=15)
        r.raise_for_status()
        return r.json().get("data")
    except:
        return None


def _fetch_all(force=False):
    c = _load(FULL_CACHE)
    if not force and c.get("fetched_at"):
        try:
            if datetime.now() - datetime.fromisoformat(c["fetched_at"]) < timedelta(
                hours=CACHE_HOURS
            ):
                return c.get("models", [])
        except:
            pass
    try:
        r = requests.get(OPENROUTER_ALL, timeout=60, stream=True)
        r.raise_for_status()
        data = r.json().get("data", [])
        _save(FULL_CACHE, {"fetched_at": datetime.now().isoformat(), "models": data})
        return data
    except:
        return c.get("models", [])


def _resolve_model(name, all_models):
    if not name:
        return None
    name = name.strip()
    lookup = {m["id"]: m for m in all_models}
    if name in lookup:
        return name
    matches = difflib.get_close_matches(
        name, list(lookup.keys()), n=5, cutoff=FUZZY_CUTOFF
    )
    return matches[0] if matches else None


def _fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


FOOTER = (
    "\n---\n"
    "Available commands:\n"
    "- /price comparable <model> - find similar models at lower cost\n"
    "- /balance - check your OpenRouter credit balance\n"
    "- /cheapest - show top 10 cheapest models"
)


class Tools:
    class Valves(BaseModel):
        preselected_models: str = Field(
            default="openai/gpt-4o,openai/gpt-4o-mini,anthropic/claude-sonnet-4.6,anthropic/claude-haiku-4.5,google/gemini-2.5-flash,google/gemini-2.5-pro,meta-llama/llama-3.1-70b-instruct,mistralai/mistral-small-3.1-24b-instruct,deepseek/deepseek-chat,qwen/qwen-plus",
            description="Comma-separated list of OpenRouter model IDs to track",
        )
        performance_tolerance: int = Field(
            default=10, description="Performance tolerance (points)"
        )
        openrouter_api_key: str = Field(
            default="",
            description="OpenRouter API key for /balance. Use $ENV_VAR_NAME to load from a Docker environment variable instead of the raw key.",
        )

    class UserValves(BaseModel):
        openrouter_api_key: str = Field(
            default="",
            description="Overrides the tool-level key for /balance. Use $ENV_VAR_NAME to load from a Docker environment variable.",
        )

    def __init__(self):
        self.citation = True
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    @staticmethod
    def _resolve_key(raw_key):
        if not raw_key:
            return ""
        s = raw_key.strip()
        if s.startswith("$"):
            env_name = s[1:]
            env_val = os.environ.get(env_name)
            if env_val:
                return env_val
            return raw_key
        return raw_key

    async def fetch_prices(self, message="", __event_emitter__=None, __user__=None):
        """Check prices (/price), compare (/price comparable <model>), balance (/balance), cheapest (/cheapest)."""
        api_key = self._resolve_key(self.valves.openrouter_api_key)

        uk = (__user__ or {}).get("valves") or self.user_valves
        uk_raw = getattr(uk, "openrouter_api_key", "") or self.user_valves.openrouter_api_key
        uk_key = self._resolve_key(uk_raw)
        if uk_key:
            api_key = uk_key

        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": "Working...", "done": False}}
            )
            await asyncio.sleep(0)

        parts = message.strip().split()
        cmd = parts[0].lower() if parts else "/price"
        sub = parts[1].lower() if len(parts) >= 2 else ""

        try:
            if cmd in ("/balance", "balance"):
                return self._check_balance(api_key)
            if cmd in ("/cheapest", "cheapest"):
                return self._cheapest()
            if cmd in ("/price", "price", "prices"):
                if sub in ("comparable", "compare", "similar"):
                    return self._comparable(
                        " ".join(parts[2:]) if len(parts) >= 3 else ""
                    )
                return self._price_report(api_key)
            return self._price_report(api_key)
        finally:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Done", "done": True, "hidden": True},
                    }
                )

    def _price_report(self, api_key):
        cache = _load(CACHE_FILE)
        now = datetime.now().isoformat(timespec="minutes")
        mids = [
            m.strip() for m in self.valves.preselected_models.split(",") if m.strip()
        ]
        L = [f"## Token Prices\n{now}\n"]
        rows, upd, errs = [], {}, []

        for mid in mids:
            m = _fetch_one(mid)
            if not m:
                errs.append(mid)
                continue
            p = m.get("pricing", {})
            inp = _per_1m(p.get("prompt"))
            out = _per_1m(p.get("completion"))
            c = cache.get(mid, {})
            op = c.get("prompt_1m")
            oc = c.get("completion_1m")

            inp_arrow, pi = _change_tag(op, inp)
            out_arrow, co = _change_tag(oc, out)

            change_text = ""
            parts_chg = []
            if pi:
                parts_chg.append(f"In{pi}")
            if co:
                parts_chg.append(f"Out{co}")
            if parts_chg:
                change_text = ", ".join(parts_chg)

            rows.append(
                [
                    mid,
                    _fmt(inp, inp_arrow + " " if inp_arrow else ""),
                    _fmt(out, out_arrow + " " if out_arrow else ""),
                    change_text,
                ]
            )
            upd[mid] = {"prompt_1m": inp, "completion_1m": out, "last_updated": now}

        if rows:
            L.append(_fmt_table(rows, ["Model", "Input /1M", "Output /1M", "Change"]))
        if errs:
            L.append(f"\nErrors: {', '.join(errs)}")
        _save(CACHE_FILE, upd)

        chg = []
        for mid, cur in upd.items():
            o = cache.get(mid, {})
            if not o:
                continue
            for k, lbl in [("prompt_1m", "Input"), ("completion_1m", "Output")]:
                ov, nv = o.get(k), cur.get(k)
                if ov is not None and nv is not None:
                    if ov == 0 and nv > 0:
                        chg.append(f"- {mid} {lbl}: Free -> ${nv:.4f} ⬆")
                    elif nv > ov + 0.0001:
                        chg.append(
                            f"- {mid} {lbl}: ${ov:.4f} -> ${nv:.4f} (+${nv-ov:.4f})"
                        )
                    elif nv < ov - 0.0001:
                        chg.append(
                            f"- {mid} {lbl}: ${ov:.4f} -> ${nv:.4f} ({nv-ov:+.4f})"
                        )
        L.append("\n### Changes\n" + ("\n".join(chg) if chg else "None."))
        L.append(FOOTER)
        return "\n".join(L)

    def _cheapest(self):
        # FIXED: Use _fetch_all() instead of raw _load(FULL_CACHE)
        # so it auto-refreshes the cache — works regardless of command order [1]
        models = _fetch_all()
        if not models:
            return "## Cheapest Models\nCould not fetch model list from OpenRouter.\n" + FOOTER
        priced = []
        for m in models:
            p = m.get("pricing", {})
            inp = _per_1m(p.get("prompt"))
            out = _per_1m(p.get("completion"))
            if inp is not None and out is not None:
                priced.append((round(inp * 0.75 + out * 0.25, 4), m["id"], inp, out))
        priced.sort(key=lambda x: x[0])
        L = ["## Top 10 Cheapest\n"]
        L.append(
            _fmt_table(
                [
                    [i, mid, f"${inp:.4f}", f"${out:.4f}", f"${b:.4f}"]
                    for i, (b, mid, inp, out) in enumerate(priced[:10], 1)
                ],
                ["#", "Model", "Input /1M", "Output /1M", "Blended"],
            )
        )
        L.append(FOOTER)
        return "\n".join(L)

    def _comparable(self, hint):
        all_m = _fetch_all()
        tol = self.valves.performance_tolerance
        if not all_m:
            return "Error: Could not fetch model list."
        lookup = {m["id"]: m for m in all_m}
        resolved = _resolve_model(hint, all_m) if hint else None
        if not resolved:
            L = ["## Comparable Models\n"]
            L.append(
                "Provide a model name, e.g. /price comparable gpt-4o\n"
            )
            L.append(FOOTER)
            return "\n".join(L)

        r = lookup[resolved]
        rp = r.get("pricing", {})
        rpi = _per_1m(rp.get("prompt"))
        rco = _per_1m(rp.get("completion"))
        rb = round(rpi * 0.75 + rco * 0.25, 4) if rpi is not None and rco is not None else None
        ba = (r.get("benchmarks") or {}).get("artificial_analysis") or {}
        ri = ba.get("intelligence_index")
        rc = ba.get("coding_index")

        L = [f"## Comparable Models\n"]
        L.append(f"**Reference:** `{resolved}`")
        if rpi is not None and rco is not None:
            L[-1] += f" | Price: In ${rpi:.4f} / Out ${rco:.4f} / Blended ${rb:.4f}\n"

        cand = []
        for m in all_m:
            if m["id"] == resolved:
                continue
            a = (m.get("benchmarks") or {}).get("artificial_analysis") or {}
            i = a.get("intelligence_index")
            if i is None or abs(i - ri) > tol:
                continue
            if rc is not None:
                c = a.get("coding_index")
                if c is not None and abs(c - rc) > tol:
                    continue
            p = m.get("pricing", {})
            pi = _per_1m(p.get("prompt"))
            co = _per_1m(p.get("completion"))
            if pi is None or co is None:
                continue
            cand.append(
                (pi * 0.75 + co * 0.25, m["id"], pi, co, i, a.get("coding_index", "—"))
            )

        cand.sort(key=lambda x: x[0])
        if not cand:
            L.append(f"No comparable models within +/-{tol} points.")
            L.append(FOOTER)
            return "\n".join(L)

        cheap = [c for c in cand if c[0] <= rb] if rb is not None else []
        pricey = [c for c in cand if c[0] > rb] if rb is not None else cand

        if cheap:
            L.append(f"\n### Cheaper or Equal (<= ${rb:.4f}) - {len(cheap)} found")
            L.append(
                _fmt_table(
                    [
                        [mid, f"${pi:.4f}", f"${co:.4f}", f"${b:.4f}", i, c]
                        for b, mid, pi, co, i, c in cheap[:15]
                    ],
                    ["Model", "In/1M", "Out/1M", "Blended", "Intel", "Coding"],
                )
            )
        if pricey:
            L.append(f"\n### More Expensive (> ${rb:.4f}) - {len(pricey)} found")
            L.append(
                _fmt_table(
                    [
                        [mid, f"${pi:.4f}", f"${co:.4f}", f"${b:.4f}", i, c]
                        for b, mid, pi, co, i, c in pricey[:10]
                    ],
                    ["Model", "In/1M", "Out/1M", "Blended", "Intel", "Coding"],
                )
            )
        if hint and hint != resolved:
            L.append(f"\nNote: '{hint}' matched to {resolved}")

        L.append(FOOTER)
        return "\n".join(L)

    def _check_balance(self, api_key):
        if not api_key:
            api_key = self._resolve_key(self.valves.openrouter_api_key)
        if not api_key:
            return (
                "## OpenRouter Balance\n\nNo API key configured.\n"
                "Set it in: Workspace > Tools > Edit > openrouter_api_key\n"
                "Or: User Settings > User Valves > openrouter_api_key\n"
                "Or: Set a Docker environment variable and use $VAR_NAME\n"
                "Get a key: https://openrouter.ai/keys\n" + FOOTER
            )
        try:
            r = requests.get(
                OPENROUTER_KEY,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            if r.status_code == 401:
                return "## OpenRouter Balance\n\nInvalid API key.\n" + FOOTER
            r.raise_for_status()
            kd = r.json().get("data", {})
            bal = None
            try:
                r2 = requests.get(
                    "https://openrouter.ai/api/v1/credits",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
                if r2.status_code == 200:
                    cd = r2.json().get("data", {})
                    t = cd.get("total_credits")
                    u = cd.get("total_usage")
                    if t is not None and u is not None:
                        bal = round(t - u, 2)
            except:
                pass
        except Exception as e:
            return f"## OpenRouter Balance\n\nError: {e}\n" + FOOTER

        rows = [["Key Label", kd.get("label", "—")]]
        if bal is not None:
            rows.append(["Account Balance", f"${bal:.2f}"])
        else:
            lr = kd.get("limit_remaining")
            rows.append(
                ["Key Limit", f"${lr:.2f}" if lr is not None else "No limit set"]
            )
        rows.append(["Total Spent", f"${kd.get('usage',0):.2f}"])
        rows.append(["Spent Today", f"${kd.get('usage_daily',0):.2f}"])
        rows.append(["Spent This Month", f"${kd.get('usage_monthly',0):.2f}"])
        rows.append(["Free Tier", "Yes" if kd.get("is_free_tier") else "No"])
        L = ["## OpenRouter Balance\n", _fmt_table(rows, ["Field", "Value"])]
        if bal is not None and bal < 1.0:
            L.append("\nLow balance!")
        L.append(FOOTER)
        return "\n".join(L)
