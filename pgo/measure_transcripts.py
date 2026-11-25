"""Before/after tool-result token measurement over Claude Code transcripts.

Prints aggregates only (no tool output content). Excludes trading and adult sessions.
"""

import argparse
import json
import re
import time
from pathlib import Path

import tiktoken

from headroom.compress import compress
from headroom.pgo_tool_filter import compress_tool_results, is_compressible, tool_result_text

ENC = tiktoken.get_encoding("cl100k_base")
EXCLUDE = ("trading", "adult", "pornograd", "acnoctem", "fatma", "codepixl")
ERROR_LINE_RE = re.compile(r"error|exception|traceback|fatal|failed|denied|enoent|timeout|warn", re.I)


def toks(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def load_session(path: Path):
    calls = {}
    results = []
    usage_in = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message") or {}
        content = msg.get("content")
        if obj.get("type") == "assistant":
            u = msg.get("usage") or {}
            usage_in += (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (
                u.get("cache_creation_input_tokens") or 0)
            for b in content or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls[b.get("id")] = b.get("name") or "tool"
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results.append({
                        "id": b.get("tool_use_id"),
                        "name": calls.get(b.get("tool_use_id"), "tool"),
                        "text": tool_result_text(b.get("content")),
                    })
    return results, usage_in


def as_messages(results):
    msgs = []
    for i, r in enumerate(results):
        tid = r["id"] or f"t{i}"
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tid, "name": r["name"], "input": {}}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": r["text"]}]})
    return msgs


def after_texts(messages):
    out = {}
    for m in messages:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out[b.get("tool_use_id")] = tool_result_text(b.get("content"))
    return out


def is_json(text):
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def select_sessions(root: Path, limit: int, min_age_s: int):
    now = time.time()
    picked = []
    files = sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        if any(x in str(f.parent).lower() for x in EXCLUDE):
            continue
        if now - f.stat().st_mtime < min_age_s:
            continue
        picked.append(f)
        if len(picked) == limit:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min-age-s", type=int, default=600)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sessions = select_sessions(Path(args.root), args.limit, args.min_age_s)
    rows = []
    agg = dict(before=0, base=0, filt=0, input=0, protected=0, protected_mismatch=0,
               json_total=0, json_broken_base=0, json_broken_filt=0,
               err_lines=0, err_lost_base=0, err_lost_filt=0, results=0, compressible=0)

    for f in sessions:
        results, usage_in = load_session(f)
        if not results:
            continue
        msgs_base = as_messages(results)
        base_res = compress(msgs_base, model="claude-sonnet-5", optimize=True)
        base_after = after_texts(base_res.messages)

        msgs_filt = as_messages(results)
        filt_res = compress_tool_results(msgs_filt, model="claude-sonnet-5")
        filt_after = after_texts(filt_res.messages)

        row = dict(results=len(results), before=0, base=0, filt=0, protected=0)
        for i, r in enumerate(results):
            tid = r["id"] or f"t{i}"
            before = toks(r["text"])
            b_txt = base_after.get(tid, r["text"])
            f_txt = filt_after.get(tid, r["text"])
            row["before"] += before
            row["base"] += toks(b_txt)
            row["filt"] += toks(f_txt)
            agg["results"] += 1
            if not is_compressible(r["text"]):
                row["protected"] += 1
                agg["protected"] += 1
                if f_txt != r["text"]:
                    agg["protected_mismatch"] += 1
            else:
                agg["compressible"] += 1
            if is_json(r["text"]):
                agg["json_total"] += 1
                if not is_json(b_txt):
                    agg["json_broken_base"] += 1
                if not is_json(f_txt):
                    agg["json_broken_filt"] += 1
            err_orig = {ln.strip() for ln in r["text"].splitlines() if ERROR_LINE_RE.search(ln)}
            if err_orig:
                agg["err_lines"] += len(err_orig)
                agg["err_lost_base"] += len(err_orig - {ln.strip() for ln in b_txt.splitlines()})
                agg["err_lost_filt"] += len(err_orig - {ln.strip() for ln in f_txt.splitlines()})
        for k in ("before", "base", "filt"):
            agg[k] += row[k]
        agg["input"] += usage_in
        row["input"] = usage_in
        rows.append(row)

    def pct(a, b):
        return round(100 * (a - b) / a, 1) if a else 0.0

    summary = dict(
        sessions=len(rows), tool_results=agg["results"],
        tool_tokens_before=agg["before"],
        tool_tokens_after_headroom_default=agg["base"], saved_pct_default=pct(agg["before"], agg["base"]),
        tool_tokens_after_pgo_filter=agg["filt"], saved_pct_pgo_filter=pct(agg["before"], agg["filt"]),
        tool_share_of_input_pct=round(100 * agg["before"] / agg["input"], 2) if agg["input"] else 0.0,
        protected_results=agg["protected"], protected_changed_by_filter=agg["protected_mismatch"],
        json_results=agg["json_total"], json_broken_headroom_default=agg["json_broken_base"],
        json_broken_pgo_filter=agg["json_broken_filt"],
        error_lines=agg["err_lines"], error_lines_lost_headroom_default=agg["err_lost_base"],
        error_lines_lost_pgo_filter=agg["err_lost_filt"],
    )
    text = json.dumps({"summary": summary, "sessions": rows}, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
