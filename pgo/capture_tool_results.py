"""Claude Code PostToolUse hook: append the full tool result to a JSONL file.

Enabled only when PGO_TOOL_CAPTURE_PATH is set. Never blocks the agent: any failure exits 0.
"""

import json
import os
import sys


def main() -> int:
    target = os.environ.get("PGO_TOOL_CAPTURE_PATH")
    if not target:
        return 0
    try:
        payload = json.load(sys.stdin)
        record = {
            "session_id": payload.get("session_id"),
            "tool_name": payload.get("tool_name"),
            "tool_use_id": payload.get("tool_use_id"),
            "tool_response": payload.get("tool_response"),
        }
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
