import json
import re
from typing import Any

from headroom.compress import compress

MIN_CHARS = 2000

_ERROR_RE = re.compile(
    r"\b(error|exception|traceback|fatal|failed|failure|denied|enoent|eacces|timeout|panic|warn|warning)\b"
    r"|\bHTTP [45]\d\d\b|\bexit code [1-9]\d*\b",
    re.IGNORECASE,
)


def tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return json.dumps(content, ensure_ascii=False)


def is_compressible(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return False
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        return False
    if _ERROR_RE.search(text):
        return False
    return True


def compress_tool_results(messages: list[dict[str, Any]], model: str = "claude-sonnet-5", **kwargs: Any):
    protected: dict[str, Any] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                if not is_compressible(tool_result_text(block.get("content"))):
                    protected[block.get("tool_use_id")] = block.get("content")

    result = compress(messages, model=model, optimize=True, **kwargs)

    for message in result.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if tool_use_id in protected:
                    block["content"] = protected[tool_use_id]
    return result
