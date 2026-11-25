import json
import os
import subprocess
import sys
from pathlib import Path

from headroom.pgo_tool_filter import compress_tool_results, is_compressible, tool_result_text

HOOK = Path(__file__).resolve().parents[1] / "pgo" / "capture_tool_results.py"


def _messages(text: str):
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}]},
    ]


def _result_text(messages):
    return tool_result_text(messages[-1]["content"][0]["content"])


def _big_log(error_line: str = "") -> str:
    lines = [f"2026-11-24T14:{i % 60:02d}:00Z INFO worker-{i % 7} processed item {i} ok" for i in range(1500)]
    if error_line:
        lines.append(error_line)
    return "\n".join(lines)


def test_large_json_is_not_compressed():
    text = json.dumps([{"id": i, "name": f"agent-{i}", "status": "idle"} for i in range(400)], indent=2)
    result = compress_tool_results(_messages(text))
    assert _result_text(result.messages) == text


def test_large_output_with_error_is_not_compressed():
    text = _big_log("2026-11-24T15:00:00Z ERROR payment-gateway: HTTP 500 upstream timeout")
    result = compress_tool_results(_messages(text))
    assert _result_text(result.messages) == text


def test_large_plain_log_is_compressed():
    text = _big_log()
    result = compress_tool_results(_messages(text))
    assert len(_result_text(result.messages)) < len(text)


def test_small_output_is_untouched():
    text = "ok\n" * 10
    assert not is_compressible(text)
    result = compress_tool_results(_messages(text))
    assert _result_text(result.messages) == text


def test_capture_hook_appends_full_response(tmp_path):
    target = tmp_path / "capture.jsonl"
    payload = {"session_id": "s1", "tool_name": "Bash", "tool_use_id": "t1", "tool_response": {"stdout": "x" * 5000}}
    env = {**os.environ, "PGO_TOOL_CAPTURE_PATH": str(target)}
    proc = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload), text=True, env=env)
    assert proc.returncode == 0
    record = json.loads(target.read_text().splitlines()[0])
    assert record["tool_response"]["stdout"] == "x" * 5000


def test_capture_hook_is_noop_without_path_and_never_fails(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PGO_TOOL_CAPTURE_PATH"}
    proc = subprocess.run([sys.executable, str(HOOK)], input="not json", text=True, env=env)
    assert proc.returncode == 0
    env["PGO_TOOL_CAPTURE_PATH"] = str(tmp_path / "capture.jsonl")
    proc = subprocess.run([sys.executable, str(HOOK)], input="not json", text=True, env=env)
    assert proc.returncode == 0
    assert not (tmp_path / "capture.jsonl").exists()
