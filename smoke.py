#!/usr/bin/env python3
"""
Minimal ACP smoke-test driver.
Spawns the binary, replays the handshake, sends a prompt, pretty-prints output.
"""
import json, os, subprocess, sys, textwrap

BINARY   = os.environ.get("BINARY", "./zed-acp-ollama")
CWD      = os.environ.get("SMOKE_CWD", os.getcwd())
MSG      = os.environ.get("SMOKE_MSG", "list the files in this project")
ENV      = {**os.environ}

proc = subprocess.Popen(
    [BINARY],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=sys.stderr,
    env=ENV,
)

def send(obj):
    line = json.dumps(obj) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()

def recv():
    line = proc.stdout.readline()
    if not line:
        return None
    return json.loads(line)

# ── handshake ──────────────────────────────────────────────────────────────────
send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
r = recv()
info = r.get("result", {}).get("agentInfo", {})
print(f"[agent] {info.get('title')} v{info.get('version')}", flush=True)

send({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"cwd": CWD}})
r = recv()
session_id = r["result"]["sessionId"]
models = [o["currentValue"] for o in r["result"].get("configOptions", []) if o["id"] == "model"]
print(f"[session] {session_id}", flush=True)
print(f"[model] {models[0] if models else '?'}", flush=True)

# ── prompt ─────────────────────────────────────────────────────────────────────
print(f"\n[user] {MSG}\n", flush=True)
send({
    "jsonrpc": "2.0",
    "id": 3,
    "method": "session/prompt",
    "params": {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": MSG}],
    },
})

# ── stream output ──────────────────────────────────────────────────────────────
while True:
    r = recv()
    if r is None:
        break

    if r.get("method") == "session/update":
        u = r["params"]["update"]
        t = u.get("sessionUpdate", "")

        if t == "agent_thought_chunk":
            snippet = u["content"]["text"][:120].replace("\n", " ")
            print(f"\033[2m[thinking] {snippet}\033[0m", flush=True)

        elif t == "agent_message_chunk":
            print(u["content"]["text"], end="", flush=True)

        elif t == "tool_call":
            print(f"\n\033[33m[tool] {u['title']} ({u['toolCallId']})\033[0m", flush=True)

        elif t == "tool_call_update":
            content = u.get("content") or []
            if content:
                snippet = content[0]["content"]["text"][:300]
                print(f"\033[2m[result] {snippet}\033[0m", flush=True)

    elif "result" in r and r.get("id") == 3:
        # prompt finished
        print(f"\n\n\033[32m[done] stop_reason={r['result'].get('stopReason')}\033[0m", flush=True)
        break

    elif "error" in r:
        print(f"\n\033[31m[error] {r['error']}\033[0m", flush=True)
        break

proc.stdin.close()
proc.wait()
