#!/usr/bin/env python3
"""
Local AI Agent
--------------
A complete local agent that can use tools via Ollama.

Tools available to the model:
  - read_file(path)
  - write_file(path, content)
  - list_dir(path)
  - run_command(cmd)          # careful: runs in a subprocess
  - calculate(expression)
  - search_files(query, root) # simple filename + content search

The agent uses a ReAct-style loop:
  Thought → Action → Observation → ... → Final Answer
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from datetime import datetime

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
MAX_STEPS = 12
WORKSPACE = Path.cwd()

# ---------- tools ----------

def tool_read_file(path: str) -> str:
    p = (WORKSPACE / path).resolve()
    if not str(p).startswith(str(WORKSPACE.resolve())):
        return "Error: path outside workspace"
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > 12000:
            return text[:12000] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"Error: {e}"

def tool_write_file(path: str, content: str) -> str:
    p = (WORKSPACE / path).resolve()
    if not str(p).startswith(str(WORKSPACE.resolve())):
        return "Error: path outside workspace"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"

def tool_list_dir(path: str = ".") -> str:
    p = (WORKSPACE / path).resolve()
    if not str(p).startswith(str(WORKSPACE.resolve())):
        return "Error: path outside workspace"
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    entries = []
    for child in sorted(p.iterdir()):
        kind = "dir " if child.is_dir() else "file"
        entries.append(f"{kind}  {child.name}")
    return "\n".join(entries) if entries else "(empty)"

def tool_run_command(cmd: str) -> str:
    # Safety: block obviously dangerous patterns
    blocked = ["rm -rf /", "mkfs", "dd if=", ":(){", "shutdown", "reboot"]
    lower = cmd.lower()
    for b in blocked:
        if b in lower:
            return f"Blocked dangerous command pattern: {b}"
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=str(WORKSPACE),
            capture_output=True, text=True, timeout=30
        )
        out = (result.stdout or "") + (result.stderr or "")
        if len(out) > 8000:
            out = out[:8000] + "\n...[truncated]"
        return out or f"(exit code {result.returncode})"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s)"
    except Exception as e:
        return f"Error: {e}"

def tool_calculate(expression: str) -> str:
    try:
        # very restricted eval
        allowed = set("0123456789+-*/().% eE")
        if not all(c in allowed or c.isspace() for c in expression):
            return "Error: invalid characters"
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def tool_search_files(query: str, root: str = ".") -> str:
    root_p = (WORKSPACE / root).resolve()
    if not str(root_p).startswith(str(WORKSPACE.resolve())):
        return "Error: path outside workspace"
    hits = []
    q = query.lower()
    for p in root_p.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".py", ".js", ".ts", ".md", ".txt", ".json", ".html", ".css", ".go", ".rs"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if q in p.name.lower() or q in text.lower():
            rel = p.relative_to(WORKSPACE)
            # short snippet
            idx = text.lower().find(q)
            snippet = ""
            if idx >= 0:
                start = max(0, idx - 40)
                snippet = text[start:start+80].replace("\n", " ")
            hits.append(f"{rel}: ...{snippet}...")
            if len(hits) >= 15:
                break
    return "\n".join(hits) if hits else "No matches"

TOOLS = {
    "read_file": lambda args: tool_read_file(args.get("path", "")),
    "write_file": lambda args: tool_write_file(args.get("path", ""), args.get("content", "")),
    "list_dir": lambda args: tool_list_dir(args.get("path", ".")),
    "run_command": lambda args: tool_run_command(args.get("cmd", "")),
    "calculate": lambda args: tool_calculate(args.get("expression", "")),
    "search_files": lambda args: tool_search_files(args.get("query", ""), args.get("root", ".")),
}

TOOL_DESCRIPTIONS = """
You have access to these tools. To use a tool, reply with EXACTLY this format:

ACTION: tool_name
ARGS: {"param": "value"}

Available tools:
- read_file        ARGS: {"path": "relative/path"}
- write_file       ARGS: {"path": "relative/path", "content": "..."}
- list_dir         ARGS: {"path": "."}
- run_command      ARGS: {"cmd": "shell command"}
- calculate        ARGS: {"expression": "2+2*3"}
- search_files     ARGS: {"query": "text", "root": "."}

When you have enough information to answer the user, reply with:

FINAL: your answer here

Do not invent tools. Always use the exact ACTION/ARGS format.
"""

# ---------- ollama ----------

def ollama(messages, model=DEFAULT_MODEL):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2}
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
            return data.get("message", {}).get("content", "")
    except urllib.error.URLError as e:
        print(f"Cannot reach Ollama at {OLLAMA_URL}")
        print("Run: ollama serve   and   ollama pull", model)
        raise SystemExit(1) from e

def parse_action(text):
    """Extract ACTION and ARGS from model output."""
    action_m = re.search(r"ACTION:\s*(\w+)", text)
    if not action_m:
        return None, None
    action = action_m.group(1).strip()
    args = {}
    args_m = re.search(r"ARGS:\s*(\{.*\})", text, re.DOTALL)
    if args_m:
        try:
            args = json.loads(args_m.group(1))
        except json.JSONDecodeError:
            # try to be forgiving
            pass
    return action, args

def run_agent(user_goal, model=DEFAULT_MODEL):
    system = (
        "You are a capable local AI agent operating inside a workspace. "
        "You solve tasks by using tools step by step.\n"
        + TOOL_DESCRIPTIONS
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_goal}
    ]

    print(f"\n🎯 Goal: {user_goal}\n")

    for step in range(1, MAX_STEPS + 1):
        print(f"── Step {step} ──")
        reply = ollama(messages, model)
        print(reply[:500] + ("..." if len(reply) > 500 else ""))
        print()

        # Final answer?
        final_m = re.search(r"FINAL:\s*(.*)", reply, re.DOTALL)
        if final_m and "ACTION:" not in reply.split("FINAL:")[0]:
            answer = final_m.group(1).strip()
            print("=" * 50)
            print("✅ FINAL ANSWER:")
            print(answer)
            return answer

        action, args = parse_action(reply)
        if not action:
            # model didn't follow format — ask it to use tools or finish
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "Please either use a tool with ACTION/ARGS or give FINAL: answer."
            })
            continue

        if action not in TOOLS:
            observation = f"Unknown tool: {action}. Available: {', '.join(TOOLS)}"
        else:
            print(f"⚙️  {action}({args})")
            observation = TOOLS[action](args)
            print(f"📋 Observation ({len(observation)} chars)")

        messages.append({"role": "assistant", "content": reply})
        messages.append({
            "role": "user",
            "content": f"Observation:\n{observation}\n\nContinue. Use another ACTION or give FINAL:"
        })

    print("Reached max steps without FINAL answer.")
    return None

def main():
    if len(sys.argv) < 2:
        print("""Local AI Agent
==============
A full local agent with tools, powered by Ollama.

Usage:
  python agent.py "Your goal here"
  python agent.py "Your goal" mistral

Examples:
  python agent.py "List all Python files and summarize what each does"
  python agent.py "Create a file hello.py that prints hello world"
  python agent.py "Calculate 17 * 24 + 100"
  python agent.py "Search for functions that contain 'def main'"

Tools the agent can use: read_file, write_file, list_dir,
run_command, calculate, search_files

Requires Ollama running + a model (ollama pull llama3.2)
""")
        return

    goal = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    run_agent(goal, model)

if __name__ == "__main__":
    main()
