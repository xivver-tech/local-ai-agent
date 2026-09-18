# Local AI Agent

A complete local AI agent with real tools, powered by Ollama.

## What it can do
- Read and write files
- List directories
- Run shell commands (with basic safety blocks)
- Calculate expressions
- Search across your codebase

It uses a ReAct-style loop: **Thought → Action → Observation → Final Answer**

## Requirements
```bash
# Install Ollama, then:
ollama pull llama3.2
# stronger models work better for agents: qwen2.5, llama3.1, mistral, etc.
```

## Usage
```bash
python agent.py "List all Python files and say what each one does"
python agent.py "Create a file notes.txt with a short todo list"
python agent.py "Search for any TODO comments"
python agent.py "Calculate compound interest on 1000 at 5% for 10 years" mistral
```

The agent stays inside the current working directory for safety.
