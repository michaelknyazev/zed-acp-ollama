# zed-acp-ollama

> Run local Ollama models in Zed's Agent panel — with native thinking blocks, model picker, and automatic project context.

[![CI](https://github.com/michaelknyazev/zed-acp-ollama/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelknyazev/zed-acp-ollama/actions/workflows/ci.yml)
[![Release](https://github.com/michaelknyazev/zed-acp-ollama/actions/workflows/release.yml/badge.svg)](https://github.com/michaelknyazev/zed-acp-ollama/releases)

---

## The problem

Zed connects to Ollama via HTTP. Models like Qwen3 produce **thinking tokens** — internal reasoning that can take 60–400 seconds before the first content token appears. During this time, Zed's HTTP client sees no data and kills the connection with a broken pipe.

The standard workaround (raising `low_speed_timeout_in_seconds`) is fragile and doesn't give users visibility into what the model is doing.

## How it works

`zed-acp-ollama` is a small Go binary that implements the [Agent Client Protocol](https://agentclientprotocol.com) over **stdio**. Zed spawns it as a subprocess and communicates via JSON-RPC 2.0 — no HTTP timeout can fire on a pipe.

```
Zed  ──stdio──▶  zed-acp-ollama  ──HTTP──▶  Ollama
      JSON-RPC                /api/chat
                              /api/tags
```

Inside `zed-acp-ollama`:

- **Thinking tokens** (`message.thinking`) are forwarded as `agent_thought_chunk` ACP notifications — Zed renders them as a native collapsible "Thinking" block
- **Content tokens** stream immediately as `agent_message_chunk` notifications
- **Model list** is fetched from `/api/tags` and sent as a `config_option_update` — Zed renders a model picker dropdown
- **Project context** is loaded from `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `README.md`, and a file tree on the first turn, with each read shown as a visible tool call
- **Conversation history** is maintained per session

## Features

- Native thinking blocks (collapsible, streams in real time)
- Model picker — all models from your Ollama instance in a dropdown
- Thinking on/off toggle (via `thought_level` config dropdown)
- Project context: auto-reads `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.rules`, `README.md`
- File tree injected into system prompt (depth 3, noise filtered)
- Conversation history per session
- Per-session model — switch models mid-conversation without losing history

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/michaelknyazev/zed-acp-ollama/main/install.sh | bash
```

The script:
1. Detects your OS and architecture
2. Downloads the correct binary from the latest GitHub release
3. Installs to `/usr/local/bin/zed-acp-ollama`
4. Patches `~/.config/zed/settings.json` with the `agent_servers` entry (handles JSONC comments and trailing commas)

Then restart Zed and open the Agent panel — **ollama-local** will appear in the model picker.

## Configuration

All configuration is via environment variables. Set them in the `env` block of your Zed `settings.json`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `qwen3:latest` | Default model (overridable in Zed UI) |

Example `settings.json`:

```json
{
  "agent_servers": {
    "ollama-local": {
      "type": "custom",
      "command": "zed-acp-ollama",
      "args": [],
      "env": {
        "OLLAMA_URL": "http://192.168.1.100:11434",
        "OLLAMA_MODEL": "qwen3:latest"
      }
    }
  }
}
```

## Project context

On the first message of a new session, `zed-acp-ollama` reads context files from the project's working directory (passed by Zed as `cwd`). Each file read appears as a visible tool call in Zed's UI.

Files read (in priority order, first match per name wins):

| File | Purpose |
|---|---|
| `AGENTS.md` | Agent instructions, rules, project conventions |
| `CLAUDE.md` | Claude-style project memory (also useful for other models) |
| `.cursorrules` | Cursor-style rules |
| `.rules` | Generic rules file |
| `README.md` | Project overview |

Plus a file tree (depth 3) with common noise directories excluded (`.git`, `node_modules`, `vendor`, `dist`, `build`, `target`, etc.).

All loaded content is injected as a `system` message on the first turn. Subsequent turns use the accumulated conversation history directly.

## Building from source

```bash
git clone https://github.com/michaelknyazev/zed-acp-ollama
cd zed-acp-ollama
go build -o zed-acp-ollama .
```

Requires Go 1.22+.

## Testing

```bash
go test ./...
```

## ACP protocol

`zed-acp-ollama` implements the following [Agent Client Protocol](https://agentclientprotocol.com) methods:

| Method | Direction | Description |
|---|---|---|
| `initialize` | client → agent | Capability handshake |
| `session/new` | client → agent | Create session, triggers model picker |
| `session/load` | client → agent | Resume session (creates fresh if lost on restart) |
| `session/prompt` | client → agent | User message, streams response |
| `session/set_config_option` | client → agent | User changed model or thinking toggle |
| `session/update` | agent → client | Streaming notifications (content, thought, tool calls, config) |

## Why not just use Zed's built-in Ollama integration?

Zed's built-in Ollama support works for simple models, but has limitations with thinking models:

- **Timeout during thinking**: The HTTP client has a configurable but fixed timeout. A 400-second thinking phase will exceed any reasonable HTTP timeout.
- **No thinking UI**: Even with a proxy that rewrites `thinking` tokens as `content`, Zed renders them as plain markdown text — no collapsible block.
- **No model picker**: The built-in integration requires editing `settings.json` to change models.

The ACP transport (stdio JSON-RPC) eliminates the timeout issue entirely and gives access to Zed's full agent UI.

## License

MIT
