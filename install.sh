#!/bin/bash
set -e

REPO="michaelknyazev/zed-acp-ollama"
BINARY="zed-acp-ollama"
INSTALL_DIR="/usr/local/bin"
ZED_SETTINGS="$HOME/.config/zed/settings.json"

# ── 1. Detect platform ────────────────────────────────────────────────────────
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)          ARCH="amd64" ;;
  arm64|aarch64)   ARCH="arm64" ;;
  *)               echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

PLATFORM="${OS}-${ARCH}"
BINARY_URL="https://github.com/${REPO}/releases/latest/download/${BINARY}-${PLATFORM}"

# ── 2. Download & install binary ──────────────────────────────────────────────
echo "==> Downloading zed-acp-ollama (${PLATFORM})..."
curl -fsSL "$BINARY_URL" -o "/tmp/$BINARY"
chmod +x "/tmp/$BINARY"

echo "==> Installing to $INSTALL_DIR/$BINARY (requires sudo)"
sudo mv "/tmp/$BINARY" "$INSTALL_DIR/$BINARY"
echo "    OK: $(which zed-acp-ollama) ($($BINARY --version 2>/dev/null || echo dev))"

# ── 3. Get Ollama URL ─────────────────────────────────────────────────────────
if [ -z "${OLLAMA_URL:-}" ]; then
  if [ -t 0 ]; then
    read -rp "Ollama URL [http://localhost:11434]: " OLLAMA_URL
    OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
  else
    OLLAMA_URL="http://localhost:11434"
  fi
fi

if [ -z "${OLLAMA_MODEL:-}" ]; then
  if [ -t 0 ]; then
    read -rp "Default model [qwen3:latest]: " OLLAMA_MODEL
    OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:latest}"
  else
    OLLAMA_MODEL="qwen3:latest"
  fi
fi

# ── 4. Patch Zed settings.json ────────────────────────────────────────────────
echo ""
echo "==> Patching Zed settings ($ZED_SETTINGS)..."

mkdir -p "$(dirname "$ZED_SETTINGS")"
[ -f "$ZED_SETTINGS" ] || echo '{}' > "$ZED_SETTINGS"

python3 - "$ZED_SETTINGS" "$OLLAMA_URL" "$OLLAMA_MODEL" <<'PYEOF'
import json, re, sys

def strip_jsonc(text):
    """Strip // and /* */ comments plus trailing commas (Zed uses JSONC)."""
    result = []
    i = 0
    while i < len(text):
        if result and text[i] == '"':
            result.append(text[i]); i += 1
            while i < len(text):
                ch = text[i]; result.append(ch); i += 1
                if ch == '\\':
                    if i < len(text): result.append(text[i]); i += 1
                elif ch == '"':
                    break
            continue
        if text[i:i+2] == '//':
            while i < len(text) and text[i] != '\n': i += 1
            continue
        if text[i:i+2] == '/*':
            end = text.find('*/', i + 2)
            i = end + 2 if end != -1 else len(text)
            continue
        result.append(text[i]); i += 1
    cleaned = ''.join(result)
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    return cleaned

path, ollama_url, ollama_model = sys.argv[1], sys.argv[2], sys.argv[3]

with open(path) as f:
    raw = f.read()

cfg = json.loads(strip_jsonc(raw))

cfg.setdefault("agent_servers", {})["ollama-local"] = {
    "type": "custom",
    "command": "zed-acp-ollama",
    "args": [],
    "env": {
        "OLLAMA_URL": ollama_url,
        "OLLAMA_MODEL": ollama_model,
    }
}

cfg.setdefault("language_models", {}).setdefault("ollama", {})["low_speed_timeout_in_seconds"] = 600

with open(path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"    OK: agent_servers.ollama-local → {ollama_url} / {ollama_model}")
print(f"    OK: language_models.ollama.low_speed_timeout_in_seconds = 600")
PYEOF

# ── 5. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "Done. Restart Zed, then open the Agent panel and select 'ollama-local'."
