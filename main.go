package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"io/fs"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// version is set at build time via -ldflags="-X main.version=vX.Y.Z"
var version = "dev"

var (
	ollamaURL = env("OLLAMA_URL", "http://localhost:11434")
	model     = env("OLLAMA_MODEL", "qwen3:latest")
)

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ── JSON-RPC 2.0 ─────────────────────────────────────────────────────────────

type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type Response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

type Notification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params"`
}

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// ── ACP types ─────────────────────────────────────────────────────────────────

type InitializeResult struct {
	ProtocolVersion   int       `json:"protocolVersion"`
	AgentCapabilities struct{}  `json:"agentCapabilities"`
	AgentInfo         AgentInfo `json:"agentInfo"`
}

type AgentInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title"`
	Version string `json:"version"`
}

type NewSessionParams struct {
	CWD string `json:"cwd"`
}

type NewSessionResult struct {
	SessionID     string                `json:"sessionId"`
	ConfigOptions []SessionConfigOption `json:"configOptions,omitempty"`
}

type LoadSessionParams struct {
	SessionID string `json:"sessionId"`
}

type LoadSessionResult struct {
	ConfigOptions []SessionConfigOption `json:"configOptions,omitempty"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type PromptParams struct {
	SessionID string         `json:"sessionId"`
	Prompt    []ContentBlock `json:"prompt"`
}

type PromptResult struct {
	StopReason string `json:"stopReason"`
}

type SessionUpdateParams struct {
	SessionID string `json:"sessionId"`
	Update    any    `json:"update"`
}

type AgentMessageChunk struct {
	SessionUpdate string       `json:"sessionUpdate"`
	Content       ContentBlock `json:"content"`
}

type ConfigOptionUpdate struct {
	SessionUpdate string                `json:"sessionUpdate"`
	ConfigOptions []SessionConfigOption `json:"configOptions"`
}

// SessionConfigOption is flat — kind fields (type/currentValue/options) are
// inlined at the top level due to #[serde(flatten)] in the Rust source.
type SessionConfigOption struct {
	ID           string                     `json:"id"`
	Name         string                     `json:"name"`
	Category     string                     `json:"category,omitempty"`
	Type         string                     `json:"type"`
	CurrentValue string                     `json:"currentValue"`
	Options      []SessionConfigSelectOption `json:"options"`
}

type SessionConfigSelectOption struct {
	Value string `json:"value"`
	Name  string `json:"name"`
}

type SetConfigOptionParams struct {
	SessionID string `json:"sessionId"`
	ConfigID  string `json:"configId"`
	Value     string `json:"value"`
}

type SetConfigOptionResponse struct {
	ConfigOptions []SessionConfigOption `json:"configOptions"`
}

type ToolCall struct {
	SessionUpdate string `json:"sessionUpdate"`
	ToolCallID    string `json:"toolCallId"`
	Title         string `json:"title"`
	Kind          string `json:"kind"`
	Status        string `json:"status"`
}

type ToolCallContent struct {
	Type    string       `json:"type"`
	Content ContentBlock `json:"content"`
}

type ToolCallUpdate struct {
	SessionUpdate string            `json:"sessionUpdate"`
	ToolCallID    string            `json:"toolCallId"`
	Status        string            `json:"status,omitempty"`
	Content       []ToolCallContent `json:"content,omitempty"`
}

// ── Ollama types ──────────────────────────────────────────────────────────────

type OllamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
}

type OllamaToolCall struct {
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type OllamaToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type OllamaTool struct {
	Type     string             `json:"type"`
	Function OllamaToolFunction `json:"function"`
}

type OllamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Tools    []OllamaTool    `json:"tools,omitempty"`
}

type OllamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type OllamaChunk struct {
	Message struct {
		Content   string           `json:"content"`
		Thinking  string           `json:"thinking,omitempty"`
		ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	Done         bool   `json:"done"`
	EvalCount    int    `json:"eval_count,omitempty"`
	EvalDuration int64  `json:"eval_duration,omitempty"`
}

// ── Tool definitions ──────────────────────────────────────────────────────────

var toolDefinitions = []OllamaTool{
	{
		Type: "function",
		Function: OllamaToolFunction{
			Name:        "read_file",
			Description: "Read the full contents of a file from the filesystem.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "File path — absolute, or relative to the project root.",
					},
				},
				"required": []string{"path"},
			},
		},
	},
	{
		Type: "function",
		Function: OllamaToolFunction{
			Name:        "write_file",
			Description: "Write content to a file, creating parent directories as needed. Overwrites if the file already exists.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "File path to write.",
					},
					"content": map[string]any{
						"type":        "string",
						"description": "Full file content to write.",
					},
				},
				"required": []string{"path", "content"},
			},
		},
	},
	{
		Type: "function",
		Function: OllamaToolFunction{
			Name:        "list_directory",
			Description: "List files and subdirectories in a directory.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Directory path — absolute, or relative to the project root.",
					},
				},
				"required": []string{"path"},
			},
		},
	},
	{
		Type: "function",
		Function: OllamaToolFunction{
			Name:        "run_command",
			Description: "Run a shell command in the project root and return combined stdout+stderr. Timeout: 30 s.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command to execute.",
					},
				},
				"required": []string{"command"},
			},
		},
	},
	{
		Type: "function",
		Function: OllamaToolFunction{
			Name:        "web_search",
			Description: "Search the web for current information and return a summary of top results.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Search query.",
					},
				},
				"required": []string{"query"},
			},
		},
	},
	{
		Type: "function",
		Function: OllamaToolFunction{
			Name:        "fetch_url",
			Description: "Fetch the text content of a URL (web page, raw file, API response, etc.).",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]any{
						"type":        "string",
						"description": "The URL to fetch.",
					},
				},
				"required": []string{"url"},
			},
		},
	},
}

// ── Session ───────────────────────────────────────────────────────────────────

type Session struct {
	ID              string
	CWD             string
	Model           string
	ThinkingEnabled bool
	History         []OllamaMessage
	contextLoaded   bool
}

// ── Server ────────────────────────────────────────────────────────────────────

type Server struct {
	mu           sync.Mutex
	sessions     map[string]*Session
	outMu        sync.Mutex
	enc          *json.Encoder
	client       *http.Client // no timeout — used for streaming Ollama responses
	webClient    *http.Client // short timeout — used for web search / fetch
	ollamaURL    string
	defaultModel string

	capsMu    sync.RWMutex
	modelCaps map[string][]string // model name → capabilities from /api/show
}

func newServer() *Server {
	return &Server{
		sessions:     make(map[string]*Session),
		enc:          json.NewEncoder(os.Stdout),
		client:       &http.Client{},
		webClient:    &http.Client{Timeout: 15 * time.Second},
		ollamaURL:    ollamaURL,
		defaultModel: model,
		modelCaps:    make(map[string][]string),
	}
}

func (s *Server) write(v any) {
	s.outMu.Lock()
	defer s.outMu.Unlock()
	s.enc.Encode(v)
}

func (s *Server) respond(id json.RawMessage, result any) {
	s.write(Response{JSONRPC: "2.0", ID: id, Result: result})
}

func (s *Server) respondErr(id json.RawMessage, code int, msg string) {
	s.write(Response{JSONRPC: "2.0", ID: id, Error: &RPCError{Code: code, Message: msg}})
}

func (s *Server) notify(sessionID string, update any) {
	s.write(Notification{
		JSONRPC: "2.0",
		Method:  "session/update",
		Params:  SessionUpdateParams{SessionID: sessionID, Update: update},
	})
}

func (s *Server) chunk(sessionID, text string) {
	s.notify(sessionID, AgentMessageChunk{
		SessionUpdate: "agent_message_chunk",
		Content:       ContentBlock{Type: "text", Text: text},
	})
}

func (s *Server) thought(sessionID, text string) {
	s.notify(sessionID, AgentMessageChunk{
		SessionUpdate: "agent_thought_chunk",
		Content:       ContentBlock{Type: "text", Text: text},
	})
}

func (s *Server) fetchModels() []string {
	resp, err := s.client.Get(s.ollamaURL + "/api/tags")
	if err != nil {
		log.Printf("[models] fetch error: %v", err)
		return nil
	}
	defer resp.Body.Close()
	var tags OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		log.Printf("[models] decode error: %v", err)
		return nil
	}
	names := make([]string, len(tags.Models))
	for i, m := range tags.Models {
		names[i] = m.Name
	}

	// Fetch capabilities for all models concurrently.
	var wg sync.WaitGroup
	for _, name := range names {
		name := name
		wg.Add(1)
		go func() {
			defer wg.Done()
			caps := s.fetchModelCapabilities(name)
			s.capsMu.Lock()
			s.modelCaps[name] = caps
			s.capsMu.Unlock()
			log.Printf("[caps] %s: %v", name, caps)
		}()
	}
	wg.Wait()

	return names
}

func (s *Server) fetchModelCapabilities(name string) []string {
	body, _ := json.Marshal(map[string]string{"model": name})
	resp, err := s.client.Post(s.ollamaURL+"/api/show", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("[caps] show error for %s: %v", name, err)
		return nil
	}
	defer resp.Body.Close()
	var result struct {
		Capabilities []string `json:"capabilities"`
	}
	json.NewDecoder(resp.Body).Decode(&result)
	return result.Capabilities
}

func (s *Server) hasCapability(modelName, cap string) bool {
	s.capsMu.RLock()
	defer s.capsMu.RUnlock()
	for _, c := range s.modelCaps[modelName] {
		if c == cap {
			return true
		}
	}
	return false
}

func resolveModel(current string, models []string) string {
	for _, m := range models {
		if m == current {
			return current
		}
	}
	if len(models) > 0 {
		return models[0]
	}
	return current
}

func (s *Server) buildConfigOptions(sess *Session, models []string) []SessionConfigOption {
	sess.Model = resolveModel(sess.Model, models)

	modelOptions := make([]SessionConfigSelectOption, len(models))
	for i, m := range models {
		displayName := m
		s.capsMu.RLock()
		caps := s.modelCaps[m]
		s.capsMu.RUnlock()
		var tags []string
		for _, c := range caps {
			switch c {
			case "tools":
				tags = append(tags, "tools")
			case "thinking":
				tags = append(tags, "thinking")
			case "vision":
				tags = append(tags, "vision")
			}
		}
		if len(tags) > 0 {
			displayName = m + " [" + strings.Join(tags, ", ") + "]"
		}
		modelOptions[i] = SessionConfigSelectOption{Value: m, Name: displayName}
	}

	return []SessionConfigOption{
		{
			ID:           "model",
			Name:         "Model",
			Category:     "model",
			Type:         "select",
			CurrentValue: sess.Model,
			Options:      modelOptions,
		},
		{
			ID:       "thinking",
			Name:     "Thinking",
			Category: "thought_level",
			Type:     "select",
			CurrentValue: func() string {
				if sess.ThinkingEnabled {
					return "enabled"
				}
				return "disabled"
			}(),
			Options: []SessionConfigSelectOption{
				{Value: "enabled", Name: "Enabled"},
				{Value: "disabled", Name: "Disabled"},
			},
		},
	}
}

func (s *Server) toolStart(sessionID, id, title, kind string) {
	s.notify(sessionID, ToolCall{
		SessionUpdate: "tool_call",
		ToolCallID:    id,
		Title:         title,
		Kind:          kind,
		Status:        "pending",
	})
}

func (s *Server) toolDone(sessionID, id, output string) {
	var content []ToolCallContent
	if output != "" {
		content = []ToolCallContent{{
			Type:    "content",
			Content: ContentBlock{Type: "text", Text: output},
		}}
	}
	s.notify(sessionID, ToolCallUpdate{
		SessionUpdate: "tool_call_update",
		ToolCallID:    id,
		Status:        "completed",
		Content:       content,
	})
}

// ── Tool execution ────────────────────────────────────────────────────────────

func strArg(args map[string]any, key string) string {
	if v, ok := args[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func (s *Server) resolvePath(sess *Session, p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(sess.CWD, p)
}

func (s *Server) executeTool(sess *Session, name string, rawArgs json.RawMessage) (string, error) {
	var args map[string]any
	if err := json.Unmarshal(rawArgs, &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	switch name {
	case "read_file":
		path := s.resolvePath(sess, strArg(args, "path"))
		data, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return string(data), nil

	case "write_file":
		path := s.resolvePath(sess, strArg(args, "path"))
		content := strArg(args, "content")
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return "", err
		}
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return "", err
		}
		return fmt.Sprintf("wrote %d bytes to %s", len(content), path), nil

	case "list_directory":
		path := s.resolvePath(sess, strArg(args, "path"))
		entries, err := os.ReadDir(path)
		if err != nil {
			return "", err
		}
		var sb strings.Builder
		for _, e := range entries {
			if e.IsDir() {
				sb.WriteString(e.Name() + "/\n")
			} else {
				info, _ := e.Info()
				sb.WriteString(fmt.Sprintf("%s  (%d bytes)\n", e.Name(), info.Size()))
			}
		}
		return sb.String(), nil

	case "run_command":
		cmdStr := strArg(args, "command")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		c := exec.CommandContext(ctx, "/bin/sh", "-c", cmdStr)
		c.Dir = sess.CWD
		out, err := c.CombinedOutput()
		result := string(out)
		if err != nil {
			if ctx.Err() == context.DeadlineExceeded {
				result += "\n[error: timed out after 30s]"
			} else {
				result += "\n[exit: " + err.Error() + "]"
			}
		}
		if len(result) > 16000 {
			result = result[:16000] + "\n… (truncated)"
		}
		return result, nil

	case "web_search":
		return s.webSearch(strArg(args, "query"))

	case "fetch_url":
		return s.fetchURL(strArg(args, "url"))

	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

func (s *Server) webSearch(query string) (string, error) {
	apiURL := "https://api.duckduckgo.com/?q=" + url.QueryEscape(query) + "&format=json&no_html=1&skip_disambig=1"
	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "zed-acp-ollama/"+version)

	resp, err := s.webClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("search failed: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Abstract       string `json:"Abstract"`
		AbstractURL    string `json:"AbstractURL"`
		Answer         string `json:"Answer"`
		Definition     string `json:"Definition"`
		RelatedTopics  []struct {
			Text     string `json:"Text"`
			FirstURL string `json:"FirstURL"`
		} `json:"RelatedTopics"`
		Results []struct {
			Text     string `json:"Text"`
			FirstURL string `json:"FirstURL"`
		} `json:"Results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("search decode failed: %w", err)
	}

	var sb strings.Builder
	if result.Answer != "" {
		sb.WriteString("Answer: " + result.Answer + "\n\n")
	}
	if result.Abstract != "" {
		sb.WriteString(result.Abstract + "\n")
		if result.AbstractURL != "" {
			sb.WriteString("Source: " + result.AbstractURL + "\n")
		}
		sb.WriteString("\n")
	}
	if result.Definition != "" {
		sb.WriteString("Definition: " + result.Definition + "\n\n")
	}
	for i, r := range result.RelatedTopics {
		if i >= 8 {
			break
		}
		if r.Text != "" {
			sb.WriteString("• " + r.Text + "\n  " + r.FirstURL + "\n")
		}
	}
	for i, r := range result.Results {
		if i >= 5 {
			break
		}
		if r.Text != "" {
			sb.WriteString("• " + r.Text + "\n  " + r.FirstURL + "\n")
		}
	}
	if sb.Len() == 0 {
		return "No results found for: " + query, nil
	}
	return strings.TrimSpace(sb.String()), nil
}

var (
	reTag    = regexp.MustCompile(`<[^>]+>`)
	reSpaces = regexp.MustCompile(`[ \t]+`)
	reLines  = regexp.MustCompile(`\n{3,}`)
)

func stripHTML(s string) string {
	s = reTag.ReplaceAllString(s, " ")
	s = html.UnescapeString(s)
	s = reSpaces.ReplaceAllString(s, " ")
	s = reLines.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

func (s *Server) fetchURL(rawURL string) (string, error) {
	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; zed-acp-ollama/"+version+")")
	req.Header.Set("Accept", "text/html,text/plain,*/*")

	resp, err := s.webClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if err != nil {
		return "", fmt.Errorf("read failed: %w", err)
	}

	text := string(body)
	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "html") || strings.HasPrefix(strings.TrimSpace(text), "<") {
		text = stripHTML(text)
	}
	if len(text) > 12000 {
		text = text[:12000] + "\n… (truncated)"
	}
	return text, nil
}

// ── Context loading ───────────────────────────────────────────────────────────

var contextFiles = []string{
	"AGENTS.md", "CLAUDE.md", ".cursorrules", ".rules", "README.md",
}

var treeIgnore = map[string]bool{
	".git": true, "node_modules": true, "vendor": true,
	".build": true, "dist": true, "build": true,
	"__pycache__": true, ".next": true, "target": true, ".cache": true,
}

func (s *Server) loadContext(sessionID, cwd string) string {
	var sb strings.Builder
	sb.WriteString("You are a powerful coding and research assistant with full tool access. " +
		"Use tools proactively: read files before editing them, run commands to verify your changes, " +
		"and search the web when you need current information. " +
		"Always make real edits — do not just suggest changes. " +
		"The user is working in a project at: " + cwd + "\n\n")

	for _, name := range contextFiles {
		path := filepath.Join(cwd, name)
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		tcID := "read_" + strings.NewReplacer(".", "_", "/", "_").Replace(name)
		s.toolStart(sessionID, tcID, "Read "+name, "read")
		content := strings.TrimSpace(string(data))
		preview := content
		if len(preview) > 2000 {
			preview = preview[:2000] + "\n… (truncated)"
		}
		s.toolDone(sessionID, tcID, preview)
		sb.WriteString(fmt.Sprintf("## %s\n\n%s\n\n", name, content))
		log.Printf("[context] loaded %s (%d bytes)", name, len(data))
	}

	s.toolStart(sessionID, "tree", "Scan project structure", "read")
	tree := fileTree(cwd, 3)
	s.toolDone(sessionID, "tree", tree)
	if tree != "" {
		sb.WriteString("## Project structure\n\n```\n" + tree + "```\n")
	}

	return sb.String()
}

func fileTree(root string, maxDepth int) string {
	var sb strings.Builder
	sb.WriteString(filepath.Base(root) + "/\n")
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || path == root {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		parts := strings.Split(rel, string(filepath.Separator))
		if treeIgnore[parts[0]] {
			return filepath.SkipDir
		}
		if len(parts) > maxDepth {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			name += "/"
		}
		sb.WriteString(strings.Repeat("  ", len(parts)) + name + "\n")
		return nil
	})
	return sb.String()
}

// ── Handlers ──────────────────────────────────────────────────────────────────

func (s *Server) handle(req Request) {
	switch req.Method {
	case "initialize":
		log.Printf("[init] handshake")
		s.respond(req.ID, InitializeResult{
			ProtocolVersion: 1,
			AgentInfo:       AgentInfo{Name: "zed-acp-ollama", Title: "Ollama ACP Bridge", Version: version},
		})

	case "session/new":
		var p NewSessionParams
		json.Unmarshal(req.Params, &p)
		sess := &Session{
			ID:              fmt.Sprintf("sess_%d", time.Now().UnixNano()),
			CWD:             p.CWD,
			Model:           model,
			ThinkingEnabled: true,
		}
		s.mu.Lock()
		s.sessions[sess.ID] = sess
		s.mu.Unlock()
		models := s.fetchModels()
		opts := s.buildConfigOptions(sess, models)
		log.Printf("[session] new %s cwd=%s models=%d", sess.ID, p.CWD, len(models))
		s.respond(req.ID, NewSessionResult{SessionID: sess.ID, ConfigOptions: opts})

	case "session/load", "session/resume":
		var p LoadSessionParams
		json.Unmarshal(req.Params, &p)
		s.mu.Lock()
		sess, exists := s.sessions[p.SessionID]
		if !exists {
			sess = &Session{ID: p.SessionID, Model: model, ThinkingEnabled: true}
			s.sessions[p.SessionID] = sess
		}
		s.mu.Unlock()
		models := s.fetchModels()
		opts := s.buildConfigOptions(sess, models)
		log.Printf("[session] load %s models=%d", p.SessionID, len(models))
		s.respond(req.ID, LoadSessionResult{ConfigOptions: opts})

	case "session/set_config_option":
		var p SetConfigOptionParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			s.respondErr(req.ID, -32600, "invalid params")
			return
		}
		s.mu.Lock()
		sess, ok := s.sessions[p.SessionID]
		if ok {
			switch p.ConfigID {
			case "model":
				sess.Model = p.Value
				log.Printf("[config] session=%s model → %s", p.SessionID, p.Value)
			case "thinking":
				sess.ThinkingEnabled = p.Value == "enabled"
				log.Printf("[config] session=%s thinking → %s", p.SessionID, p.Value)
			}
		}
		s.mu.Unlock()
		go func() {
			models := s.fetchModels()
			s.mu.Lock()
			var opts []SessionConfigOption
			if sess, ok := s.sessions[p.SessionID]; ok {
				opts = s.buildConfigOptions(sess, models)
			}
			s.mu.Unlock()
			s.respond(req.ID, SetConfigOptionResponse{ConfigOptions: opts})
		}()

	case "session/prompt":
		var p PromptParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			s.respondErr(req.ID, -32600, "invalid params")
			return
		}
		go s.handlePrompt(req.ID, p)

	default:
		if req.ID != nil {
			s.respondErr(req.ID, -32601, fmt.Sprintf("unknown method: %s", req.Method))
		}
	}
}

func (s *Server) handlePrompt(id json.RawMessage, params PromptParams) {
	start := time.Now()

	s.mu.Lock()
	sess, ok := s.sessions[params.SessionID]
	if !ok {
		s.mu.Unlock()
		s.respondErr(id, -32600, "session not found")
		return
	}

	if !sess.contextLoaded {
		sess.contextLoaded = true
		s.mu.Unlock()
		systemPrompt := s.loadContext(params.SessionID, sess.CWD)
		s.mu.Lock()
		if systemPrompt != "" {
			sess.History = append([]OllamaMessage{{Role: "system", Content: systemPrompt}}, sess.History...)
		}
	}

	var userText strings.Builder
	for _, b := range params.Prompt {
		if b.Type == "text" || b.Type == "resource" {
			userText.WriteString(b.Text)
		}
	}
	sess.History = append(sess.History, OllamaMessage{Role: "user", Content: userText.String()})
	messages := make([]OllamaMessage, len(sess.History))
	copy(messages, sess.History)
	activeModel := sess.Model
	thinkingEnabled := sess.ThinkingEnabled
	s.mu.Unlock()

	log.Printf("[prompt] session=%s model=%s input=%d chars history=%d msgs",
		params.SessionID, activeModel, userText.Len(), len(messages))

	// Agentic loop: re-prompt after each set of tool calls, up to maxLoops times.
	const maxLoops = 20
	var lastLoopText string
	supportsTools := s.hasCapability(activeModel, "tools")
	log.Printf("[prompt] model=%s supportsTools=%v", activeModel, supportsTools)

	for loop := 0; loop < maxLoops; loop++ {
		var tools []OllamaTool
		if supportsTools {
			tools = toolDefinitions
		}
		body, _ := json.Marshal(OllamaChatRequest{
			Model:    activeModel,
			Messages: messages,
			Stream:   true,
			Tools:    tools,
		})

		resp, err := s.client.Post(s.ollamaURL+"/api/chat", "application/json", bytes.NewReader(body))
		if err != nil {
			log.Printf("[prompt] ollama error: %v", err)
			s.respondErr(id, -32000, "upstream error: "+err.Error())
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

		var loopText strings.Builder
		var toolCalls []OllamaToolCall
		thinkingStarted, thinkingDone := false, false
		thinkingTokens, contentTokens := 0, 0

		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			var chunk OllamaChunk
			if err := json.Unmarshal(line, &chunk); err != nil {
				continue
			}

			// Capture tool calls from any chunk (some models emit them before done=true).
			if len(chunk.Message.ToolCalls) > 0 {
				toolCalls = chunk.Message.ToolCalls
			}

			if chunk.Done {
				if chunk.EvalCount > 0 && chunk.EvalDuration > 0 {
					tps := float64(chunk.EvalCount) / (float64(chunk.EvalDuration) / 1e9)
					log.Printf("[prompt] loop=%d done: %d thinking, %d content, %.1f tok/s, %s",
						loop, thinkingTokens, contentTokens, tps, time.Since(start))
				}
				break
			}

			if chunk.Message.Thinking != "" && chunk.Message.Content == "" {
				if !thinkingStarted {
					thinkingStarted = true
					log.Printf("[prompt] loop=%d thinking started", loop)
				}
				thinkingTokens++
				if thinkingEnabled {
					s.thought(params.SessionID, chunk.Message.Thinking)
				}
				continue
			}

			if thinkingStarted && !thinkingDone && chunk.Message.Content != "" {
				thinkingDone = true
				log.Printf("[prompt] loop=%d thinking done: %d tokens in %s", loop, thinkingTokens, time.Since(start))
			}

			if chunk.Message.Content != "" {
				contentTokens++
				loopText.WriteString(chunk.Message.Content)
				s.chunk(params.SessionID, chunk.Message.Content)
			}
		}

		if err := scanner.Err(); err != nil {
			log.Printf("[prompt] loop=%d scanner error: %v", loop, err)
		}
		resp.Body.Close()

		if len(toolCalls) == 0 {
			// No tool calls — this is the final response.
			lastLoopText = loopText.String()
			break
		}

		// Add the assistant's turn (with tool calls) to the working message list.
		messages = append(messages, OllamaMessage{
			Role:      "assistant",
			Content:   loopText.String(),
			ToolCalls: toolCalls,
		})

		// Execute each tool call and append the result.
		for i, tc := range toolCalls {
			name := tc.Function.Name
			tcID := fmt.Sprintf("tool_%d_%d_%d", time.Now().UnixNano(), loop, i)
			log.Printf("[tool] loop=%d name=%s args=%s", loop, name, string(tc.Function.Arguments))
			s.toolStart(params.SessionID, tcID, name, "tool")

			result, err := s.executeTool(sess, name, tc.Function.Arguments)
			if err != nil {
				result = fmt.Sprintf("[error: %v]", err)
				log.Printf("[tool] loop=%d %s error: %v", loop, name, err)
			}

			preview := result
			if len(preview) > 2000 {
				preview = preview[:2000] + "\n… (truncated)"
			}
			s.toolDone(params.SessionID, tcID, preview)

			messages = append(messages, OllamaMessage{
				Role:    "tool",
				Content: result,
			})
		}

		log.Printf("[prompt] loop=%d executed %d tool(s), re-prompting", loop, len(toolCalls))
	}

	// Persist the full conversation (all tool calls + results + final response).
	s.mu.Lock()
	if sess, ok := s.sessions[params.SessionID]; ok {
		sess.History = append(messages, OllamaMessage{
			Role:    "assistant",
			Content: lastLoopText,
		})
	}
	s.mu.Unlock()

	log.Printf("[prompt] session=%s complete in %s", params.SessionID, time.Since(start))
	s.respond(id, PromptResult{StopReason: "end_turn"})
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	log.SetOutput(os.Stderr)
	log.Printf("zed-acp-ollama starting: url=%s model=%s", ollamaURL, model)

	srv := newServer()
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var req Request
		if err := json.Unmarshal(line, &req); err != nil {
			log.Printf("parse error: %v", err)
			continue
		}
		srv.handle(req)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("stdin error: %v", err)
	}
}
