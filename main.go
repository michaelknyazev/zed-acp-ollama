package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ── defaults ──────────────────────────────────────────────────────────────────

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
	SessionID string `json:"sessionId"`
}

type LoadSessionParams struct {
	SessionID string `json:"sessionId"`
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
	SessionUpdate string                `json:"sessionUpdate"` // "config_option_update"
	ConfigOptions []SessionConfigOption `json:"configOptions"`
}

// SessionConfigOption is flat — kind fields (type/currentValue/options) are
// inlined at the top level due to #[serde(flatten)] in the Rust source.
type SessionConfigOption struct {
	ID           string                     `json:"id"`
	Name         string                     `json:"name"`
	Category     string                     `json:"category,omitempty"`
	Type         string                     `json:"type"`         // "select"
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
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

type OllamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type OllamaChunk struct {
	Message struct {
		Content  string `json:"content"`
		Thinking string `json:"thinking,omitempty"`
	} `json:"message"`
	Done         bool   `json:"done"`
	EvalCount    int    `json:"eval_count,omitempty"`
	EvalDuration int64  `json:"eval_duration,omitempty"`
}

// ── Session ───────────────────────────────────────────────────────────────────

type Session struct {
	ID             string
	CWD            string
	Model          string
	ThinkingEnabled bool
	History        []OllamaMessage
	contextLoaded  bool
}

// ── Server ────────────────────────────────────────────────────────────────────

type Server struct {
	mu           sync.Mutex
	sessions     map[string]*Session
	outMu        sync.Mutex
	enc          *json.Encoder
	client       *http.Client
	ollamaURL    string
	defaultModel string
}

func newServer() *Server {
	return &Server{
		sessions:     make(map[string]*Session),
		enc:          json.NewEncoder(os.Stdout),
		client:       &http.Client{},
		ollamaURL:    ollamaURL,
		defaultModel: model,
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
	return names
}

func (s *Server) buildConfigOptions(sess *Session, models []string) []SessionConfigOption {
	modelOptions := make([]SessionConfigSelectOption, len(models))
	for i, m := range models {
		modelOptions[i] = SessionConfigSelectOption{Value: m, Name: m}
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

func (s *Server) sendModelPicker(sessionID string) {
	models := s.fetchModels()

	s.mu.Lock()
	sess := s.sessions[sessionID]
	if sess == nil {
		s.mu.Unlock()
		return
	}
	opts := s.buildConfigOptions(sess, models)
	s.mu.Unlock()

	s.notify(sessionID, ConfigOptionUpdate{
		SessionUpdate: "config_option_update",
		ConfigOptions: opts,
	})
	log.Printf("[models] sent picker: %d models", len(models))
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

// ── Context loading ───────────────────────────────────────────────────────────

var contextFiles = []string{
	"AGENTS.md", "CLAUDE.md", ".cursorrules", ".rules", "README.md",
}

var treeIgnore = map[string]bool{
	".git": true, "node_modules": true, "vendor": true,
	".build": true, "dist": true, "build": true,
	"__pycache__": true, ".next": true, "target": true, ".cache": true,
}

// loadContext reads project files with visible tool_call notifications.
// Returns the system prompt string. Called on the first prompt turn.
func (s *Server) loadContext(sessionID, cwd string) string {
	var sb strings.Builder
	sb.WriteString("You are a coding assistant. The user is working in a project at: " + cwd + "\n\n")

	for _, name := range contextFiles {
		path := filepath.Join(cwd, name)
		data, err := os.ReadFile(path)
		if err != nil {
			continue // file doesn't exist — skip silently
		}

		tcID := "read_" + strings.NewReplacer(".", "_", "/", "_").Replace(name)
		s.toolStart(sessionID, tcID, "Read "+name, "read")

		content := strings.TrimSpace(string(data))
		// Show up to 2000 chars in the tool output
		preview := content
		if len(preview) > 2000 {
			preview = preview[:2000] + "\n… (truncated)"
		}
		s.toolDone(sessionID, tcID, preview)

		sb.WriteString(fmt.Sprintf("## %s\n\n%s\n\n", name, content))
		log.Printf("[context] loaded %s (%d bytes)", name, len(data))
	}

	// File tree
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
		log.Printf("[session] new %s cwd=%s", sess.ID, p.CWD)
		s.respond(req.ID, NewSessionResult{SessionID: sess.ID})
		go s.sendModelPicker(sess.ID)

	case "session/load", "session/resume":
		var p LoadSessionParams
		json.Unmarshal(req.Params, &p)
		s.mu.Lock()
		if _, exists := s.sessions[p.SessionID]; !exists {
			s.sessions[p.SessionID] = &Session{ID: p.SessionID}
		}
		s.mu.Unlock()
		log.Printf("[session] load %s", p.SessionID)
		s.respond(req.ID, NewSessionResult{SessionID: p.SessionID})

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

	// Load project context on the first turn, with visible tool calls
	if !sess.contextLoaded {
		sess.contextLoaded = true
		s.mu.Unlock()
		systemPrompt := s.loadContext(params.SessionID, sess.CWD)
		s.mu.Lock()
		if systemPrompt != "" {
			sess.History = append([]OllamaMessage{{Role: "system", Content: systemPrompt}}, sess.History...)
		}
	}

	// Collect user message
	var userText strings.Builder
	for _, b := range params.Prompt {
		if b.Type == "text" || b.Type == "resource" {
			userText.WriteString(b.Text)
		}
	}
	sess.History = append(sess.History, OllamaMessage{Role: "user", Content: userText.String()})
	messages := make([]OllamaMessage, len(sess.History))
	copy(messages, sess.History)
	s.mu.Unlock()

	s.mu.Lock()
	activeModel := sess.Model
	thinkingEnabled := sess.ThinkingEnabled
	s.mu.Unlock()

	log.Printf("[prompt] session=%s model=%s input=%d chars history=%d msgs", params.SessionID, activeModel, userText.Len(), len(messages))

	body, _ := json.Marshal(OllamaChatRequest{Model: activeModel, Messages: messages, Stream: true})

	resp, err := s.client.Post(s.ollamaURL+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("[prompt] ollama error: %v", err)
		s.respondErr(id, -32000, "upstream error: "+err.Error())
		return
	}
	defer resp.Body.Close()
	log.Printf("[prompt] upstream %d in %s", resp.StatusCode, time.Since(start))

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	thinkingStarted, thinkingDone := false, false
	thinkingTokens, contentTokens := 0, 0
	var fullResponse strings.Builder

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var chunk OllamaChunk
		if err := json.Unmarshal(line, &chunk); err != nil {
			continue
		}
		if chunk.Done {
			if chunk.EvalCount > 0 && chunk.EvalDuration > 0 {
				tps := float64(chunk.EvalCount) / (float64(chunk.EvalDuration) / 1e9)
				log.Printf("[prompt] done: %d thinking, %d content tokens, %.1f tok/s, total %s",
					thinkingTokens, contentTokens, tps, time.Since(start))
			}
			break
		}

		if chunk.Message.Thinking != "" && chunk.Message.Content == "" {
			if !thinkingStarted {
				thinkingStarted = true
				log.Printf("[prompt] thinking started (streaming=%v)", thinkingEnabled)
			}
			thinkingTokens++
			if thinkingTokens%50 == 0 {
				log.Printf("[prompt] thinking... %d tokens (+%s)", thinkingTokens, time.Since(start))
			}
			if thinkingEnabled {
				s.thought(params.SessionID, chunk.Message.Thinking)
			}
			continue
		}

		if thinkingStarted && !thinkingDone && chunk.Message.Content != "" {
			thinkingDone = true
			log.Printf("[prompt] thinking done: %d tokens in %s", thinkingTokens, time.Since(start))
		}

		if chunk.Message.Content != "" {
			contentTokens++
			fullResponse.WriteString(chunk.Message.Content)
			s.chunk(params.SessionID, chunk.Message.Content)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("[prompt] scanner error: %v", err)
	}

	s.mu.Lock()
	if sess, ok := s.sessions[params.SessionID]; ok {
		sess.History = append(sess.History, OllamaMessage{Role: "assistant", Content: fullResponse.String()})
	}
	s.mu.Unlock()

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
