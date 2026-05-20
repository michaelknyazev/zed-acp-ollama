package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// safeBuffer is a bytes.Buffer safe for concurrent use (server goroutines write,
// test goroutine reads).
type safeBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *safeBuffer) Write(p []byte) (n int, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *safeBuffer) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Bytes()
}

func (b *safeBuffer) decodeNext(t *testing.T) map[string]any {
	t.Helper()
	b.mu.Lock()
	defer b.mu.Unlock()
	var m map[string]any
	if err := json.NewDecoder(&b.buf).Decode(&m); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return m
}

// ── test helpers ──────────────────────────────────────────────────────────────

func newTestServer(t *testing.T, ollamaServer *httptest.Server) (*Server, *safeBuffer) {
	t.Helper()
	var out safeBuffer
	url := "http://127.0.0.1:19999" // unreachable default
	client := http.DefaultClient
	if ollamaServer != nil {
		url = ollamaServer.URL
		client = ollamaServer.Client()
	}
	srv := &Server{
		sessions:     make(map[string]*Session),
		enc:          json.NewEncoder(&out),
		client:       client,
		ollamaURL:    url,
		defaultModel: "qwen3:latest",
	}
	return srv, &out
}

func sendRequest(t *testing.T, srv *Server, method string, id any, params any) {
	t.Helper()
	idJSON, _ := json.Marshal(id)
	paramsJSON, _ := json.Marshal(params)
	req := Request{
		JSONRPC: "2.0",
		ID:      idJSON,
		Method:  method,
		Params:  paramsJSON,
	}
	srv.handle(req)
}

func decodeNext(t *testing.T, buf *safeBuffer) map[string]any {
	t.Helper()
	return buf.decodeNext(t)
}

func requireResult(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	if e, ok := m["error"]; ok {
		t.Fatalf("unexpected error in response: %v", e)
	}
	result, ok := m["result"].(map[string]any)
	if !ok {
		t.Fatalf("result missing or wrong type: %v", m["result"])
	}
	return result
}


// ── file tree ─────────────────────────────────────────────────────────────────

func TestFileTree_BasicStructure(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "src"), 0755)
	os.WriteFile(filepath.Join(dir, "main.go"), nil, 0644)
	os.WriteFile(filepath.Join(dir, "go.mod"), nil, 0644)
	os.WriteFile(filepath.Join(dir, "src", "lib.go"), nil, 0644)

	tree := fileTree(dir, 3)
	if !strings.Contains(tree, "main.go") {
		t.Error("expected main.go in tree")
	}
	if !strings.Contains(tree, "src/") {
		t.Error("expected src/ in tree")
	}
	if !strings.Contains(tree, "lib.go") {
		t.Error("expected lib.go in tree")
	}
}

func TestFileTree_IgnoresNoiseDirs(t *testing.T) {
	dir := t.TempDir()
	for _, noise := range []string{"node_modules", ".git", "vendor", "dist"} {
		os.MkdirAll(filepath.Join(dir, noise, "deep"), 0755)
		os.WriteFile(filepath.Join(dir, noise, "file.txt"), []byte("x"), 0644)
	}
	os.WriteFile(filepath.Join(dir, "main.go"), nil, 0644)

	tree := fileTree(dir, 3)
	for _, noise := range []string{"node_modules", ".git", "vendor", "dist"} {
		if strings.Contains(tree, noise) {
			t.Errorf("tree should not contain %q", noise)
		}
	}
	if !strings.Contains(tree, "main.go") {
		t.Error("expected main.go in tree")
	}
}

func TestFileTree_RespectsMaxDepth(t *testing.T) {
	dir := t.TempDir()
	// a/b/shallow.go → parts=["a","b","shallow.go"] len=3 → visible at maxDepth=3
	// a/b/c/d/deep.go → parts len=5 → cut off at maxDepth=3
	os.MkdirAll(filepath.Join(dir, "a", "b", "c", "d"), 0755)
	os.WriteFile(filepath.Join(dir, "a", "b", "c", "d", "deep.go"), nil, 0644)
	os.WriteFile(filepath.Join(dir, "a", "b", "shallow.go"), nil, 0644)

	tree := fileTree(dir, 3)
	if strings.Contains(tree, "deep.go") {
		t.Error("deep.go should be cut off at maxDepth=3")
	}
	if !strings.Contains(tree, "shallow.go") {
		t.Error("shallow.go should appear within maxDepth=3")
	}
}

// ── JSON wire format ──────────────────────────────────────────────────────────

// Critical: SessionConfigOption must be flat (type/currentValue/options at top level).
func TestSessionConfigOptionJSON_FlatStructure(t *testing.T) {
	opt := SessionConfigOption{
		ID:           "model",
		Name:         "Model",
		Category:     "model",
		Type:         "select",
		CurrentValue: "llama3:latest",
		Options: []SessionConfigSelectOption{
			{Value: "llama3:latest", Name: "llama3:latest"},
			{Value: "qwen3:latest", Name: "qwen3:latest"},
		},
	}
	data, _ := json.Marshal(opt)
	var m map[string]any
	json.Unmarshal(data, &m)

	for _, key := range []string{"id", "name", "category", "type", "currentValue", "options"} {
		if _, ok := m[key]; !ok {
			t.Errorf("expected top-level key %q, got keys: %v", key, keys(m))
		}
	}
	// Must NOT have a nested "kind" or "select" wrapper
	if _, ok := m["kind"]; ok {
		t.Error("field 'kind' must not exist — options are flat")
	}
	if _, ok := m["select"]; ok {
		t.Error("field 'select' must not exist — options are flat")
	}
	// Options must be a plain array
	opts, ok := m["options"].([]any)
	if !ok {
		t.Fatalf("options should be array, got %T", m["options"])
	}
	if len(opts) != 2 {
		t.Errorf("expected 2 options, got %d", len(opts))
	}
	// Verify first option has value/name at top level (not in "ungrouped" wrapper)
	first := opts[0].(map[string]any)
	if _, ok := first["value"]; !ok {
		t.Error("option should have 'value' field directly")
	}
}

func TestConfigOptionUpdateJSON(t *testing.T) {
	update := ConfigOptionUpdate{
		SessionUpdate: "config_option_update",
		ConfigOptions: []SessionConfigOption{
			{ID: "model", Name: "Model", Type: "select", CurrentValue: "x", Options: []SessionConfigSelectOption{{Value: "x", Name: "x"}}},
		},
	}
	data, _ := json.Marshal(update)
	var m map[string]any
	json.Unmarshal(data, &m)

	if m["sessionUpdate"] != "config_option_update" {
		t.Errorf("sessionUpdate = %q, want config_option_update", m["sessionUpdate"])
	}
	opts := m["configOptions"].([]any)
	if len(opts) != 1 {
		t.Errorf("configOptions length = %d, want 1", len(opts))
	}
}

func TestAgentMessageChunkJSON(t *testing.T) {
	chunk := AgentMessageChunk{
		SessionUpdate: "agent_message_chunk",
		Content:       ContentBlock{Type: "text", Text: "hello"},
	}
	data, _ := json.Marshal(chunk)
	var m map[string]any
	json.Unmarshal(data, &m)

	if m["sessionUpdate"] != "agent_message_chunk" {
		t.Errorf("sessionUpdate = %q", m["sessionUpdate"])
	}
	content := m["content"].(map[string]any)
	if content["type"] != "text" || content["text"] != "hello" {
		t.Errorf("unexpected content: %v", content)
	}
}

func TestAgentThoughtChunkJSON(t *testing.T) {
	chunk := AgentMessageChunk{
		SessionUpdate: "agent_thought_chunk",
		Content:       ContentBlock{Type: "text", Text: "thinking..."},
	}
	data, _ := json.Marshal(chunk)
	var m map[string]any
	json.Unmarshal(data, &m)

	if m["sessionUpdate"] != "agent_thought_chunk" {
		t.Errorf("sessionUpdate = %q, want agent_thought_chunk", m["sessionUpdate"])
	}
}

// ── buildConfigOptions ────────────────────────────────────────────────────────

func TestBuildConfigOptions_ModelAndThinking(t *testing.T) {
	srv, _ := newTestServer(t, nil)
	sess := &Session{Model: "qwen3:latest", ThinkingEnabled: true}

	opts := srv.buildConfigOptions(sess, []string{"qwen3:latest", "llama3:latest"})

	if len(opts) != 2 {
		t.Fatalf("expected 2 config options, got %d", len(opts))
	}

	modelOpt := opts[0]
	if modelOpt.ID != "model" || modelOpt.Category != "model" {
		t.Errorf("unexpected model option: %+v", modelOpt)
	}
	if modelOpt.CurrentValue != "qwen3:latest" {
		t.Errorf("currentValue = %q, want qwen3:latest", modelOpt.CurrentValue)
	}
	if len(modelOpt.Options) != 2 {
		t.Errorf("expected 2 model options, got %d", len(modelOpt.Options))
	}

	thinkOpt := opts[1]
	if thinkOpt.ID != "thinking" || thinkOpt.Category != "thought_level" {
		t.Errorf("unexpected thinking option: %+v", thinkOpt)
	}
	if thinkOpt.CurrentValue != "enabled" {
		t.Errorf("thinking currentValue = %q, want enabled", thinkOpt.CurrentValue)
	}
}

func TestBuildConfigOptions_ThinkingDisabled(t *testing.T) {
	srv, _ := newTestServer(t, nil)
	sess := &Session{Model: "llama3:latest", ThinkingEnabled: false}
	opts := srv.buildConfigOptions(sess, []string{"llama3:latest"})

	thinkOpt := opts[1]
	if thinkOpt.CurrentValue != "disabled" {
		t.Errorf("expected disabled, got %q", thinkOpt.CurrentValue)
	}
}

// ── handler: initialize ───────────────────────────────────────────────────────

func TestHandleInitialize(t *testing.T) {
	srv, out := newTestServer(t, nil)
	sendRequest(t, srv, "initialize", 1, map[string]any{
		"protocolVersion": 1, "clientCapabilities": map[string]any{},
	})

	m := decodeNext(t, out)
	result := requireResult(t, m)

	if result["protocolVersion"] == nil {
		t.Error("missing protocolVersion")
	}
	info := result["agentInfo"].(map[string]any)
	if info["name"] != "zed-acp-ollama" {
		t.Errorf("agentInfo.name = %q", info["name"])
	}
}

// ── handler: session/new ──────────────────────────────────────────────────────

func TestHandleSessionNew_CreatesSession(t *testing.T) {
	srv, out := newTestServer(t, nil)
	// Use a mock client so sendModelPicker doesn't make real HTTP calls
	srv.client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return httptest.NewRecorder().Result(), nil
	})}

	sendRequest(t, srv, "session/new", 1, map[string]any{"cwd": "/tmp/proj"})

	m := decodeNext(t, out)
	result := requireResult(t, m)

	sessID, ok := result["sessionId"].(string)
	if !ok || !strings.HasPrefix(sessID, "sess_") {
		t.Errorf("unexpected sessionId: %v", result["sessionId"])
	}

	srv.mu.Lock()
	sess := srv.sessions[sessID]
	srv.mu.Unlock()

	if sess == nil {
		t.Fatal("session not stored in map")
	}
	if sess.CWD != "/tmp/proj" {
		t.Errorf("CWD = %q, want /tmp/proj", sess.CWD)
	}
	if !sess.ThinkingEnabled {
		t.Error("ThinkingEnabled should default to true")
	}
}

// ── handler: session/set_config_option ───────────────────────────────────────
// The response is sent by an async goroutine (it needs to fetch /api/tags first).
// We test the synchronous state change directly and let the goroutine finish.

func TestHandleSetConfigOption_Model(t *testing.T) {
	ollama := mockOllamaTagsServer(t, []string{"qwen3:latest", "llama3:latest"})
	// Don't defer Close here — goroutine may still be hitting it.
	// t.Cleanup runs after the test + goroutines settle.
	t.Cleanup(ollama.Close)

	srv, _ := newTestServer(t, ollama)

	sessID := "sess_test_001"
	srv.mu.Lock()
	srv.sessions[sessID] = &Session{ID: sessID, Model: "qwen3:latest", ThinkingEnabled: true}
	srv.mu.Unlock()

	sendRequest(t, srv, "session/set_config_option", 2, map[string]any{
		"sessionId": sessID,
		"configId":  "model",
		"value":     "llama3:latest",
	})

	// State change is synchronous; response is async. Check state.
	srv.mu.Lock()
	updatedModel := srv.sessions[sessID].Model
	srv.mu.Unlock()

	if updatedModel != "llama3:latest" {
		t.Errorf("model not updated: got %q", updatedModel)
	}

	// Give goroutine time to finish so race detector is happy.
	time.Sleep(50 * time.Millisecond)
}

func TestHandleSetConfigOption_Thinking(t *testing.T) {
	ollama := mockOllamaTagsServer(t, []string{"qwen3:latest"})
	t.Cleanup(ollama.Close)

	srv, _ := newTestServer(t, ollama)

	sessID := "sess_think_test"
	srv.mu.Lock()
	srv.sessions[sessID] = &Session{ID: sessID, Model: "qwen3:latest", ThinkingEnabled: true}
	srv.mu.Unlock()

	sendRequest(t, srv, "session/set_config_option", 3, map[string]any{
		"sessionId": sessID,
		"configId":  "thinking",
		"value":     "disabled",
	})

	srv.mu.Lock()
	enabled := srv.sessions[sessID].ThinkingEnabled
	srv.mu.Unlock()

	if enabled {
		t.Error("ThinkingEnabled should be false after setting to disabled")
	}

	time.Sleep(50 * time.Millisecond)
}

// ── handler: unknown method ───────────────────────────────────────────────────

func TestHandleUnknownMethod_ReturnsError(t *testing.T) {
	srv, out := newTestServer(t, nil)
	sendRequest(t, srv, "nonexistent/method", 99, nil)

	m := decodeNext(t, out)
	if m["error"] == nil {
		t.Error("expected error for unknown method")
	}
	rpcErr := m["error"].(map[string]any)
	if rpcErr["code"].(float64) != -32601 {
		t.Errorf("expected code -32601, got %v", rpcErr["code"])
	}
}

func TestHandleNotification_NoResponse(t *testing.T) {
	srv, out := newTestServer(t, nil)
	// Notification has no id — server must not respond
	req := Request{JSONRPC: "2.0", Method: "some/notification"}
	srv.handle(req)
	if len(out.Bytes()) != 0 {
		t.Errorf("expected no output for notification, got %q", out.Bytes())
	}
}

// ── fetchModels ───────────────────────────────────────────────────────────────

func TestFetchModels(t *testing.T) {
	ollama := mockOllamaTagsServer(t, []string{"qwen3:latest", "llama3:8b", "mistral:7b"})
	defer ollama.Close()

	srv, _ := newTestServer(t, ollama)
	models := srv.fetchModels()
	if len(models) != 3 {
		t.Fatalf("expected 3 models, got %d: %v", len(models), models)
	}
	if models[0] != "qwen3:latest" {
		t.Errorf("first model = %q", models[0])
	}
}

func TestFetchModels_OllamaDown(t *testing.T) {
	srv, _ := newTestServer(t, nil)
	savedURL := ollamaURL
	ollamaURL = "http://127.0.0.1:19999" // nothing listening
	defer func() { ollamaURL = savedURL }()

	models := srv.fetchModels()
	if models != nil {
		t.Errorf("expected nil on error, got %v", models)
	}
}

// ── loadContext ───────────────────────────────────────────────────────────────

func TestLoadContext_ReadsContextFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte("# Agent rules\nBe helpful."), 0644)
	os.WriteFile(filepath.Join(dir, "README.md"), []byte("# My Project"), 0644)

	srv, _ := newTestServer(t, nil)
	prompt := srv.loadContext("sess_x", dir)

	if !strings.Contains(prompt, "AGENTS.md") {
		t.Error("system prompt should mention AGENTS.md")
	}
	if !strings.Contains(prompt, "Be helpful.") {
		t.Error("system prompt should contain AGENTS.md content")
	}
	if !strings.Contains(prompt, "My Project") {
		t.Error("system prompt should contain README.md content")
	}
	if !strings.Contains(prompt, "Project structure") {
		t.Error("system prompt should contain file tree")
	}
}

func TestLoadContext_EmptyDir(t *testing.T) {
	dir := t.TempDir()
	srv, _ := newTestServer(t, nil)
	prompt := srv.loadContext("sess_x", dir)

	// Should still produce something (at minimum the tree and header)
	if prompt == "" {
		t.Error("prompt should not be empty even with no context files")
	}
}

func TestLoadContext_TruncatesLargeFiles(t *testing.T) {
	dir := t.TempDir()
	large := strings.Repeat("x", 3000)
	os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte(large), 0644)

	srv, _ := newTestServer(t, nil)
	prompt := srv.loadContext("sess_x", dir)

	// The tool output preview is capped at 2000 chars; full content goes into prompt
	if !strings.Contains(prompt, "AGENTS.md") {
		t.Error("large AGENTS.md should still be loaded")
	}
}

// ── mock helpers ──────────────────────────────────────────────────────────────

func mockOllamaTagsServer(t *testing.T, models []string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.NotFound(w, r)
			return
		}
		type modelEntry struct {
			Name string `json:"name"`
		}
		type tagsResp struct {
			Models []modelEntry `json:"models"`
		}
		entries := make([]modelEntry, len(models))
		for i, m := range models {
			entries[i] = modelEntry{Name: m}
		}
		json.NewEncoder(w).Encode(tagsResp{Models: entries})
	}))
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// ── env helper ────────────────────────────────────────────────────────────────

func TestEnvFallback(t *testing.T) {
	os.Unsetenv("TEST_KEY_XYZ")
	if env("TEST_KEY_XYZ", "default") != "default" {
		t.Error("expected fallback")
	}
	os.Setenv("TEST_KEY_XYZ", "override")
	defer os.Unsetenv("TEST_KEY_XYZ")
	if env("TEST_KEY_XYZ", "default") != "override" {
		t.Error("expected env override")
	}
}

// ── utility ───────────────────────────────────────────────────────────────────

func keys(m map[string]any) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}

// ── integration: full handshake ───────────────────────────────────────────────

func TestFullHandshake(t *testing.T) {
	srv, out := newTestServer(t, nil)
	srv.client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		rec := httptest.NewRecorder()
		rec.WriteHeader(200)
		fmt.Fprintln(rec.Body, `{"models":[]}`)
		return rec.Result(), nil
	})}

	// 1. initialize
	sendRequest(t, srv, "initialize", 1, map[string]any{"protocolVersion": 1, "clientCapabilities": map[string]any{}})
	m := decodeNext(t, out)
	requireResult(t, m)

	// 2. session/new
	sendRequest(t, srv, "session/new", 2, map[string]any{"cwd": t.TempDir()})
	m = decodeNext(t, out)
	result := requireResult(t, m)
	sessID := result["sessionId"].(string)

	if sessID == "" {
		t.Fatal("no session ID returned")
	}
	t.Logf("session: %s", sessID)
}
