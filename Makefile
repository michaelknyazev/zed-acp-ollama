BINARY      := zed-acp-ollama
INSTALL     := /usr/local/bin/$(BINARY)
VERSION     := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS     := -ldflags="-X main.version=$(VERSION)"
SMOKE_CWD   ?= $(CURDIR)
SMOKE_MSG   ?= list the files in this project
OLLAMA_URL  ?= https://ollama.lan

.PHONY: build install uninstall test lint clean smoke

build:
	go build $(LDFLAGS) -o $(BINARY) .

install: build
	sudo cp $(BINARY) $(INSTALL)
	@echo "Installed $(INSTALL) ($(VERSION))"

uninstall:
	sudo rm -f $(INSTALL)
	@echo "Removed $(INSTALL)"

test:
	go test ./... -v

lint:
	go vet ./...

clean:
	rm -f $(BINARY)

# Replay a real ACP session against live Ollama — no Zed needed.
# Override prompt:  make smoke SMOKE_MSG="write a hello world in Go"
# Override Ollama:  make smoke OLLAMA_URL=http://localhost:11434
smoke: build
	BINARY=./$(BINARY) OLLAMA_URL=$(OLLAMA_URL) SMOKE_CWD='$(SMOKE_CWD)' SMOKE_MSG='$(SMOKE_MSG)' \
	python3 smoke.py
