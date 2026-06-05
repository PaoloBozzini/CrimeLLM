.PHONY: help up down logs install install-all graph-init graph-status graph-wipe \
        ingest-cl ingest-cl-status ingest-cl-index load-cl \
        test lint format check clean-artifacts

DATE ?= 2024-12-31
LIMIT ?= 5000

help:
	@echo "Targets:"
	@echo "  up                  start Neo4j (docker compose)"
	@echo "  down                stop Neo4j"
	@echo "  logs                tail Neo4j logs"
	@echo "  install             uv sync --extra clg --extra dev"
	@echo "  install-all         uv sync --extra all --extra dev"
	@echo "  graph-init          apply Neo4j schema (constraints + vector index)"
	@echo "  graph-status        show schema + node counts"
	@echo "  graph-wipe          DETACH DELETE all nodes (asks for --yes)"
	@echo "  ingest-cl           clg ingest courtlistener --date $$DATE"
	@echo "  ingest-cl-status    clg ingest courtlistener-status --date $$DATE"
	@echo "  ingest-cl-index     clg ingest courtlistener-index  --date $$DATE"
	@echo "  load-cl             clg load courtlistener --date $$DATE --limit $$LIMIT"
	@echo "  test                pytest -q"
	@echo "  lint                ruff check src/ tests/"
	@echo "  format              ruff format src/ tests/"
	@echo "  check               lint + format --check + test"
	@echo ""
	@echo "Overrides: make load-cl DATE=2025-03-01 LIMIT=10000"

up:
	docker compose up -d neo4j

down:
	docker compose down

logs:
	docker compose logs -f neo4j

install:
	uv sync --extra clg --extra dev

install-all:
	uv sync --extra all --extra dev

graph-init:
	uv run clg graph init

graph-status:
	uv run clg graph status

graph-wipe:
	uv run clg graph wipe --yes

ingest-cl:
	uv run clg ingest courtlistener --date $(DATE)

ingest-cl-status:
	uv run clg ingest courtlistener-status --date $(DATE)

ingest-cl-index:
	uv run clg ingest courtlistener-index --date $(DATE)

load-cl:
	uv run clg load courtlistener --date $(DATE) --limit $(LIMIT)

test:
	uv run pytest -q

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

check: lint
	uv run ruff format --check src/ tests/
	uv run pytest -q

clean-artifacts:
	@echo "Refusing to delete artifacts/ without explicit consent."
	@echo "Run manually: rm -rf artifacts/"
	@false
