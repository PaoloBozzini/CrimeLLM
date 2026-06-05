.PHONY: up down logs install graph-init graph-status test

up:
	docker compose up -d neo4j

down:
	docker compose down

logs:
	docker compose logs -f neo4j

install:
	uv sync --extra clg --extra dev

graph-init:
	uv run clg graph init

graph-status:
	uv run clg graph status

test:
	uv run pytest -q
