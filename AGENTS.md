# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11+ FastAPI service that runs a LangGraph-based, multi-agent conversation-analysis pipeline.

- `src/api/`: FastAPI application, endpoints, and execution-status storage. The application entry point is `src.api.main:app`.
- `src/expert/`: individual analysis nodes (preprocessing, translation, DPICS labeling, coaching, aggregation).
- `src/router/`: LangGraph state schema and graph routing. Keep pipeline wiring here rather than in API handlers.
- `src/utils/`: shared LLM, embedding, vector-store, prompt, and DPICS helpers.
- `data/expert_advice/` and `data/sql/`: RAG source data and PostgreSQL/pgvector schema scripts.
- `scripts/`: operational utilities, including vector-index build and search verification scripts.

## Build, Test, and Development Commands

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API with reload using `./run_server.sh`, or directly:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Use `http://localhost:8000/docs` to exercise the API. Build RAG data with `python scripts/build_vector_index.py`; validate database-backed retrieval with `python scripts/test_vector_search.py --query "칭찬하기"`. Container workflows use `docker compose up --build`.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` for functions, variables, and modules, and `PascalCase` for classes. Name graph nodes after their responsibility, such as `summarize_node` or `coaching_node`. Keep API request/response handling in `src/api/`, reusable integrations in `src/utils/`, and avoid embedding provider-specific behavior in agent nodes when a shared helper fits.

No formatter or linter is configured in the repository. Preserve the surrounding file’s import order, typing style, docstrings, and Korean/English terminology; keep changes focused and readable.

## Testing Guidelines

There is currently no `pytest` suite. For vector-search changes, run `scripts/test_vector_search.py` against a configured PostgreSQL/pgvector instance. For API changes, start the server and verify the affected endpoint through Swagger or `curl`, including `/analyze` and `/status/{execution_id}`. Add focused tests alongside new test infrastructure when introducing behavior that can be isolated.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit-style subjects, commonly `feat:`, `chore:`, and `test:`; Korean descriptions are also established. Example: `feat: add OpenRouter provider for LLM and embeddings`.

Keep commits narrowly scoped. Pull requests should explain the behavioral change, link the relevant issue when available, list validation performed, and include request/response examples or screenshots for API-visible changes. Never commit `.env`, API keys, local models, or database credentials.
