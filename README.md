# Semantica Codegraph

Python code graph indexing and hybrid retrieval for code understanding tools.
The repository combines source parsing, code chunks, graph relationships, and
lexical/semantic/fuzzy retrieval behind CLI, HTTP, and MCP interfaces.

## Scope and status

This is an earlier Python implementation of Semantica Codegraph. The repository
contains implementation, fixtures, and tests; this overview does not establish
current release qualification or comparative search performance.

## Architecture

| Area | Source | Responsibility |
| --- | --- | --- |
| Parsing | [src/parser](src/parser) | Python/TypeScript parsing, SCIP integration, and text fallback. |
| Indexing | [src/indexer](src/indexer) | Repository scanning and the indexing pipeline. |
| Storage | [src/chunking](src/chunking), [src/graph](src/graph), [src/embedding](src/embedding) | Code chunks, PostgreSQL graph data, and pgvector embeddings. |
| Retrieval | [src/search](src/search) | Lexical, semantic, graph, and fuzzy adapters with ranking and context packing. |
| Interfaces | [apps/cli](apps/cli), [apps/api](apps/api), [apps/mcp_server](apps/mcp_server) | Interactive CLI, FastAPI service, and MCP tools. |

Dependencies are assembled in [src/core/bootstrap.py](src/core/bootstrap.py).
Configuration and environment variables are defined in
[src/core/config.py](src/core/config.py).

## Local development

Requires Python 3.10 or later. Install the package and development dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The storage and retrieval paths require configured PostgreSQL/pgvector and a
lexical backend. [docker-compose.yml](docker-compose.yml) describes the local
services and expects a local `.env` file. Embedding configuration depends on the
selected provider; the default configuration uses Codestral Embed and expects
`EMBEDDING_API_KEY`. Keep local credentials outside version control.

The declared commands in [pyproject.toml](pyproject.toml) are:

- `semantica`: interactive CLI.
- `semantica-api`: HTTP API service.
- `semantica-mcp`: MCP server.

These commands require the selected services, schema, and provider configuration;
package installation alone does not provision them.

## Tests and documentation

- [tests](tests): parser, indexing, storage, retrieval, and integration coverage.
- [examples/full_search_pipeline.py](examples/full_search_pipeline.py): retrieval pipeline example.
- [docs/cli-development-guide.md](docs/cli-development-guide.md): CLI development.
- [.doc/migrations-guide.md](.doc/migrations-guide.md): database migrations.
- [.doc/benchmarks-guide.md](.doc/benchmarks-guide.md): historical benchmark workflow.

Run tests in a configured development environment with `python -m pytest`.
Integration tests require their backing services. No test or benchmark results
are claimed by this documentation update.
