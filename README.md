# RAG Search Engine

Minimal RAG-style keyword search engine.

## Structure
- `cli/` – CLI entrypoints and helpers
- `data/` – input data (tracked)
  - `movies.json` – corpus
  - `stopwords.txt` – stopword list
- `cache/` – generated index files (ignored)
- `pyproject.toml` – project config
- `uv.lock` – dependency lockfile

## Setup
```bash
uv venv
uv sync
