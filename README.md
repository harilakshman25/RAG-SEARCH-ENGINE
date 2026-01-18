# RAG Search Engine

A comprehensive RAG (Retrieval-Augmented Generation) search engine with multiple retrieval strategies, hybrid search capabilities, multimodal search, and LLM-powered augmented generation.

## Features

### Search Methods
- **Keyword Search (BM25)**: Traditional text-based search using BM25 ranking algorithm with inverted index
- **Semantic Search**: Dense vector-based search using SentenceTransformer embeddings
- **Chunked Semantic Search**: Semantic search with document chunking for better granularity
- **Hybrid Search**: Combines BM25 and semantic search with multiple fusion strategies:
  - **Weighted Fusion**: Alpha-weighted combination of keyword and semantic scores
  - **Reciprocal Rank Fusion (RRF)**: Rank-based fusion (k-parameter configurable)
- **Multimodal Search**: Search movies using image queries with CLIP embeddings

### Ranking & Reranking
- **Cross-Encoder Reranking**: High-precision local reranking
- **LLM Reranking (Async)**: Individual document scoring using LLMs, parallelized with `asyncio` for high performance
- **LLM Batch Reranking**: Ranked ordering by LLM
- **Query Enhancement**:
  - Spelling correction
  - Query rewriting (Text & Multimodal)
  - Query expansion

### RAG Capabilities
- Augmented generation with retrieved context
- Summary generation from search results
- Citation-aware answer generation
- **LLM Evaluation**: AI-based relevance scoring of search results (0-3 scale)

### Performance Optimizations
- **Async Processing**: Reranking operations run in parallel to minimize latency
- **Disk Caching**: LLM responses are hashed and cached to disk, preventing redundant API calls and saving quota
- **Fail-Fast Logic**: Smart handling of API quotas to switch fallback models immediately

## Structure
- `cli/` – CLI entrypoints and main commands
  - `semantic_search_cli.py` – Semantic search commands
  - `keyword_search_cli.py` – Keyword/BM25 search commands
  - `hybrid_search_cli.py` – Hybrid search commands
  - `multimodal_search_cli.py` – Image-based search
  - `describe_image_cli.py` – Vision-language query rewriting
  - `augmented_generation_cli.py` – RAG generation
  - `evaluation_cli.py` – Evaluation utilities (Precision/Recall/MAP)
- `lib/` – Core search implementations
  - `semantic_search.py` – SemanticSearch & ChunkedSemanticSearch classes
  - `keyword_search.py` – InvertedIndex & BM25 ranking
  - `hybrid_search.py` – HybridSearch with fusion methods
  - `multimodal_search.py` – MultimodalSearch with CLIP
  - `augmented_generation.py` – RAG functions
  - `search_utils.py` – Shared utilities, Async LLM integration, Caching
- `data/` – input data (tracked)
  - `movies.json` – Movie corpus (~12k+ movies)
  - `stopwords.txt` – Common English stopwords
  - `golden_dataset.json` – Evaluation dataset
- `cache/` – generated index files (ignored)
  - `index.pkl` – Inverted index
  - `llm_responses/` – Disk cache for LLM prompts
  - `docmap.pkl` – Document mapping
  - `doc_lengths.pkl` – Document length cache
  - `term_frequencies.pkl` – Term frequency cache
  - `movie_embeddings.npz` – Semantic embeddings
  - `chunk_embeddings.npz` – Chunked embeddings
  - `chunk_metadata.json` – Chunk metadata
  - `clip_text_embeddings.npz` – CLIP embeddings

## Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate on Windows

# Install dependencies
uv sync

# Set environment variable for Gemini API
export GEMINI_API_KEY="your-api-key-here"

```

## Usage

### Keyword Search (BM25)

```bash
# Build the inverted index
uv run cli/keyword_search_cli.py build

# Search with BM25
uv run cli/keyword_search_cli.py search "action adventure"

# BM25 search with custom parameters
uv run cli/keyword_search_cli.py bm25search "sci-fi thriller" --limit 10

```

### Semantic Search

```bash
# Build embeddings
uv run cli/semantic_search_cli.py embed_chunks

# Search with embeddings
uv run cli/semantic_search_cli.py search "space adventure"

# Search over chunked documents
uv run cli/semantic_search_cli.py search_chunked "romantic comedy" --limit 5

```

### Hybrid Search

```bash
# Weighted fusion (alpha=0.5 balances both methods)
uv run cli/hybrid_search_cli.py weighted-search "action movie" --alpha 0.6 --limit 10

# RRF (Reciprocal Rank Fusion)
uv run cli/hybrid_search_cli.py rrf-search "thriller mystery" --k 60 --limit 5

# RRF with query enhancement and Async LLM reranking
uv run cli/hybrid_search_cli.py rrf-search "akshun movie" \
  --enhance spell \
  --rerank-method individual \
  --limit 5

# RRF with LLM evaluation (0-3 Relevance Score)
uv run cli/hybrid_search_cli.py rrf-search "superhero film" \
  --rerank-method cross_encoder \
  --evaluate

```

### Multimodal Search & Vision

> **Note:** All image files must be placed in the `data/` directory for the scripts to access them.

```bash
# Search by image
uv run cli/multimodal_search_cli.py image_search "poster.jpg"

# Rewrite a text query using visual context from an image
uv run cli/describe_image_cli.py --image "poster.jpg" --query "find movies like this"

```

### RAG & Augmented Generation

```bash
# Basic RAG answer
uv run cli/augmented_generation_cli.py rag "What are some good sci-fi movies?"

# Generate summary from search results
uv run cli/augmented_generation_cli.py summarize "action movies" --limit 5

# Answer with citations
uv run cli/augmented_generation_cli.py citations "Best thriller movies"

```

### Evaluation

The evaluation CLI runs the engine against `data/golden_dataset.json` and calculates global metrics.

```bash
# Run evaluation (Precision@K, Recall@K, F1)
# Also reports MAP (Mean Average Precision) and MAR (Mean Average Recall)
uv run cli/evaluation_cli.py --limit 5

```

## Implementation Notes

### Architecture & Performance

* **AsyncIO**: The `HybridSearch` class uses `asyncio` and `semaphores` to perform LLM-based reranking in parallel, significantly reducing the wait time compared to sequential processing.
* **Caching**: A persistent disk cache (`cache/llm_responses/`) stores every unique prompt sent to the LLM. Re-running queries or tests is nearly instantaneous and consumes zero API quota.

### Hybrid Search Fusion Methods

1. **Weighted Fusion**: `(1-α)*BM25 + α*Semantic` where α ∈ [0,1]
2. **RRF**: `Σ(1/(k + rank))` for each ranking method

### Multimodal Search

* CLIP model (`clip-ViT-B-32`) for image-text matching
* Converts images to embeddings and scores against text descriptions
* Cached embeddings for efficiency

## Notes

* All cache files are gitignored; they're regenerated as needed.
* LLM features require `GEMINI_API_KEY` environment variable.
* To force a refresh of LLM results (bypass cache), delete the `cache/llm_responses` directory.
