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
- Cross-Encoder based reranking (local, fast)
- LLM-based individual reranking (per-document scoring)
- LLM-based batch reranking (ranked ordering by LLM)
- Query enhancement methods:
  - Spelling correction
  - Query rewriting
  - Query expansion

### RAG Capabilities
- Augmented generation with retrieved context
- Summary generation from search results
- Citation-aware answer generation
- LLM evaluation of search results (0-3 relevance scale)

## Structure
- `cli/` – CLI entrypoints and main commands
  - `semantic_search_cli.py` – Semantic search commands
  - `keyword_search_cli.py` – Keyword/BM25 search commands
  - `hybrid_search_cli.py` – Hybrid search commands
  - `multimodal_search_cli.py` – Image-based search
  - `augmented_generation_cli.py` – RAG generation
  - `evaluation_cli.py` – Evaluation utilities
  - `lib/` – Core search implementations
    - `semantic_search.py` – SemanticSearch & ChunkedSemanticSearch classes
    - `keyword_search.py` – InvertedIndex & BM25 ranking
    - `hybrid_search.py` – HybridSearch with fusion methods
    - `multimodal_search.py` – MultimodalSearch with CLIP
    - `augmented_generation.py` – RAG functions
    - `search_utils.py` – Shared utilities, LLM integration, query enhancement
- `data/` – input data (tracked)
  - `movies.json` – Movie corpus (~12k+ movies)
  - `stopwords.txt` – Common English stopwords
  - `golden_dataset.json` – Evaluation dataset
- `cache/` – generated index files (ignored)
  - `index.pkl` – Inverted index
  - `docmap.pkl` – Document mapping
  - `doc_lengths.pkl` – Document length cache
  - `term_frequencies.pkl` – Term frequency cache
  - `movie_embeddings.npz` – Semantic embeddings
  - `chunk_embeddings.npz` – Chunked embeddings
  - `chunk_metadata.json` – Chunk metadata
  - `clip_text_embeddings.npz` – CLIP embeddings
- `pyproject.toml` – Project configuration
- `available-models.txt` – List of available Gemini models

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

## Environment Setup

Requires a `.env` file with:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Keyword Search (BM25)
```bash
# Build the inverted index
uv run cli/keyword_search_cli.py build

# Search with BM25
uv run cli/keyword_search_cli.py search "action adventure"

# BM25 search with custom parameters
uv run cli/keyword_search_cli.py bm25 "sci-fi thriller" --limit 10

# Get TF-IDF scores
uv run cli/keyword_search_cli.py tfidf 1 "space"
```

### Semantic Search
```bash
# Build embeddings
uv run cli/semantic_search_cli.py embed_chunks

# Search with embeddings
uv run cli/semantic_search_cli.py search "space adventure"

# Search over chunked documents
uv run cli/semantic_search_cli.py search_chunked "romantic comedy" --limit 5

# Verify model and embeddings
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py verify_embeddings
```

### Hybrid Search
```bash
# Weighted fusion (alpha=0.5 balances both methods)
uv run cli/hybrid_search_cli.py weighted-search "action movie" --alpha 0.6 --limit 10

# RRF (Reciprocal Rank Fusion)
uv run cli/hybrid_search_cli.py rrf-search "thriller mystery" --k 60 --limit 5

# RRF with query enhancement and reranking
uv run cli/hybrid_search_cli.py rrf-search "akshun movie" \
  --enhance spell \
  --rerank-method cross_encoder \
  --limit 5

# RRF with LLM evaluation
uv run cli/hybrid_search_cli.py rrf-search "superhero film" \
  --rerank-method cross_encoder \
  --evaluate
```

### Multimodal Search (Image-based)
```bash
# Search by image
uv run cli/multimodal_search_cli.py image_search "path/to/image.jpg"

# Verify image embedding
uv run cli/multimodal_search_cli.py verify_image_embedding "path/to/image.jpg"
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
```bash
# Evaluate search quality
uv run cli/evaluation_cli.py evaluate_results query_text
```

## Key Components

### BM25 Implementation
- Inverted index with term frequencies and document lengths
- BM25 ranking with configurable k1 (default 1.5) and b (default 0.75)
- Caches created for fast retrieval

### Semantic Search
- Uses `all-MiniLM-L6-v2` model (384 dimensions)
- Normalizes embeddings for cosine similarity via dot product
- Supports chunked retrieval for better granularity

### Hybrid Search Fusion Methods
1. **Weighted Fusion**: `(1-α)*BM25 + α*Semantic` where α ∈ [0,1]
2. **RRF**: `Σ(1/(k + rank))` for each ranking method

### LLM Integration
- Fallback model chain: tries multiple Gemini models
- Handles quota exhaustion gracefully
- Supports query enhancement and LLM-based reranking

### Multimodal Search
- CLIP model (`clip-ViT-B-32`) for image-text matching
- Converts images to embeddings and scores against text descriptions
- Cached embeddings for efficiency

## Performance Characteristics

| Method | Speed | Quality | Memory |
|--------|-------|---------|--------|
| BM25 | Very Fast | Moderate | Low |
| Semantic | Medium | High | Medium |
| Chunked Semantic | Medium | Very High | Medium |
| Hybrid (Weighted) | Medium | High | Medium |
| Hybrid (RRF) | Medium | Very High | Medium |
| Multimodal | Medium | High | High |

## Implementation Notes

### Logic & Architecture
- Clean separation between CLI interfaces and core libraries
- Modular design allows independent use of each search method
- Proper error handling and fallback strategies
- Caching mechanisms prevent redundant computation
- Type hints throughout for code clarity

### Data Processing 
- Text tokenization with stopword removal and Porter stemming
- Document chunking at sentence level for semantic search
- Proper normalization of scores for fair comparison
- Support for special characters and unicode

### LLM Features 
- Automatic fallback to alternative models on failure
- Graceful quota handling prevents cascading failures
- Query enhancement methods for improved search
- Multiple reranking strategies (local and LLM-based)

## Notes

- All cache files are gitignored; they're regenerated as needed
- The corpus contains ~12,000+ movies with descriptions
- Semantic search embeddings use normalized vectors (cosine similarity via dot product)
- BM25 uses standard parameters; tune k1 and b for domain-specific data
- LLM features require GEMINI_API_KEY environment variable
- Image search uses CLIP for cross-modal retrieval
