# RAG Search Engine

Minimal RAG-style keyword search engine.

## Development Branches

This project was built incrementally. Each branch represents a successive stage in the development of the RAG search engine, with every stage building directly on top of the previous one.

The branches form a **linear history** with no merge branches:

```text
preprocessing_and_tf_idf
        ↓
keyword_search
        ↓
semantic_search
        ↓
chunking
        ↓
hybrid_Search
        ↓
LLMS
        ↓
Reranking
        ↓
evaluation
        ↓
augmented_generation
        ↓
multimodal
```

### Branch Progression

1. **`preprocessing_and_tf_idf`** — Text preprocessing and TF-IDF scoring
2. **`keyword_search`** — BM25-based keyword search
3. **`semantic_search`** — Semantic search using document and query embeddings
4. **`chunking`** — Semantic chunking, chunk embeddings, and chunk-aware retrieval
5. **`hybrid_Search`** — Hybrid retrieval using score normalization, weighted combination, and Reciprocal Rank Fusion
6. **`LLMS`** — Query processing with spell correction, query rewriting, and query expansion
7. **`Reranking`** — LLM and cross-encoder based reranking with error handling
8. **`evaluation`** — Retrieval and LLM evaluation using a golden dataset and Precision/Recall/F1 metrics
9. **`augmented_generation`** — Retrieval-Augmented Generation with summarization and citation support
10. **`multimodal`** — Multimodal search and the final evolution of the project

The `multimodal` branch represents the final stage of development. It contains the complete implementation and the latest project structure and instructions.

To explore how the system evolved, the branches can be visited sequentially in the order listed above.

