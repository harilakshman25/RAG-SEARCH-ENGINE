import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies as load_documents

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """Perform weighted hybrid search combining BM25 and semantic search."""

        # 1. Retrieve large candidate sets
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        # 2. Build raw score maps
        bm25_dict = {item["id"]: item["score"] for item in bm25_results}
        semantic_dict = {item["id"]: item["score"] for item in semantic_results}

        # 3. Normalize scores ACROSS ALL DOCUMENTS
        bm25_ids = list(bm25_dict.keys())
        bm25_scores = [bm25_dict[i] for i in bm25_ids]
        bm25_norm_scores = normalize_command(bm25_scores)
        bm25_norm = dict(zip(bm25_ids, bm25_norm_scores))

        semantic_ids = list(semantic_dict.keys())
        semantic_scores = [semantic_dict[i] for i in semantic_ids]
        semantic_norm_scores = normalize_command(semantic_scores)
        semantic_norm = dict(zip(semantic_ids, semantic_norm_scores))

        # 4. Combine scores
        all_doc_ids = set(bm25_dict.keys()).union(semantic_dict.keys())
        hybrid_scores = []

        for doc_id in all_doc_ids:
            bm25_score = bm25_norm.get(doc_id, 0.0)
            semantic_score = semantic_norm.get(doc_id, 0.0)

            hybrid_score = (1 - alpha) * bm25_score + alpha * semantic_score

            hybrid_scores.append({
                "doc_id": doc_id,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "hybrid_score": hybrid_score,
                "metadata": self.documents[doc_id],
            })

        # 5. Sort and truncate
        hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_scores[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def weighted_search_command(query: str, alpha: float, limit: int) -> list[dict]:
    """Perform weighted hybrid search and print results."""

    documents = load_documents()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)

    return results


def normalize_command(scores: list[float]) -> list[float]:
    """Normalize a list of scores to the range [0, 1]."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [0.0 for _ in scores]

    normalized_scores = [
        (score - min_score) / (max_score - min_score) for score in scores
    ]
    return normalized_scores
