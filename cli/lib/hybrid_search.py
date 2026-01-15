import os
import time
from .search_utils import llm_individual_rerank, llm_batch_rerank
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies as load_documents, enhance_query

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.doc_by_id = {doc["id"]: doc for doc in documents}
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def _individual_rerank(self, query: str, results: list[dict]) -> list[dict]:
        for item in results:
            score = llm_individual_rerank(query, item["metadata"])
            item["rerank_score"] = score
            time.sleep(15)  # To respect rate limits
        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    
    def _batch_rerank(self, query: str, results: list[dict]) -> list[dict]:
        ordered_ids = llm_batch_rerank(query, results)
        rank_map = {doc_id: rank for rank, doc_id in enumerate(ordered_ids, 1)}
        for item in results:
            item["rerank_rank"] = rank_map.get(item["doc_id"], float("inf"))
        return sorted(results, key=lambda x: x["rerank_rank"])

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """Perform weighted hybrid search combining BM25 and semantic search."""

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_dict = {item["id"]: item["score"] for item in bm25_results}
        semantic_dict = {item["id"]: item["score"] for item in semantic_results}

        bm25_ids = list(bm25_dict.keys())
        bm25_scores = [bm25_dict[i] for i in bm25_ids]
        bm25_norm_scores = normalize_command(bm25_scores)
        bm25_norm = dict(zip(bm25_ids, bm25_norm_scores))

        semantic_ids = list(semantic_dict.keys())
        semantic_scores = [semantic_dict[i] for i in semantic_ids]
        semantic_norm_scores = normalize_command(semantic_scores)
        semantic_norm = dict(zip(semantic_ids, semantic_norm_scores))

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
                "metadata": self.doc_by_id.get(doc_id),
            })

        hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_scores[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10, enhance_method: str = None, rerank_method: str = None) -> list[dict]:
        "Perform RRF hybrid search combining BM25 and semantic search."

        if enhance_method:
            query = enhance_query(query, enhance_method)
        
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_results.sort(key=lambda x: x["score"], reverse=True)
        semantic_results.sort(key=lambda x: x["score"], reverse=True)

        bm25_dict = {item["id"]: rank for rank, item in enumerate(bm25_results, 1)}
        semantic_dict = {item["id"]: rank for rank, item in enumerate(semantic_results, 1)}

        all_doc_ids = set(bm25_dict.keys()).union(semantic_dict.keys())
        rrf_scores = []
        for doc_id in all_doc_ids:
            bm25_rank = bm25_dict.get(doc_id, float('inf'))
            semantic_rank = semantic_dict.get(doc_id, float('inf'))

            rrf_score = self.__rrf_score(bm25_rank, k) + self.__rrf_score(semantic_rank, k)

            rrf_scores.append({
                "doc_id": doc_id,
                "rrf_score": rrf_score,
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "metadata": self.doc_by_id.get(doc_id),
            })

        rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)
        rrf_scores = rrf_scores[:limit * 5]

        if rerank_method == "individual":
            print(f"Reranking top {limit} results using individual method...")
            reranked = self._individual_rerank(query, rrf_scores)
            return reranked[:limit]
        
        if rerank_method == "batch":
            print(f"Reranking top {limit} results using batch method...")
            reranked = self._batch_rerank(query, rrf_scores)
            return reranked[:limit]
        
        return rrf_scores[:limit]

    def __rrf_score(self, rank: int, k: int) -> float:
        return 1.0 / (k + rank)


def rrf_search_command(query: str, k: int, limit: int, enhance_method: str = None, rerank_method: str = None) -> None:
    """Perform RRF hybrid search and print results."""

    documents = load_documents()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k, limit, enhance_method, rerank_method)
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
    for i, item in enumerate(results, 1):
        print(f"{i}. {item['metadata']['title']}")
        if 'rerank_score' in item:
            print(f"   Rerank Score: {item['rerank_score']:.4f}/10")
        elif 'rerank_rank' in item:
            print(f"   Rerank Rank: {item['rerank_rank']}")
        print(f"   RRF Score: {item['rrf_score']:.4f}")
        print(f"   BM25 Rank: {item['bm25_rank']}, Semantic Rank: {item['semantic_rank']}")
        print(f"   {item['metadata']['description'][:150]}...\n")


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    """Perform weighted hybrid search and print results."""

    documents = load_documents()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, item in enumerate(results, 1):
        print(f"{i}. {item['metadata']['title']}")
        print(f"   Hybrid Score: {item['hybrid_score']:.4f}")
        print(f"   BM25 Score: {item['bm25_score']:.4f}, Semantic Score: {item['semantic_score']:.4f}")
        print(f"   {item['metadata']['description'][:150]}...\n")


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
