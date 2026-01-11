from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import hashlib
from lib.search_utils import CACHE_DIR, load_movies, SCORE_PRECISION
import re


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.embeddings = None
        self.documents = None
        self.document_map = {}

        os.makedirs(CACHE_DIR, exist_ok=True)
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npz")

    # ---------------- internal helpers ----------------

    def _hash_documents(self, documents: list[dict]) -> str:
        doc_str = json.dumps(documents, sort_keys=True)
        return hashlib.sha256(doc_str.encode()).hexdigest()

    def _doc_to_text(self, doc: dict) -> str:
        return f"{doc.get('title', '')} {doc.get('description', '')}"

    # ---------------- embeddings ----------------

    def generate_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        texts = [self._doc_to_text(doc) for doc in documents]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        np.savez(
            self.embeddings_path,
            embeddings=self.embeddings,
            model_name=self.model_name,
            doc_hash=self._hash_documents(documents),
        )

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        if os.path.exists(self.embeddings_path):
            data = np.load(self.embeddings_path, allow_pickle=True)

            if (
                data["model_name"] == self.model_name
                and data["doc_hash"] == self._hash_documents(documents)
            ):
                self.embeddings = data["embeddings"]
                self.documents = documents
                self.document_map = {i: doc for i, doc in enumerate(documents)}
                return self.embeddings

        return self.build_embeddings(documents)

    # ---------------- search ----------------

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Please load or create embeddings first.")

        query_embedding = self.generate_embedding(query)

        # embeddings are normalized → dot product == cosine similarity
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[-limit:][::-1]

        results = []
        for idx in top_indices:
            doc = self.document_map[idx]
            results.append({
                "title": doc.get("title"),
                "description": doc.get("description"),
                "score": float(scores[idx]),
            })

        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npz")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        chunks = []
        metadata = []
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        for doc_id, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description.strip():
                continue
            text_chunks = semantic_chunk_command(description, 4, 1)
            l = len(text_chunks)
            for chunk_idx, chunk in enumerate(text_chunks):
                chunks.append(chunk)
                metadata.append({
                    "movie_idx": doc_id,
                    "chunk_idx": chunk_idx,
                    "total_chunks": l,
                })

        self.chunk_embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        self.chunk_metadata = metadata

        np.savez(
            self.chunk_embeddings_path,
            embeddings=self.chunk_embeddings,
            metadata=self.chunk_metadata,
            model_name=self.model_name,
            doc_hash=self._hash_documents(documents),
        )
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(metadata)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            data = np.load(self.chunk_embeddings_path, allow_pickle=True)

            if (
                data["model_name"] == self.model_name
                and data["doc_hash"] == self._hash_documents(documents)
            ):
                self.chunk_embeddings = data["embeddings"]
                self.documents = documents
                self.document_map = {i: doc for i, doc in enumerate(documents)}
                with open(self.chunk_metadata_path, "r") as f:
                    meta = json.load(f)
                    self.chunk_metadata = meta["chunks"]
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int) -> list[dict]:
        if self.chunk_embeddings is None:
            raise ValueError("Chunk embeddings not loaded. Please load or create chunk embeddings first.")

        query_embedding = self.generate_embedding(query)

        # embeddings are normalized → dot product == cosine similarity
        scores = self.chunk_embeddings @ query_embedding

        chunk_scores = []
        for idx, score in enumerate(scores):
            meta = self.chunk_metadata[idx]
            chunk_scores.append({
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": float(score),
            })

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if movie_idx not in movie_scores or chunk_score["score"] > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = chunk_score
                
        ranked_movies = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)[:limit]
        results = []
        for item in ranked_movies:
            movie = self.document_map[item["movie_idx"]]
            results.append({
                "id": movie.get("id"),
                "title": movie.get("title"),
                "document": movie.get("description", "")[:100],
                "score": round(item["score"], SCORE_PRECISION),
                "metadata": movie or {},
            })

        return results


def search_chunks_command(query: str, limit: int) -> None:
    model = ChunkedSemanticSearch()
    documents = load_movies()
    model.load_or_create_chunk_embeddings(documents)

    results = model.search_chunks(query, limit)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")


def embed_chunks_command() -> None:
    model = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = model.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def semantic_chunk_command(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be smaller than max_chunk_size")

    # 1. Strip input text
    text = text.strip()
    if not text:
        return []

    # 2. Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # 3. Handle single-sentence-without-punctuation case
    if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
        sentences = [text]

    # 4. Strip each sentence and remove empties
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            cleaned_sentences.append(s)

    if not cleaned_sentences:
        return []

    # 5. Chunk by sentence count
    chunks = []
    start = 0
    while start < len(cleaned_sentences):
        end = min(start + max_chunk_size, len(cleaned_sentences))
        chunk = " ".join(cleaned_sentences[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += max_chunk_size - overlap

    return chunks


def chunk_command(text: str, chunk_size: int = 200, overlap: int = 5) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def search_command(query: str, limit: int) -> None:
    model = SemanticSearch()
    documents = load_movies()
    model.load_or_create_embeddings(documents)

    results = model.search(query, limit)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Description: {result['description']}\n")


def embed_text(text: str) -> None:
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape[0]} dimensions")


def verify_embeddings() -> None:
    model = SemanticSearch()
    documents = load_movies()
    embeddings = model.load_or_create_embeddings(documents)
    print(f"Number of documents: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} x {embeddings.shape[1]}")


def verify_model() -> None:
    model = SemanticSearch()
    print(f"Model Loaded: {model.model_name}")
    print(f"Max Sequence Length: {model.model.max_seq_length}")