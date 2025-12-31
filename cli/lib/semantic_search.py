from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import hashlib
from lib.search_utils import CACHE_DIR, load_movies


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