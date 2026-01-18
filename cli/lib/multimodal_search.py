from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, CACHE_DIR
import os
import numpy as np
import json, hashlib


class MultimodalSearch:
    def __init__(self, docs: list[dict] | None = None, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = docs
        self.embeddings_path = os.path.join(CACHE_DIR, "clip_text_embeddings.npz")
        if docs:
            self.document_map = {i: doc for i, doc in enumerate(docs)}
            self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs]
            self._load_or_create_text_embeddings(docs)

    def _hash_documents(self, docs: list[dict]) -> str:
        doc_str = json.dumps(docs, sort_keys=True)
        return hashlib.sha256(doc_str.encode()).hexdigest()
    
    def _load_or_create_text_embeddings(self, docs: list[dict]) -> None:
        doc_hash = self._hash_documents(docs)

        if os.path.exists(self.embeddings_path):
            data = np.load(self.embeddings_path, allow_pickle=True)
            if data["doc_hash"] == doc_hash:
                self.text_embeddings = data["embeddings"]
                return

        self.text_embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        np.savez(
            self.embeddings_path,
            embeddings=self.text_embeddings,
            doc_hash=doc_hash
        )

    def embed_image(self, image_path : str) -> np.ndarray:
        image = Image.open(image_path)
        image_embedding = self.model.encode(image, normalize_embeddings=True)
        return image_embedding
    
    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        image_embedding = image_embedding.reshape(-1)
        similarities = self.text_embeddings @ image_embedding
        top_indices = np.argsort(similarities)[-DEFAULT_SEARCH_LIMIT:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.document_map[idx]
            item = doc.copy()
            item["similarity"] = float(similarities[idx])
            results.append(item) 
        
        return results


def image_search_command(image_path: str) -> None:
    docs = load_movies()
    search_engine = MultimodalSearch(docs)
    results = search_engine.search_with_image(image_path)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc['title']} (Similarity: {doc['similarity']:.4f})")
        print(f"   {doc['description'][:150]}...\n")


def verify_image_embedding(image_path: str) -> None:
    embedding_model = MultimodalSearch()
    image_embedding = embedding_model.embed_image(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")