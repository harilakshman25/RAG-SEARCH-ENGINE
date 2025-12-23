from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import CACHE_DIR, load_movies
import os

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str) -> list[float]:
        return self.model.encode([text]).tolist()[0]
    
    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        texts = [f"{doc['title']} {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
        if self.embeddings is not None and self.embeddings.shape[0] == len(documents):
            self.documents = documents
            self.document_map = {i: doc for i, doc in enumerate(documents)}
        else:
            self.build_embeddings(documents)
        return self.embeddings


def embed_query_text(query: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    model = SemanticSearch(model_name)
    embedding = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {len(embedding)} dimensions")
    

def verify_embeddings() -> None:
    model = SemanticSearch()
    documents = load_movies()
    embeddings = model.load_or_create_embeddings(documents)
    print(f"Number of documents: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    

def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    model = SemanticSearch(model_name)
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {len(embedding)}")


def verify_model() -> None:
    model = SemanticSearch()
    print(f"Model Loaded: {model.model}")
    print(f"Max Sequence Length: {model.model.max_seq_length}")