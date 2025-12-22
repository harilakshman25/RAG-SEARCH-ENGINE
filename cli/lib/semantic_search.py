from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> list[float]:
        return self.model.encode([text]).tolist()[0]
    

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