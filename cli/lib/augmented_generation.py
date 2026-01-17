from .search_utils import (
    generate_content_with_fallback,
    load_movies,
)
from .hybrid_search import HybridSearch

def perform_rag(query: str):
    movies = load_movies()
    search_engine = HybridSearch(movies)

    retrieved_docs = search_engine.rrf_search(query, k = 60, limit=5)
    context = "\n".join([f"- {doc['metadata']['title']}: {doc['metadata']['description']}" for doc in retrieved_docs])

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Context: {context}
                Provide a comprehensive answer that addresses the query:"""
    
    answer = generate_content_with_fallback(prompt)

    print("Search Results:")
    for doc in retrieved_docs:
        print(f"  - {doc['metadata']['title']}")
    print("\nRAG Response:")
    print(answer)
