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
                Documents: 
                {context}
                Provide a comprehensive answer that addresses the query:"""
    
    answer = generate_content_with_fallback(prompt)

    print("Search Results:")
    for doc in retrieved_docs:
        print(f"  - {doc['metadata']['title']}")
    print("\nRAG Response:")
    print(answer)

def summarize(query: str, limit: int = 5):
    movies = load_movies()
    search_engine = HybridSearch(movies)

    retrieved_docs = search_engine.rrf_search(query, k = 60, limit=limit)
    context = "\n".join([f"- {doc['metadata']['title']}: {doc['metadata']['description']}" for doc in retrieved_docs])

    prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.
                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Search Results: 
                {context}
                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
            """
    
    summary = generate_content_with_fallback(prompt)

    print("Search Results:")
    for doc in retrieved_docs:
        print(f"  - {doc['metadata']['title']}")
    print("\nLLM Summary:")
    print(summary)

def search_with_citations(query: str, limit: int = 5):
    movies = load_movies()
    search_engine = HybridSearch(movies)

    retrieved_docs = search_engine.rrf_search(query, k = 60, limit=limit)
    context = "\n".join([f"- {doc['metadata']['title']}: {doc['metadata']['description']}" for doc in retrieved_docs])

    prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {context}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
    
    citations = generate_content_with_fallback(prompt)

    print("Search Results:")
    for doc in retrieved_docs:
        print(f"  - {doc['metadata']['title']}")
    print("\nLLM Answer:")
    print(citations)