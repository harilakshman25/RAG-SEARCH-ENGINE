import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError, ServerError, ClientError

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 4

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.json")
STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

FALLBACK_MODELS = [
    "gemini-2.5-flash",          
    "gemini-2.5-flash-lite",     
    "gemini-3-flash-preview",    
    "gemini-2.0-flash",          
    "gemini-2.0-flash-lite",     
    "gemini-flash-latest",       
    "gemini-flash-lite-latest",  
    "gemini-3-pro-preview",      
    "gemini-2.0-flash-001",      
    "gemma-3-27b-it"             
]

def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = [line.strip() for line in f]
    return stopwords

client = genai.Client(api_key=api_key)

def generate_content_with_fallback(prompt: str) -> str:
    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if not response or not getattr(response, "text", None):
                print(f"[EMPTY] {model_name} returned no text. Trying next model...")
                continue

            return response.text.strip()

        except ClientError as e:
            msg = str(e).upper()
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                print(f"[HARD LIMIT] Quota exhausted on {model_name}. Skipping remaining LLMs.")
                break
            print(f"[CLIENT ERROR] {model_name}: {e}. Trying next model...")
            continue

        except (APIError, ServerError) as e:
            print(f"[SERVER ERROR] {model_name} failed: {e}. Trying fallback...")
            continue

        except Exception as e:
            print(f"[ERROR] Unexpected failure with {model_name}: {e}. Trying fallback...")
            continue

    raise RuntimeError("CRITICAL: All available Gemini models failed or returned empty responses.")

def enhance_query(query: str, method: str) -> str:
    try:
        if method == "spell":
            enhanced_query = spelling_corrector(query)
        elif method == "rewrite":
            enhanced_query = rewriter(query)
        elif method == "expand":
            enhanced_query = expander(query)
        else:
            enhanced_query = query

        if query.strip() != enhanced_query:
            print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")

        return enhanced_query

    except Exception as e:
        print(f"[WARN] Query enhancement ({method}) failed: {str(e)[:100]} – using original query.\n")
        return query

def spelling_corrector(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.
                Only correct obvious typos. Don't change correctly spelled words.
                Query: "{query}"
                If no errors, return the original query.
                Corrected:"""
    response_text = generate_content_with_fallback(prompt)
    if "Corrected:" in response_text:
        return response_text.split("Corrected:")[-1].strip().strip('"')
    return response_text.strip().strip('"')

def rewriter(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.
                Original: "{query}"
                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions
                - Keep it concise (under 10 words)
                - Don't use boolean logic
                Rewritten query:"""
    response_text = generate_content_with_fallback(prompt)
    if "Rewritten query:" in response_text:
        return response_text.split("Rewritten query:")[-1].strip().strip('"')
    return response_text.strip().strip('"')

def expander(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.
                Add synonyms and related concepts that might appear in movie descriptions.
                Query: "{query}"
                Expanded query:"""
    response_text = generate_content_with_fallback(prompt)
    if "Expanded query:" in response_text:
        return response_text.split("Expanded query:")[-1].strip().strip('"')
    return response_text.strip().strip('"')

def llm_individual_rerank(query: str, doc: dict) -> float:
    prompt = f"""Rate how well this movie matches the search query.
                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("description", "")}
                Rate 0-10 (10 = perfect match).
                Give me ONLY the number.
                Score:"""
    response_text = generate_content_with_fallback(prompt)
    try:
        return float(response_text.split()[0])
    except (ValueError, IndexError):
        print(f"[WARN] Could not parse score '{response_text}', defaulting to 0.0")
        return 0.0

def llm_batch_rerank(query: str, results: list[dict]) -> list[int]:
    doc_list_str = "\n".join(
        f"{item['doc_id']}: {item['metadata'].get('title','')} - {item['metadata'].get('description','')}"
        for item in results
    )
    prompt = f"""Rank these movies by relevance to the search query.
                Query: "{query}"
                Movies:
                {doc_list_str}
                Return ONLY the IDs in order of relevance (best match first).
                Return a valid JSON list, nothing else."""
    response_text = generate_content_with_fallback(prompt)
    try:
        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse batch rerank JSON. Returning original order.")
        return [item['doc_id'] for item in results]

def llm_evaluate_results(query: str, results: list[dict]) -> list[int]:
    """
    Evaluates search results using an LLM on a 0-3 scale.
    3: Highly relevant, 2: Relevant, 1: Marginally relevant, 0: Not relevant.
    """
    formatted_results = [
        f"{i+1}. {item['metadata']['title']}: {item['metadata']['description']}"
        for i, item in enumerate(results)
    ]
    
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(formatted_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers out than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""

    response_text = generate_content_with_fallback(prompt)
    
    try:
        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to parse evaluation JSON. Response was: {response_text}")
        return [0] * len(results)