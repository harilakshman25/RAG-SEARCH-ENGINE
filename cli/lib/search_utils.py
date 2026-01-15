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
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Ordered list of 10 models to cycle through if limits are hit
FALLBACK_MODELS = [
    "gemini-2.5-flash",          # Priority 1
    "gemini-2.5-flash-lite",     # Priority 2
    "gemini-3-flash-preview",    # Priority 3
    "gemini-2.0-flash",          # Priority 4
    "gemini-2.0-flash-lite",     # Priority 5
    "gemini-flash-latest",       # Priority 6
    "gemini-flash-lite-latest",  # Priority 7
    "gemini-3-pro-preview",      # Priority 8
    "gemini-2.0-flash-001",      # Priority 9
    "gemma-3-27b-it"             # Priority 10 (Open Model)
]

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = [line.strip() for line in f]
    return stopwords

client = genai.Client(api_key=api_key)

def generate_content_with_fallback(prompt: str) -> str:
    """
    Attempts to generate content by rotating through models if 
    Quota or Rate Limits (429) are encountered.
    """
    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt
            )
            return response.text.strip()
        
        except ClientError as e:
            # Check for 429 Resource Exhausted (Daily limit or RPM)
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper():
                print(f"[LIMIT] {model_name} exhausted. Trying next model...")
                continue
            print(f"[ERROR] Client error with {model_name}: {e}")
            raise e
            
        except (APIError, ServerError) as e:
            print(f"[SERVER ERROR] {model_name} failed: {e}. Trying fallback...")
            continue
            
    # If the loop finishes without returning, all models failed
    raise Exception("CRITICAL: All available Gemini models have reached their limits.")

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
            print(
                f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n"
            )

        return enhanced_query

    except Exception as e:
        print(
            f"[WARN] Query enhancement ({method}) failed: "
            f"{e.__class__.__name__}: {str(e)[:100]} – falling back to original query.\n"
        )
        return query

def spelling_corrector(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{query}"

                If no errors, return the original query.
                Corrected:"""
    
    response_text = generate_content_with_fallback(prompt)
    
    if "Corrected:" in response_text:
        spell_corrected_query = response_text.split("Corrected:")[-1].strip().strip('"')
    else:
        spell_corrected_query = response_text.strip().strip('"')
        
    return spell_corrected_query

def rewriter(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

                Original: "{query}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                Rewritten query:"""
    
    response_text = generate_content_with_fallback(prompt)
    
    if "Rewritten query:" in response_text:
        rewritten_query = response_text.split("Rewritten query:")[-1].strip().strip('"')
    else:
        rewritten_query = response_text.strip().strip('"')
        
    return rewritten_query

def expander(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                This will be appended to the original query.

                Examples:

                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                Query: "{query}"
                Expanded query:
                """
    response_text = generate_content_with_fallback(prompt)
    
    if "Expanded query:" in response_text:
        expanded_query = response_text.split("Expanded query:")[-1].strip().strip('"')
    else:
        expanded_query = response_text.strip().strip('"')
        
    return expanded_query

def llm_individual_rerank(query: str, doc: dict) -> float:
    prompt = f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("description", "")}

                Rate 0-10 (10 = perfect match).
                Give me ONLY the number.

                Score:"""
    
    response_text = generate_content_with_fallback(prompt)
    
    # Extract only the number from response
    try:
        # Split in case the model adds extra text
        score = float(response_text.split()[0])
        return score
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
                Return a valid JSON list, nothing else.
                """

    response_text = generate_content_with_fallback(prompt)
    
    try:
        # Handle cases where the model wraps JSON in markdown blocks
        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse batch rerank JSON. Returning original order.")
        return [item['doc_id'] for item in results]