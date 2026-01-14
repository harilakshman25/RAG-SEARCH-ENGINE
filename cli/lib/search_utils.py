import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError, ServerError

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

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = [line.strip() for line in f]
    return stopwords

client = genai.Client(api_key=api_key)

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

    except (APIError, ServerError) as e:
        print(
            f"[WARN] Query enhancement ({method}) failed: "
            f"{e.__class__.__name__} – falling back to original query.\n"
        )
        return query

    except Exception as e:
        print(
            f"[WARN] Unexpected error during query enhancement: "
            f"{e.__class__.__name__} – falling back to original query.\n"
        )
        return query

def spelling_corrector(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{query}"

                If no errors, return the original query.
                Corrected:"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    # print(response.text) #debug
    spell_corrected_query = response.text.strip()[len("Corrected:"):].strip().strip('"')
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
    
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    # print(response.text) #debug
    rewritten_query = response.text.strip()[len("Rewritten query:"):].strip().strip('"')
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
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    # print(response.text) #debug
    expanded_query = response.text.strip().strip().strip('"')
    return expanded_query