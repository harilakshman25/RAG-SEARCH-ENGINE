import json
import os
from dotenv import load_dotenv
from google import genai

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

def enhance_query(query: str, method : str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{query}"

                If no errors, return the original query.
                Corrected:"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    enhaned_query = response.text.strip()[len("Corrected:"):].strip().strip('"')
    if query.strip() != enhaned_query:
        print(f"Enhanced query ({method}): '{query}' -> '{enhaned_query}'\n")

    return enhaned_query