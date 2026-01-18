import argparse
import mimetypes
from google.genai import types
from google import genai
import os
from lib.search_utils import api_key, DATA_DIR

def main():
    parser = argparse.ArgumentParser(description="Query Rewritten using Image")
    parser.add_argument("--image", type=str, help="Path to an image file")
    parser.add_argument("--query", type=str, help="A Text query to rewrite based on the image")

    args = parser.parse_args()
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    
    IMAGE_PATH = os.path.join(DATA_DIR, args.image)

    with open(IMAGE_PATH, "rb") as f:
        img = f.read()

    client = genai.Client(api_key=api_key)

    system_prompt = f""" Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
                            - Synthesize visual and textual information
                            - Focus on movie-specific details (actors, scenes, style, etc.)
                            - Return only the rewritten query, without any additional commentary
                       """
    
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    model_name = "gemini-2.5-flash"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=parts
        )
        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    except Exception as e:
       print(f"[ERROR] Unexpected failure with {model_name}: {e}. Trying fallback...")
            
if __name__ == "__main__":
    main()

    



