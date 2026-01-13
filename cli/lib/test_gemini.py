import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.5-flash", contents="What is date today?"
)
print(response.text)
print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}, Response Tokens: {response.usage_metadata.candidates_token_count}")