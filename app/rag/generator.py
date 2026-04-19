import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Validate API key at module load time
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is not set!")
else:
    logger.info(f"GROQ_API_KEY loaded (length: {len(GROQ_API_KEY)} chars)")

# Fallback order: best to worst
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it"
]

def generate_answer(prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty — cannot send to Groq")
    
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Please check your environment variables.")

    if len(prompt) > 15000:
        prompt = prompt[:15000]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    last_error = None

    for model in GROQ_MODELS:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided document context. Answer only using the context given. If the answer is not in the context, say so clearly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "max_tokens": 1024
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload)

        if response.ok:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return answer

        last_error = response

    last_error.raise_for_status()