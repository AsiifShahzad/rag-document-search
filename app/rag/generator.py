import os
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Fallback order: best to worst
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it"
]

def generate_answer(prompt: str) -> str:
    print(f"\n{'='*60}")
    print(f"[GENERATOR] Generating answer from LLM")
    print(f"{'='*60}")
    print(f"\n[GENERATOR] Prompt length: {len(prompt)} characters")
    print(f"[GENERATOR] Prompt preview (first 300 chars):\n{prompt[:300]}...")
    print(f"\n[GENERATOR] Full prompt:\n{prompt}")

    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty — cannot send to Groq")

    # Increased limit to 15000 to avoid cutting off questions
    if len(prompt) > 15000:
        print(f"[GENERATOR] WARNING: Prompt exceeds 15000 chars, truncating to avoid API limits...")
        prompt = prompt[:15000]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    last_error = None

    for model in GROQ_MODELS:
        print(f"\n[GENERATOR] Attempting model: {model}")
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
            print(f"[GENERATOR] ✓ Model {model} succeeded")
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            print(f"[GENERATOR] Answer: {answer[:200]}...")
            print(f"{'='*60}\n")
            return answer

        error = response.json().get("error", {})
        print(f"[GENERATOR] ✗ Model {model} failed: {error.get('message', response.status_code)}")
        last_error = response

    print(f"[GENERATOR] All models failed. Last error:")
    print(f"[GENERATOR] Status: {last_error.status_code}")
    print(f"[GENERATOR] Response: {last_error.text}")
    last_error.raise_for_status()