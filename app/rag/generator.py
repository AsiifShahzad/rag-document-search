import os
import requests


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate_answer(prompt: str) -> str:

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)

    response.raise_for_status()

    data = response.json()

    answer = data["choices"][0]["message"]["content"]

    return answer