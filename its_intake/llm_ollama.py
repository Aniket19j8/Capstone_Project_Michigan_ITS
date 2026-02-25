from __future__ import annotations

import requests


class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, system: str, user: str, temperature: float = 0.2, timeout_s: int = 600) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature
            },
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]