from __future__ import annotations

import json
from typing import Any

import requests


class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict[str, Any], timeout_s: int):
        url = f"{self.base_url}{path}"
        return requests.post(url, json=payload, timeout=timeout_s)

    def _build_chat_payload(self, system: str, user: str, temperature: float):
        return {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature},
        }

    def chat(self, system: str, user: str, temperature: float = 0.0, timeout_s: int = 600) -> str:
        payload = self._build_chat_payload(system, user, temperature)

        try:
            response = self._post("/api/chat", payload, timeout_s)
            response.raise_for_status()
        except requests.HTTPError as e:
            body = (e.response.text if e.response is not None else "")[:400]
            status = e.response.status_code if e.response is not None else "error"
            raise RuntimeError(f"Ollama /api/chat HTTP {status}: {body}") from e
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama /api/chat request failed: {e}") from e

        try:
            data = response.json()
        except Exception:
            raise RuntimeError(f"Ollama /api/chat returned non-JSON: {response.text[:1000]!r}")

        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            return data["message"]["content"]

        if "response" in data and isinstance(data["response"], str):
            return data["response"]

        raise RuntimeError(f"Unexpected /api/chat response shape: {json.dumps(data)[:200]}")
