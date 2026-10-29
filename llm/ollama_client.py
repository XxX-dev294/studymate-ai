"""
llm/ollama_client.py

Клиент для локального Ollama API. Не обращается к облачным API —
только http://localhost:11434 (или STUDYMATE_OLLAMA_HOST) по умолчанию.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from config import settings


class OllamaUnavailableError(RuntimeError):
    """Ollama не запущена или недоступна по указанному адресу."""


@dataclass
class GenerationResult:
    text: str
    model: str
    done: bool = True


class OllamaClient:
    """Тонкая обёртка над HTTP API Ollama (/api/tags, /api/generate)."""

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.llm_model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                f"Не удалось получить список моделей Ollama: {exc}"
            ) from exc

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> GenerationResult:
        """Синхронный (не стримящий) вызов генерации."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": (
                    temperature if temperature is not None else settings.llm_temperature
                )
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=timeout or settings.llm_timeout_seconds,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                f"Ollama недоступна по адресу {self.host}. "
                f"Запустите 'ollama serve' и убедитесь, что модель "
                f"'{self.model}' загружена ('ollama pull {self.model}'). "
                f"Подробности: {exc}"
            ) from exc

        data = resp.json()
        return GenerationResult(
            text=data.get("response", ""), model=self.model, done=data.get("done", True)
        )
