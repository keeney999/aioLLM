from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from src.llm.core.client import BaseLLMClient
from src.llm.core.config import OllamaConfig
from src.llm.core.exceptions import LLMConnectionError, LLMResponseError
from src.llm.schemas.requests import CompletionResponse, Message


class OllamaClient(BaseLLMClient):
    def __init__(self, config: OllamaConfig) -> None:
        super().__init__(config)
        self.config: OllamaConfig = config

    async def _setup_client(self) -> None:
        """Инициализация HTTP клиента для Ollama."""
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            limits=httpx.Limits(max_keepalive_connections=5),
        )
        logger.debug("Ollama HTTP client initialized")

    async def _teardown_client(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.debug("Ollama HTTP client closed")

    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        if not self._client:
            raise LLMConnectionError("Client not initialized")

        # Формируем промпт из сообщений (Ollama ожидает строку)
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Ollama response received: {data}")

            # Преобразуем ответ в наш формат
            return CompletionResponse(
                id="ollama-" + data.get("created_at", ""),
                choices=[
                    {"message": {"role": "assistant", "content": data["response"]}}
                ],
                usage={
                    "total_tokens": data.get("total_duration", 0) // 1_000_000
                },  # приблизительно
                model=data.get("model", self.config.model),
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise LLMResponseError(f"Ollama request failed: {e}")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise LLMConnectionError(f"Ollama connection failed: {e}")
