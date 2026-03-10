from typing import List, Optional

import openai
from loguru import logger
from openai import AsyncOpenAI

from src.llm.core.client import BaseLLMClient
from src.llm.core.config import OpenAIConfig
from src.llm.core.exceptions import LLMConnectionError, LLMResponseError
from src.llm.schemas.requests import CompletionResponse, Message


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)
        self.config: OpenAIConfig = config

    async def _setup_client(self) -> None:
        try:
            self._client = AsyncOpenAI(
                api_key=(
                    self.config.api_key.get_secret_value()
                    if self.config.api_key
                    else None
                ),
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            logger.debug("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to setup OpenAI client: {e}")
            raise LLMConnectionError(f"OpenAI client setup failed: {e}")

    async def _teardown_client(self) -> None:
        if self._client:
            await self._client.close()
            logger.debug("OpenAI client closed")

    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        if not self._client:
            raise LLMConnectionError(
                "Client not initialized. Call _setup_client() first."
            )

        openai_messages = [msg.model_dump() for msg in messages]

        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=openai_messages,
                temperature=(
                    temperature or self.config.temperature
                    if hasattr(self.config, "temperature")
                    else 0.7
                ),
                max_tokens=(
                    max_tokens or self.config.max_tokens
                    if hasattr(self.config, "max_tokens")
                    else 1000
                ),
                **kwargs,
            )
            logger.debug(f"Received response from OpenAI: {response.id}")

            return CompletionResponse(
                id=response.id,
                choices=[choice.model_dump() for choice in response.choices],
                usage=response.usage.model_dump(),
                model=response.model,
            )
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise LLMConnectionError(f"OpenAI connection failed: {e}")
        except Exception as e:
            logger.error(f"OpenAI response error: {e}")
            raise LLMResponseError(f"OpenAI response processing failed: {e}")
