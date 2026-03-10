from typing import List, Optional

import anthropic
from anthropic import AsyncAnthropic
from loguru import logger

from src.llm.core.client import BaseLLMClient
from src.llm.core.config import AnthropicConfig
from src.llm.core.exceptions import LLMConnectionError, LLMResponseError
from src.llm.schemas.requests import CompletionResponse, Message


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: AnthropicConfig) -> None:
        super().__init__(config)
        self.config: AnthropicConfig = config

    async def _setup_client(self) -> None:
        try:
            self._client = AsyncAnthropic(
                api_key=(
                    self.config.api_key.get_secret_value()
                    if self.config.api_key
                    else None
                ),
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            logger.debug("Anthropic client initialized")
        except Exception as e:
            logger.error(f"Failed to setup Anthropic client: {e}")
            raise LLMConnectionError(f"Anthropic client setup failed: {e}")

    async def _teardown_client(self) -> None:
        if self._client:
            await self._client.close()
            logger.debug("Anthropic client closed")

    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        if not self._client:
            raise LLMConnectionError("Client not initialized")

        # Преобразуем сообщения в формат Anthropic
        anthropic_messages = []
        system_prompt = None
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        try:
            response = await self._client.messages.create(
                model=self.config.model,
                messages=anthropic_messages,
                system=system_prompt,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                **kwargs,
            )
            logger.debug(f"Received response from Anthropic: {response.id}")

            # Преобразуем в унифицированную модель
            return CompletionResponse(
                id=response.id,
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": response.content[0].text,
                        }
                    }
                ],
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                model=response.model,
            )
        except anthropic.APIConnectionError as e:
            logger.error(f"Anthropic connection error: {e}")
            raise LLMConnectionError(f"Anthropic connection failed: {e}")
        except Exception as e:
            logger.error(f"Anthropic response error: {e}")
            raise LLMResponseError(f"Anthropic response processing failed: {e}")
