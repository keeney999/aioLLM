from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, Optional, Type

from loguru import logger

from src.llm.core.config import LLMConfig
from src.llm.schemas.requests import CompletionResponse, Message


class BaseLLMClient(ABC):
    """Абстрактный клиент для взаимодействия с LLM провайдерами."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: Optional[Any] = None
        logger.info(
            f"Initializing {self.__class__.__name__} with config: {config.model_dump(exclude={'api_key'})}"
        )

    @abstractmethod
    async def _setup_client(self) -> None:
        """Инициализация HTTP клиента или SDK."""
        pass

    @abstractmethod
    async def _teardown_client(self) -> None:
        """Закрытие клиента."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """Основной метод для генерации текста."""
        pass

    async def __aenter__(self) -> "BaseLLMClient":
        await self._setup_client()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self._teardown_client()
