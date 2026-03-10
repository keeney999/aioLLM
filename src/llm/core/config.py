from typing import Optional

from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    """Базовая конфигурация для любого LLM провайдера."""

    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)


class OpenAIConfig(LLMConfig):
    """Специфичная конфигурация для OpenAI."""

    model: str = "gpt-3.5-turbo"
    organization: Optional[str] = None


class AnthropicConfig(LLMConfig):
    model: str = "claude-3-haiku-20240307"
