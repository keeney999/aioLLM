from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from src.llm.core.config import OpenAIConfig
from src.llm.providers.openai import OpenAIClient
from src.llm.schemas.requests import Message


@pytest.fixture
def mock_openai_response():
    """Создаёт мок, имитирующий ответ OpenAI."""
    # Создаём объект choice с методом model_dump
    mock_choice = MagicMock()
    mock_choice.model_dump.return_value = {
        "message": {"role": "assistant", "content": "Hello"}
    }
    # Создаём объект usage с методом model_dump
    mock_usage = MagicMock()
    mock_usage.model_dump.return_value = {"total_tokens": 10}
    # Создаём сам ответ
    mock_response = AsyncMock()
    mock_response.id = "test-id"
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "gpt-3.5-turbo"
    return mock_response


@pytest.mark.asyncio
async def test_openai_generate_success(mock_openai_response):
    config = OpenAIConfig(api_key=SecretStr("test-key"))
    client = OpenAIClient(config)

    with patch.object(client, "_client", AsyncMock()) as mock_client:
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        messages = [Message(role="user", content="Hi")]
        response = await client.generate(messages)

        assert response.id == "test-id"
        assert response.choices[0]["message"]["content"] == "Hello"
        assert response.usage["total_tokens"] == 10
        assert response.model == "gpt-3.5-turbo"
