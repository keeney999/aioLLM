from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from src.llm.core.config import AnthropicConfig
from src.llm.providers.anthropic import AnthropicClient
from src.llm.schemas.requests import Message


@pytest.fixture
def mock_anthropic_response():
    """Создаёт мок, имитирующий ответ Anthropic."""
    # Создаём объект content block с текстом
    content_block = MagicMock()
    content_block.text = "Hello from Claude"
    # Создаём сам ответ
    mock_response = AsyncMock()
    mock_response.id = "msg_123"
    mock_response.content = [content_block]
    mock_response.model = "claude-3-haiku-20240307"
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    return mock_response


@pytest.mark.asyncio
async def test_anthropic_generate_success(mock_anthropic_response):
    config = AnthropicConfig(api_key=SecretStr("test-key"))
    client = AnthropicClient(config)

    with patch.object(client, "_client", AsyncMock()) as mock_client:
        mock_client.messages.create = AsyncMock(return_value=mock_anthropic_response)

        messages = [Message(role="user", content="Hello")]
        response = await client.generate(messages)

        assert response.id == "msg_123"
        assert response.choices[0]["message"]["content"] == "Hello from Claude"
        assert response.model == "claude-3-haiku-20240307"
        assert response.usage["input_tokens"] == 10
        assert response.usage["output_tokens"] == 20
