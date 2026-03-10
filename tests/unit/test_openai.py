from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from src.llm.core.config import OpenAIConfig
from src.llm.providers.openai import OpenAIClient
from src.llm.schemas.requests import Message


@pytest.mark.asyncio
async def test_openai_generate_success(mock_openai_response: AsyncMock) -> None:
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
