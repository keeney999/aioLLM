from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.core.config import OllamaConfig
from src.llm.providers.ollama import OllamaClient
from src.llm.schemas.requests import Message


@pytest.mark.asyncio
async def test_ollama_generate_success():
    config = OllamaConfig(base_url="http://localhost:11434", model="llama2")
    client = OllamaClient(config)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={
            "model": "llama2",
            "created_at": "2023-12-01T12:00:00Z",
            "response": "Привет! Чем могу помочь?",
            "total_duration": 1234567890,
        }
    )
    mock_response.raise_for_status = MagicMock()

    with patch.object(client, "_client", AsyncMock()) as mock_httpx:
        mock_httpx.post.return_value = mock_response

        messages = [Message(role="user", content="Привет")]
        response = await client.generate(messages)

        assert response.id.startswith("ollama-")
        assert response.choices[0]["message"]["content"] == "Привет! Чем могу помочь?"
        assert response.model == "llama2"
        assert response.usage["total_tokens"] == 1234  # 1234567890 // 1_000_000 ≈ 1234
