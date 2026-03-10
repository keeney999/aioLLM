from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_openai_response() -> AsyncMock:
    mock = AsyncMock()
    mock.id = "test-id"
    mock.choices = [{"message": {"role": "assistant", "content": "Hello"}}]
    mock.usage = {"total_tokens": 10}
    mock.model = "gpt-3.5-turbo"
    return mock
