from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import SecretStr

from src.llm.core.config import OpenAIConfig
from src.llm.providers.openai import OpenAIClient
from src.llm.schemas.requests import CompletionRequest, CompletionResponse

router = APIRouter(prefix="/v1", tags=["llm"])


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    """Эндпоинт для генерации текста через OpenAI."""

    # Загружаем конфиг из переменных окружения
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    config = OpenAIConfig(api_key=SecretStr(api_key), model="gpt-3.5-turbo")
    async with OpenAIClient(config) as client:
        try:
            response = await client.generate(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            return response
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
