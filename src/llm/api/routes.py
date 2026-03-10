import os

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import SecretStr

from src.llm.core.config import AnthropicConfig, OllamaConfig, OpenAIConfig
from src.llm.providers.anthropic import AnthropicClient
from src.llm.providers.ollama import OllamaClient
from src.llm.providers.openai import OpenAIClient
from src.llm.schemas.requests import CompletionRequest, CompletionResponse

router = APIRouter(prefix="/v1", tags=["llm"])


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    """Эндпоинт для генерации текста с выбором провайдера."""
    try:
        if request.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
            config = OpenAIConfig(api_key=SecretStr(api_key), model="gpt-3.5-turbo")
            async with OpenAIClient(config) as client:
                return await client.generate(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

        elif request.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
            config = AnthropicConfig(api_key=SecretStr(api_key))
            async with AnthropicClient(config) as client:
                return await client.generate(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

        elif request.provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            config = OllamaConfig(base_url=base_url)
            async with OllamaClient(config) as client:
                return await client.generate(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider: {request.provider}. Supported: openai, anthropic, ollama",
            )

    except Exception as e:
        logger.error(f"Error in /completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
