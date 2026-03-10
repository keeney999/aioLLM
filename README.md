# aioLLM

[![CI](https://github.com/твой-username/aioLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/твой-username/aioLLM/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**aioLLM** — асинхронный Python-инструментарий для работы с большими языковыми моделями. Поддерживает OpenAI, Anthropic, локальные модели через Ollama и других провайдеров. Проект включает готовый FastAPI сервер, модульные тесты, CI и pre-commit хуки.

## 🚀 Возможности

- ⚡ Асинхронные клиенты на базе `httpx` и `openai`.
- 🔌 Единый интерфейс для разных провайдеров.
- 🧩 Лёгкое добавление новых провайдеров.
- 🚀 FastAPI сервер с эндпоинтами `/v1/completions` и `/health`.
- 📦 Готов к продакшену: тесты, линтеры, CI/CD.
- 📚 Примеры использования.

## 📦 Установка

git clone https://github.com/keeney999/aioLLM.git
cd aioLLM
poetry install

🔧 Конфигурация
Скопируйте .env.example в .env и укажите API ключи:

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
🧪 Использование в коде
OpenAI
python
import asyncio
from src.llm.providers.openai import OpenAIClient
from src.llm.core.config import OpenAIConfig
from src.llm.schemas.requests import Message

async def main():
    config = OpenAIConfig(api_key="sk-...")
    async with OpenAIClient(config) as client:
        resp = await client.generate([Message(role="user", content="Hello")])
        print(resp.choices[0]['message']['content'])

asyncio.run(main())
Ollama (локально)
python
from src.llm.providers.ollama import OllamaClient
from src.llm.core.config import OllamaConfig

config = OllamaConfig(base_url="http://localhost:11434", model="llama2")
async with OllamaClient(config) as client:
    ...
🌐 FastAPI сервер
Запустите сервер:

bash
uvicorn src.llm.api:app --reload
Отправьте запрос:

bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Привет"}],
    "temperature": 0.7
  }'
🧪 Тестирование
bash
poetry run pytest tests/ -v --cov=src.llm
