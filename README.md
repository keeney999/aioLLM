# aioLLM

[![CI](https://github.com/keeney999/aioLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/keeney999/aioLLM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/keeney999/aioLLM/branch/main/graph/badge.svg)](https://codecov.io/gh/keeney999/aioLLM)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**aioLLM** — асинхронный Python-инструментарий для работы с большими языковыми моделями (LLM).
Проект предоставляет единый интерфейс для популярных провайдеров (OpenAI, Anthropic, локальные модели через Ollama) и готовый FastAPI-сервер для быстрого развёртывания.

---

## 📚 Содержание
- [Возможности](#-возможности)
- [Установка](#-установка)
- [Конфигурация](#-конфигурация)
- [Использование](#-использование)
  - [OpenAI](#openai)
  - [Anthropic (Claude)](#anthropic-claude)
  - [Ollama (локально)](#ollama-локально)
- [FastAPI сервер](#-fastapi-сервер)
- [Тестирование](#-тестирование)
- [Makefile команды](#-makefile-команды)
- [Структура проекта](#-структура-проекта)
- [Как добавить нового провайдера](#-как-добавить-нового-провайдера)
- [Лицензия](#-лицензия)

---

## 🚀 Возможности

- ⚡ **Полностью асинхронный** — на базе `asyncio`, `httpx` и официальных SDK.
- 🔌 **Единый интерфейс** — одинаковые методы для всех провайдеров.
- 🧩 **Лёгкое расширение** — добавление нового провайдера за несколько минут.
- 🚀 **FastAPI сервер** — готовые эндпоинты `/v1/completions` и `/health` с поддержкой выбора провайдера.
- 📦 **Готов к продакшену** — тесты с покрытием, линтеры (black, isort, flake8), pre-commit хуки, CI/CD (GitHub Actions).

---

## 📦 Установка

```bash
# Клонируем репозиторий
git clone https://github.com/keeney999/aioLLM.git
cd aioLLM

# Устанавливаем зависимости через poetry
poetry install
💡 Если у вас нет Poetry, установите его по инструкции.

🔧 Конфигурация
Скопируйте файл .env.example в .env и укажите свои API-ключи:

bash
cp .env.example .env
Пример содержимого .env:

ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
🧪 Использование
OpenAI
python
import asyncio
from src.llm.providers.openai import OpenAIClient
from src.llm.core.config import OpenAIConfig
from src.llm.schemas.requests import Message

async def main():
    config = OpenAIConfig(api_key="sk-...")  # или загрузите из .env
    async with OpenAIClient(config) as client:
        messages = [Message(role="user", content="Напиши короткое приветствие")]
        response = await client.generate(messages)
        print(response.choices[0]['message']['content'])

asyncio.run(main())
Anthropic (Claude)
python
import asyncio
from src.llm.providers.anthropic import AnthropicClient
from src.llm.core.config import AnthropicConfig
from src.llm.schemas.requests import Message

async def main():
    config = AnthropicConfig(api_key="sk-ant-...")
    async with AnthropicClient(config) as client:
        messages = [Message(role="user", content="Расскажи анекдот")]
        response = await client.generate(messages)
        print(response.choices[0]['message']['content'])

asyncio.run(main())
Ollama (локально)
python
import asyncio
from src.llm.providers.ollama import OllamaClient
from src.llm.core.config import OllamaConfig
from src.llm.schemas.requests import Message

async def main():
    config = OllamaConfig(base_url="http://localhost:11434", model="llama2")
    async with OllamaClient(config) as client:
        messages = [Message(role="user", content="Привет! Как дела?")]
        response = await client.generate(messages)
        print(response.choices[0]['message']['content'])

asyncio.run(main())

🌐 FastAPI сервер
Запустите сервер (автоматически подгрузит переменные из .env):

bash
make run
# или напрямую:
poetry run uvicorn src.llm.api:app --reload
Сервер будет доступен по адресу http://localhost:8000.

Эндпоинты
GET /health — проверка работоспособности.

POST /v1/completions — генерация текста. Тело запроса:

json
{
  "provider": "openai",      // openai, anthropic, ollama
  "messages": [{"role": "user", "content": "Привет"}],
  "temperature": 0.7,
  "max_tokens": 100
}
Пример запроса через curl:

bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "messages": [{"role": "user", "content": "Как погода?"}]
  }'
🧪 Тестирование
Запуск всех тестов с отчётом о покрытии:

bash
make test
# или
poetry run pytest tests/ -v --cov=src.llm
🛠 Makefile команды
Команда	Действие
make install	Установка зависимостей (poetry install)
make test	Запуск тестов с покрытием
make lint	Запуск всех линтеров (pre-commit)
make run	Запуск FastAPI сервера
make clean	Очистка кэша и временных файлов

➕ Как добавить нового провайдера
Создайте класс в src/llm/providers/, унаследовавшись от BaseLLMClient.

Реализуйте методы _setup_client, _teardown_client и generate.

Добавьте конфигурацию в core/config.py.

Напишите тесты в tests/unit/ (с моками).

Добавьте провайдер в роутер api/routes.py (опционально).


📄 Лицензия
Проект распространяется под лицензией MIT. Подробнее — в файле LICENSE.

aioLLM создан с ❤️ для сообщества. Если проект оказался полезным, поставьте звезду ⭐ на GitHub!
