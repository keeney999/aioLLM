from typing import List, Literal, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    messages: List[Message]
    provider: str = "openai"  # openai, anthropic, ollama
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: bool = False


class CompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    usage: dict
    model: str
