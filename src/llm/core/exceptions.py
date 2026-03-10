class LLMError(Exception):
    """Базовое исключение для ошибок LLM."""

    pass


class LLMConnectionError(LLMError):
    """Ошибка соединения с провайдером."""

    pass


class LLMResponseError(LLMError):
    """Ошибка при обработке ответа."""

    pass
