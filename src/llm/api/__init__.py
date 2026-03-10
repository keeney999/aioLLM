from fastapi import FastAPI

from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Toolkit API",
        description="Unified API for various LLM providers",
        version="0.1.0",
    )

    app.include_router(router)

    @app.get("/health")
    async def health() -> FastAPI.Response:
        return {"status": "ok"}

    return app
