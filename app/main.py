from fastapi import FastAPI

from .routers.chat import router as chat_router
from .db import init_db
from .routers.auth import router as auth_router
from .routers.health import router as health_router


def create_app() -> FastAPI:
    application = FastAPI(title="Health Assistant Backend")
    application.include_router(auth_router)
    application.include_router(health_router)
    application.include_router(chat_router)

    @application.on_event("startup")
    def startup_init_db():
        init_db()

    return application


app = create_app()
