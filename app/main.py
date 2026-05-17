import logging

from fastapi import FastAPI

from .routers.chat import router as chat_router
from .db import init_db
from .routers.auth import router as auth_router
from .routers.health import router as health_router
from .services.ai import warm_up_vectorstore


logger = logging.getLogger("uvicorn.error")


def create_app() -> FastAPI:
    application = FastAPI(title="Health Assistant Backend")
    application.include_router(auth_router)
    application.include_router(health_router)
    application.include_router(chat_router)

    @application.on_event("startup")
    def startup_init_db():
        init_db()
        _warm_up_vectorstore()

    return application


def _warm_up_vectorstore():
    logger.info("starting vectorstore warm-up before accepting requests")
    if warm_up_vectorstore():
        logger.info("vectorstore warm-up finished; server is ready for report requests")
    else:
        logger.warning("vectorstore warm-up skipped or failed; report requests may build it lazily")


app = create_app()
