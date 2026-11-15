"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, experiments, health
from app.core.config import settings
from app.core.database import close_pool
from app.db.init_db import ensure_database_exists, init_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting KNN classification platform backend")
    ensure_database_exists()
    init_schema()
    yield
    close_pool()
    logger.info("Backend shut down")


app = FastAPI(
    title="PostgreSQL KNN Classification Platform",
    description=(
        "Interactive platform for K-Nearest Neighbors classification using "
        "PostgreSQL + pgvector, comparing exact and HNSW approximate search."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(datasets.router, prefix=settings.api_prefix)
app.include_router(experiments.router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict:
    return {
        "name": "KNN Classification Platform API",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
