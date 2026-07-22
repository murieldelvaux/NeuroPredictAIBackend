from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.routers import patients, prediction
from app.services.ai_service import model_service
from app.db.in_memory import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    model_service.load()
    yield
    model_service.unload()


app = FastAPI(
    title="NeuroPredict AI",
    description="Backend para predição de risco de Alzheimer via MRI 3D",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router, prefix="/patients", tags=["patients"])
app.include_router(prediction.router, prefix="/predict", tags=["prediction"])


@app.get("/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded,
    }
