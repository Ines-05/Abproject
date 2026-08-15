from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.inference import Detector
from app.routers import detect, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.localizer = Detector(settings.localizer_model_path)
    app.state.classifier = Detector(settings.classifier_model_path)
    yield
    app.state.localizer = None
    app.state.classifier = None


app = FastAPI(title="Shelf Vision Detection API", lifespan=lifespan)
app.include_router(health.router)
app.include_router(detect.router)
