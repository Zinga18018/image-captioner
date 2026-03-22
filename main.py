"""
Neural Image Captioner
======================
Vision-Language model that generates natural language descriptions
of images. Upload a photo, get a caption -- uses a ViT encoder
paired with a GPT-2 decoder from HuggingFace.

Usage:
    python main.py
    # Then open http://localhost:8004/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import ImageCaptioner, CaptionConfig
from api import register_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

config = CaptionConfig()
captioner = ImageCaptioner(config)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    captioner.load()
    yield


app = FastAPI(
    title="Neural Image Captioner",
    description="ViT + GPT-2 vision-language image captioning",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app, captioner)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.port, reload=True)
