import io

from PIL import Image
from fastapi import File, HTTPException, UploadFile

from .schemas import CaptionResponse, MultiCaptionResponse


def register_routes(app, captioner):
    """wire up image captioning endpoints."""

    @app.get("/health")
    async def health():
        return captioner.health()

    @app.post("/caption", response_model=CaptionResponse)
    async def caption_image(file: UploadFile = File(...)):
        if not captioner.is_loaded:
            raise HTTPException(503, "model not loaded")

        content = await file.read()
        try:
            image = Image.open(io.BytesIO(content))
        except Exception:
            raise HTTPException(400, "invalid image file")

        captions, ms = captioner.caption(image)
        return CaptionResponse(
            caption=captions[0],
            inference_ms=ms,
            model=captioner.config.model_id,
            device=str(captioner.device),
            image_size=list(image.size),
        )

    @app.post("/caption/multi", response_model=MultiCaptionResponse)
    async def caption_multi(file: UploadFile = File(...), num_captions: int = 3):
        if not captioner.is_loaded:
            raise HTTPException(503, "model not loaded")

        content = await file.read()
        try:
            image = Image.open(io.BytesIO(content))
        except Exception:
            raise HTTPException(400, "invalid image file")

        captions, ms = captioner.caption(image, num_captions)
        return MultiCaptionResponse(
            captions=captions,
            count=len(captions),
            inference_ms=ms,
            image_size=list(image.size),
        )

    @app.post("/caption/url")
    async def caption_from_url(body: dict):
        import httpx

        url = body.get("url", "")
        if not url:
            raise HTTPException(400, "URL required")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=captioner.config.url_fetch_timeout)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content))
        except Exception as e:
            raise HTTPException(400, f"failed to fetch image: {e}")

        captions, ms = captioner.caption(image)
        return {
            "caption": captions[0],
            "url": url,
            "inference_ms": ms,
            "image_size": list(image.size),
        }
