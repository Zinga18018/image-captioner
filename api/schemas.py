from pydantic import BaseModel, Field


class CaptionResponse(BaseModel):
    caption: str
    inference_ms: float
    model: str
    device: str
    image_size: list[int]


class MultiCaptionResponse(BaseModel):
    captions: list[str]
    count: int
    inference_ms: float
    image_size: list[int]
