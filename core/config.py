from dataclasses import dataclass


@dataclass
class CaptionConfig:
    """settings for the image captioning model."""

    model_id: str = "nlpconnect/vit-gpt2-image-captioning"
    max_caption_length: int = 64
    default_num_beams: int = 4
    max_captions: int = 5
    url_fetch_timeout: int = 10     # seconds
    port: int = 8004
