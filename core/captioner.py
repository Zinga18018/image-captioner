import time
import logging

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from .config import CaptionConfig

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """generates natural language descriptions of images using ViT + GPT-2.

    the ViT encoder converts the image into a sequence of patch embeddings,
    which GPT-2 then decodes into a caption. beam search produces multiple
    candidate captions ranked by likelihood.
    """

    def __init__(self, config: CaptionConfig | None = None):
        self.config = config or CaptionConfig()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None

    def load(self):
        """pull model weights and move to the best available device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("loading %s on %s", self.config.model_id, self.device)

        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.model_id)
        self.processor = ViTImageProcessor.from_pretrained(self.config.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        self.model.to(self.device)
        self.model.eval()
        logger.info("captioner ready on %s", self.device)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ---- image preprocessing ----

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        """convert a PIL image to model-ready pixel values."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.processor(
            images=[image], return_tensors="pt",
        ).pixel_values.to(self.device)

    # ---- caption generation ----

    def caption(self, image: Image.Image, num_captions: int = 1) -> tuple[list[str], float]:
        """generate one or more captions for an image.

        returns (list_of_captions, inference_time_ms).
        """
        pixel_values = self._prepare(image)
        num_captions = min(num_captions, self.config.max_captions)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=self.config.max_caption_length,
                num_beams=max(num_captions, self.config.default_num_beams),
                num_return_sequences=num_captions,
                use_cache=True,
            )
        elapsed = (time.perf_counter() - start) * 1000

        captions = [
            c.strip()
            for c in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ]
        return captions, round(elapsed, 1)

    # ---- health check ----

    def health(self) -> dict:
        return {
            "status": "healthy" if self.is_loaded else "loading",
            "model": self.config.model_id,
            "device": str(self.device),
            "architecture": "ViT (encoder) + GPT-2 (decoder)",
        }
