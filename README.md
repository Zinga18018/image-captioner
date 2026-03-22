# Vision-Language Caption Synthesis with ViT + GPT-2

Generates natural language descriptions of images using a Vision Transformer encoder paired with a GPT-2 decoder. Upload an image, get back one or more captions describing what's in it.

## how it works

```
image → ViT encoder → cross-attention → GPT-2 decoder → beam search → caption
```

the model is `nlpconnect/vit-gpt2-image-captioning` — a VisionEncoderDecoder that maps image patches through ViT, then decodes captions token-by-token with GPT-2. beam search (4 beams by default) picks the most probable sequence.

**details:**
- pretrained ViT for visual features + GPT-2 for language generation
- KV cache enabled for faster decoding
- supports multi-caption generation (up to 5 candidates)
- max caption length: 64 tokens

## setup

```bash
pip install -r requirements.txt
python main.py
```

runs at `localhost:8004`. swagger docs at `/docs`.

## api

| endpoint | method | what it does |
|----------|--------|-------------|
| `/health` | GET | model status + device info |
| `/caption` | POST | single caption from uploaded image |
| `/caption/multi` | POST | multiple caption candidates |
| `/caption/url` | POST | caption an image from a URL |

## architecture

```
core/
├── config.py      → model ID, beam params, all settings
└── captioner.py   → ImageCaptioner class, ViT+GPT2 inference

api/
├── schemas.py     → pydantic request/response models
└── routes.py      → endpoint handlers, file/URL handling

main.py            → FastAPI entry point, lifespan
app.py             → streamlit frontend
```

## streamlit demo

```bash
streamlit run app.py
```

upload an image or provide a URL, get captions with inference timing.

## requirements

- python 3.10+
- ~1GB for model weights
- GPU recommended for faster inference (~1.5s on GPU vs ~5s on CPU)
