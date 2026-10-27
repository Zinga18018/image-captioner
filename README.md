# Image Captioner

Generates captions using the pretrained `nlpconnect/vit-gpt2-image-captioning` model: a ViT image encoder and GPT-2 text decoder. It converts images to RGB and uses beam search to produce one or more candidate descriptions. Captions may omit objects or describe things that are not present.

## Run locally

Use Python 3.10+. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8004
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Open `http://127.0.0.1:8004/docs`. Startup loads the model, image processor and tokenizer, downloading uncached files. CUDA is used when available, otherwise CPU. Requirements are unpinned.

`streamlit run app.py` starts an upload-only interface with its own model instance. URL input is an API feature, not a Streamlit control.

## API

| Method | Path | Behavior |
|---|---|---|
| GET | `/health` | Model and device status |
| POST | `/caption` | One caption from an uploaded image |
| POST | `/caption/multi` | Multiple caption candidates |
| POST | `/caption/url` | Fetch an image URL and caption it |

[core/captioner.py](core/captioner.py) uses at least four beams, caps output at five candidates and sets a maximum caption length of 64 tokens. Use `num_captions` from 1 to 5: only the upper bound is enforced in code. Beam search ranks generated sequences; it does not verify what is in an image.

## Safety and evaluation limits

[api/routes.py](api/routes.py) reads complete uploads without a size cap. The URL endpoint fetches caller-supplied URLs without host/private-network restrictions or a response-size cap, and lacks the upload routes' loaded-model check. Do not expose it to untrusted callers; authentication, URL restrictions and resource limits require separate code work.

The API permits all CORS origins. The [loopback launch](https://uvicorn.dev/settings/) above avoids the all-interface, reload-enabled configuration in `python main.py`, but does not fix these application-level gaps.

No caption-quality evaluation, hardware-specific timing benchmark or test suite is committed. Returned inference times are observations of individual calls, not promised CPU/GPU performance. Do not rely on generated captions for safety-critical identification.

Setup, model execution and browser flows were not rerun for this documentation update.

## Related implementations

Related versions are in [ML-Portfolio-Projects](https://github.com/Zinga18018/ML-Portfolio-Projects): an [API variant](https://github.com/Zinga18018/ML-Portfolio-Projects/tree/master/project4-image-captioner) and a separate [HF Space variant](https://github.com/Zinga18018/ML-Portfolio-Projects/tree/master/hf-spaces/image-captioner). They share the project/model identity but have different code; their screenshots and deployment status are not evidence of a run of this repository.

