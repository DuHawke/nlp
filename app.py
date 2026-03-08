from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import torch
import logging
import os

DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MarianMT Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LANGUAGE_PAIRS = {
    ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
    ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("en", "ko"): "Helsinki-NLP/opus-mt-en-ko",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-pt",
    ("pt", "en"): "Helsinki-NLP/opus-mt-pt-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE}")

_cache: dict = {} 


def _load_model(src: str, tgt: str):
    key = (src, tgt)
    if key in _cache:
        return _cache[key]
    model_name = LANGUAGE_PAIRS.get(key)
    if not model_name:
        raise HTTPException(400, f"Không có model cho cặp '{src}'→'{tgt}'")
    logger.info(f"Loading {model_name} ...")
    try:
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        mdl.eval()
        _cache[key] = (mdl, tok)
        logger.info(f"Loaded ✓  {model_name}")
        return mdl, tok
    except Exception as e:
        raise HTTPException(500, f"Lỗi tải model: {e}")


def _run_model(text: str, src: str, tgt: str) -> str:
    model, tok = _load_model(src, tgt)
    inputs = tok(text, return_tensors="pt",
                 padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def translate_with_pivot(text: str, src: str, tgt: str) -> tuple[str, bool]:
    if src == tgt:
        return text, False

    if (src, tgt) in LANGUAGE_PAIRS:
        return _run_model(text, src, tgt), False

    pivot_ok = (src, "en") in LANGUAGE_PAIRS and ("en", tgt) in LANGUAGE_PAIRS
    if pivot_ok:
        logger.info(f"Pivot: {src}→en→{tgt}")
        en_text = _run_model(text, src, "en")
        result  = _run_model(en_text, "en", tgt)
        return result, True

    supported = ", ".join(f"{s}→{t}" for s, t in LANGUAGE_PAIRS)
    raise HTTPException(400, f"Không hỗ trợ '{src}'→'{tgt}' (kể cả qua pivot). Hỗ trợ: {supported}")


class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

LANG_MAP = {
    "zh-cn": "zh", "zh-tw": "zh",
    "ja": "ja", "ko": "ko",
    "vi": "vi", "en": "en",
    "fr": "fr", "de": "de",
    "es": "es", "it": "it",
    "pt": "pt", "ru": "ru",
    "ar": "ar",
}
SUPPORTED_DETECT = set(LANG_MAP.values())


def detect_language(text: str) -> str:
    if len(text.strip()) < 10:
        return "en"
    try:
        raw = detect(text)
        mapped = LANG_MAP.get(raw, raw)
        all_langs = set(l for pair in LANGUAGE_PAIRS for l in pair)
        return mapped if mapped in all_langs else "en"
    except LangDetectException:
        return "en"


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cached": [f"{s}→{t}" for s, t in _cache],
    }


@app.get("/api/pairs")
def pairs():
    direct = set(LANGUAGE_PAIRS.keys())
    all_langs = set(l for pair in LANGUAGE_PAIRS for l in pair)
    result = []
    for s in all_langs:
        for t in all_langs:
            if s == t:
                continue
            if (s, t) in direct:
                result.append({"src": s, "tgt": t, "pivot": False})
            elif (s, "en") in direct and ("en", t) in direct:
                result.append({"src": s, "tgt": t, "pivot": True})
    return result


@app.post("/api/translate")
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(400, "Văn bản trống")

    src = detect_language(req.text.strip()) if req.src_lang == "auto" else req.src_lang

    if src == req.tgt_lang:
        return {"translated": req.text, "pivot": False, "detected_src": src}

    result, is_pivot = translate_with_pivot(req.text.strip(), src, req.tgt_lang)
    return {"translated": result, "pivot": is_pivot, "detected_src": src}


if os.path.exists("index.html"):
    @app.get("/")
    def frontend():
        return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)