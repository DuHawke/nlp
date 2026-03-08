from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import torch
import logging
import os

DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NLLB-200 Translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LANG_CODES = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "th": "tha_Thai",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "uk": "ukr_Cyrl",
    "he": "heb_Hebr",
}

DETECT_MAP = {
    "zh-cn": "zh", "zh-tw": "zh",
    "vi": "vi", "en": "en", "ja": "ja", "ko": "ko",
    "fr": "fr", "de": "de", "es": "es", "ru": "ru",
    "ar": "ar", "pt": "pt", "it": "it", "th": "th",
    "hi": "hi", "tr": "tr", "nl": "nl", "pl": "pl",
    "sv": "sv", "da": "da", "fi": "fi", "cs": "cs",
    "ro": "ro", "hu": "hu", "id": "id", "ms": "ms",
    "uk": "uk", "he": "he",
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE}")

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    logger.info(f"Loading {MODEL_NAME} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    _model.eval()
    logger.info("NLLB-200 loaded ✓")


def detect_language(text: str) -> str:
    if len(text.strip()) < 8:
        return "en"
    try:
        raw = detect(text)
        mapped = DETECT_MAP.get(raw, raw)
        return mapped if mapped in LANG_CODES else "en"
    except LangDetectException:
        return "en"


def translate(text: str, src: str, tgt: str) -> str:
    load_model()
    src_nllb = LANG_CODES.get(src)
    tgt_nllb = LANG_CODES.get(tgt)
    if not src_nllb:
        raise HTTPException(400, f"Ngôn ngữ nguồn '{src}' không được hỗ trợ")
    if not tgt_nllb:
        raise HTTPException(400, f"Ngôn ngữ đích '{tgt}' không được hỗ trợ")

    _tokenizer.src_lang = src_nllb
    inputs = _tokenizer(
        text, return_tensors="pt",
        padding=True, truncation=True, max_length=512,
    ).to(DEVICE)

    tgt_lang_id = _tokenizer.convert_tokens_to_ids(tgt_nllb)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=512,
            num_beams=5,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return _tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


class TranslateRequest(BaseModel):
    text: str
    src_lang: str   # "auto" hoặc mã ngôn ngữ
    tgt_lang: str

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "loaded": _model is not None,
    }


@app.get("/api/langs")
def langs():
    return [{"code": k, "nllb": v} for k, v in LANG_CODES.items()]


@app.post("/api/translate")
def api_translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(400, "Văn bản trống")

    src = detect_language(req.text.strip()) if req.src_lang == "auto" else req.src_lang

    if src == req.tgt_lang:
        return {"translated": req.text, "detected_src": src}

    if req.tgt_lang not in LANG_CODES:
        raise HTTPException(400, f"Ngôn ngữ đích '{req.tgt_lang}' không hỗ trợ")

    result = translate(req.text.strip(), src, req.tgt_lang)
    return {"translated": result, "detected_src": src}


if os.path.exists("index.html"):
    @app.get("/")
    def frontend():
        return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)