from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .models import SummarizeRequest, SummarizeResponse
import torch

APP_TITLE = "ru-summarizer-api"
MODEL_NAME = "cointegrated/rut5-base-absum"   # <-- новая модель

app = FastAPI(title=APP_TITLE, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tokenizer = None
_model = None

def get_model():
    """Ленивая загрузка модели один раз на процесс."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        _model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    tokenizer, model = get_model()

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=req.max_length,
            min_length=req.min_length,
            num_beams=req.num_beams,
            length_penalty=1.2,
            early_stopping=True,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return SummarizeResponse(
        summary=summary,
        tokens_in=int(inputs.input_ids.shape[1]),
        tokens_out=int(summary_ids.shape[1]),
    )
