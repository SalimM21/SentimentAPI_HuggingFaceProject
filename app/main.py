import os
import time
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from transformers import pipeline

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("sentiment-api")

# ---------------- Config -----------------
MODEL_NAME = os.getenv("HF_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")

# ---------------- App --------------------
app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description=(
        "Binary sentiment (positive/negative) with confidence using a lightweight CPU model: "
        f"{MODEL_NAME}"
    ),
)

# CORS (keep permissive for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (page HTML de test)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def root_page():
    """Serve the minimal HTML test page at the root."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"status": "ok", "model": MODEL_NAME})
    return FileResponse(index_path)


# Load pipeline once at startup
try:
    logger.info(f"Loading model pipeline: {MODEL_NAME}")
    sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_NAME)
    logger.info("Model pipeline loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model pipeline.")
    raise e


class SentimentRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be a non-empty string")
        return v.strip()


class SentimentResponse(BaseModel):
    label: Literal["positive", "negative"]
    confidence: float
    model: str


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=SentimentResponse, summary="Analyse de sentiments (binaire)")
def predict(req: SentimentRequest, request: Request):
    """
    Entrée : { "text": "your sentence" }
    Sortie : { "label": "positive|negative", "confidence": 0.0-1.0, "model": "<model_name>" }
    """
    start = time.time()
    try:
        result = sentiment_pipe(req.text)[0]  # e.g., {'label': 'POSITIVE', 'score': 0.999}
        label = "positive" if result["label"].upper().startswith("POS") else "negative"
        confidence = float(result["score"])
        duration_ms = (time.time() - start) * 1000.0

        logger.info(
            "prediction success | bytes=%d | ms=%.2f | label=%s | conf=%.4f",
            len(req.text.encode("utf-8")),
            duration_ms,
            label,
            confidence,
        )

        return SentimentResponse(label=label, confidence=confidence, model=MODEL_NAME)
    except Exception as e:
        logger.exception("prediction error")
        raise HTTPException(status_code=500, detail=str(e))