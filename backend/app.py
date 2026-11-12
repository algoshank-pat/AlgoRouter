"""FastAPI entry point. Run from Week5 with uvicorn backend.app:app."""

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from .model import InvalidPrediction, ModelUnavailable, Router
from .tracing import route_with_trace

logger = logging.getLogger(__name__)
router = Router()


@asynccontextmanager
async def lifespan(app):
    yield
    with router.lock:
        router.close()


app = FastAPI(title="Bank Complaint Router", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


class Complaint(BaseModel):
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def clean_text(cls, text):
        text = " ".join(text.split())
        if not text:
            raise ValueError("Enter a complaint narrative.")
        # Match Phase 1 cleaning and truncation.
        if len(text) > 1200:
            text = text[:1201].rpartition(" ")[0] or text[:1200]
        return text


class RouteResult(BaseModel):
    label: str
    queue: str
    urgent: bool
    latency_ms: float
    model: Literal["finetuned", "baseline"]
    trace_url: str | None


class Comparison(BaseModel):
    baseline: RouteResult
    finetuned: RouteResult


def predict(text, variant, *, endpoint, request_id):
    try:
        return route_with_trace(router, text, variant, endpoint=endpoint, request_id=request_id)
    except ModelUnavailable as error:
        logger.exception("Model unavailable")
        raise HTTPException(503, str(error)) from error
    except InvalidPrediction as error:
        raise HTTPException(502, str(error)) from error
    except Exception as error:
        logger.exception("Inference failed")
        raise HTTPException(503, "Inference failed. Check the server logs.") from error


@app.post("/route", response_model=RouteResult)
def route(complaint: Complaint, model: Literal["finetuned", "baseline"] = "finetuned"):
    return predict(complaint.text, model, endpoint="/route", request_id=str(uuid4()))


@app.post("/compare", response_model=Comparison)
def compare(complaint: Complaint):
    request_id = str(uuid4())
    return {
        "baseline": predict(complaint.text, "baseline", endpoint="/compare", request_id=request_id),
        "finetuned": predict(complaint.text, "finetuned", endpoint="/compare", request_id=request_id),
    }


@app.get("/eval")
def evaluation():
    directory = Path(os.getenv("EVAL_RESULTS_DIR", str(ROOT / "eval_results")))
    results = {}
    try:
        for path in sorted(directory.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                # Serve metrics and experiment links without thousands of
                # individual prediction records.
                value = {key: item for key, item in value.items() if key != "predictions"}
                if path.stem == "baseline_audit":
                    value = {
                        key: {name: item for name, item in record.items() if name != "predictions"}
                        for key, record in value.items()
                    }
            results[path.stem] = value
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Evaluation results could not be read.") from error
    if not results:
        raise HTTPException(404, "No evaluation results are available.")
    return results


@app.get("/health")
def health():
    return {
        "status": "ok", "engine": router.engine, "loaded_model": router.variant,
        "device": router.device, "tracing_configured": bool(os.getenv("LANGSMITH_API_KEY")),
    }
