"""Optional LangSmith tracing with opt-in complaint text capture."""

import logging
import os
from functools import lru_cache

from langsmith import Client, traceable, tracing_context
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def tracing_client(api_key):
    return Client(api_key=api_key)


def route_with_trace(router, text, variant, *, endpoint="/route", request_id=None):
    include_text = os.getenv("LANGSMITH_TRACE_INPUT_TEXT", "").lower() == "true"
    inputs = {
        "model_variant": variant,
        "endpoint": endpoint,
        "request_id": request_id,
        "character_count": len(text),
        "word_count": len(text.split()),
        "text_included": include_text,
    }
    if include_text:
        inputs["complaint_text"] = text
    result = None
    attempted = False
    tree = None

    @traceable(name="classify", run_type="chain")
    def invoke(complaint):
        nonlocal result, attempted, tree
        attempted = True
        tree = get_current_run_tree()
        if tree is not None:
            tree.add_metadata({"request_id": request_id, "endpoint": endpoint})
        result = router.classify(text, complaint["model_variant"])
        if tree is not None:
            tree.add_metadata({
                "input_length": complaint["character_count"], "predicted_label": result["label"],
                "urgent": result["urgent"], "latency_ms": result["latency_ms"],
            })
        return result

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        with tracing_context(enabled=False):
            return {**invoke(inputs), "trace_url": None}

    try:
        client = tracing_client(api_key)
        with tracing_context(
            enabled=True, client=client, project_name="bank-complaint-router",
            tags=[variant, os.getenv("ADAPTER_VERSION", "lora-r8")],
        ):
            result = invoke(inputs)
        url = tree.get_url() if tree is not None else None
    except Exception:
        # Inference failures must propagate. Tracing failures after inference
        # return the completed result and never perform inference twice.
        if result is None and attempted:
            raise
        if result is None:
            with tracing_context(enabled=False):
                result = invoke(inputs)
        logger.warning("LangSmith trace URL unavailable", exc_info=True)
        url = None
    return {**result, "trace_url": url}
