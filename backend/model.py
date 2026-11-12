"""Serialized, lazy inference for the evaluated complaint router."""

import gc
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

from .eval import BASE, classify as classify_transformers

ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    "Credit reporting", "Debt collection", "Credit card", "Checking or savings account",
    "Mortgage", "Student loan", "Money transfer",
)
QUEUES = {
    "Credit reporting": "Credit bureau disputes, inaccurate entries",
    "Debt collection": "Collections practices, validation, harassment",
    "Credit card": "Billing disputes, fees, rewards, card fraud",
    "Checking or savings account": "Overdrafts, deposits, closures",
    "Mortgage": "Servicing, escrow, modification, foreclosure",
    "Student loan": "Servicing, repayment, forgiveness",
    "Money transfer": "Zelle/wire/transfer fraud, mobile wallet, prepaid",
}
SYSTEM = "You are a bank complaint router. Reply with exactly one label from: " + ", ".join(LABELS) + "."


class ModelUnavailable(RuntimeError):
    pass


class InvalidPrediction(RuntimeError):
    pass


class Router:
    def __init__(self):
        self.engine = os.getenv("MODEL_ENGINE", "transformers")
        if self.engine not in {"transformers", "llama_cpp"}:
            raise ValueError("MODEL_ENGINE must be transformers or llama_cpp")
        self.lock = Lock()
        self.variant = None
        self.model = None
        self.tokenizer = None
        self.stopping = None
        self.device = None

    def close(self):
        if self.engine == "llama_cpp" and self.model is not None:
            self.model.close()
        self.model = None
        self.tokenizer = None
        self.stopping = None
        self.variant = None
        gc.collect()
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()

    def load(self, variant):
        if self.variant == variant:
            return
        self.close()
        if self.engine == "llama_cpp":
            from llama_cpp import Llama
            variable = "GGUF_BASELINE_PATH" if variant == "baseline" else "GGUF_MODEL_PATH"
            path = Path(os.getenv(variable, ""))
            if not path.is_file():
                raise ModelUnavailable(f"Set {variable} to a GGUF file.")
            self.model = Llama(model_path=str(path), n_ctx=2048, verbose=False)
            self.device = "cpu"
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria, StoppingCriteriaList
            local_base = ROOT / "models" / "qwen_base"
            source = os.getenv("BASE_MODEL_PATH") or (
                local_base if (local_base / "config.json").is_file() else BASE
            )
            if variant == "finetuned":
                source = Path(os.getenv("MERGED_MODEL_PATH", str(ROOT / "models" / "merged_model")))
                if not (source / "config.json").is_file():
                    raise ModelUnavailable("Extract the merged model and set MERGED_MODEL_PATH.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(source)
            model = AutoModelForCausalLM.from_pretrained(
                source, dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device).eval()
            self.tokenizer = tokenizer
            self.model = model
            self.device = device
            if variant == "finetuned":
                self.stopping = StoppingCriteriaList([StopStringCriteria(tokenizer, list(LABELS))])
        self.variant = variant

    def classify(self, text, variant):
        # One model resident at a time keeps CPU/GPU memory bounded.
        with self.lock:
            try:
                self.load(variant)
            except ModelUnavailable:
                raise
            except Exception as error:
                self.close()
                raise ModelUnavailable("Model loading failed. Check the model path and server logs.") from error
            start = perf_counter()
            if self.engine == "llama_cpp":
                label = self.classify_gguf(text, variant)
            else:
                import torch
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = perf_counter()
                with torch.inference_mode():
                    label = classify_transformers(
                        self.model, self.tokenizer, text, LABELS, SYSTEM,
                        baseline=variant == "baseline", stopping=self.stopping,
                    )
                if self.device == "cuda":
                    torch.cuda.synchronize()
            latency = (perf_counter() - start) * 1000
            if label not in QUEUES:
                raise InvalidPrediction("Model did not return one of the seven routing labels.")
            return {
                "label": label, "queue": QUEUES[label],
                "urgent": label in {"Mortgage", "Money transfer"},
                "latency_ms": round(latency, 2), "model": variant,
            }

    def classify_gguf(self, text, variant):
        from llama_cpp import LlamaGrammar
        from string import ascii_uppercase
        if variant == "baseline":
            choices = "\n".join(f"{letter}. {label}" for letter, label in zip(ascii_uppercase, LABELS))
            prompt = f"Route the bank complaint. Reply with one letter only.\n{choices}\n\nComplaint: {text}\nAnswer:"
            values = list(ascii_uppercase[:7])
        else:
            prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            values = list(LABELS)
        grammar = LlamaGrammar.from_string("root ::= " + " | ".join(f'"{value}"' for value in values))
        completion = self.model(prompt, max_tokens=20, temperature=0, grammar=grammar)
        answer = completion["choices"][0]["text"].strip()
        if variant == "baseline" and answer in values:
            return LABELS[values.index(answer)]
        return answer
