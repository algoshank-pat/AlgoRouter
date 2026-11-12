"""Compare Qwen base and the merged complaint router on the held-out CSV."""

import argparse
import csv
import gc
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from string import ascii_uppercase
from time import perf_counter


BASE = "Qwen/Qwen3-1.7B-Base"
VARIANTS = ("baseline-qwen3-1.7b", "finetuned-lora-r8")
DATASET = "cfpb-complaints-val"
VAL_HASH = "4fa1aa1cb1528bdcc21d6b5fb83d0fabbebeb57623eb96b76f3acacb586ad4cc"
INVALID = "<invalid>"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def classify(model, tokenizer, text, labels, system, *, baseline, stopping=None):
    letters = ascii_uppercase[:len(labels)]
    if baseline:
        choices = "\n".join(f"{letter}. {label}" for letter, label in zip(letters, labels))
        prompt = f"Route the bank complaint. Reply with one letter only.\n{choices}\n\nComplaint: {text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Compare the conditional likelihood of each single-token answer.
        candidates = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in letters]
        if any(len(tokens) != 1 for tokens in candidates):
            raise ValueError("Expected single-token baseline answer choices.")
        token_ids = [tokens[0] for tokens in candidates]
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
        return labels[logits[token_ids].argmax().item()]

    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs, max_new_tokens=16, do_sample=False, stopping_criteria=stopping,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate_model(source, rows, manifest, baseline):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria, StoppingCriteriaList

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype).to(device).eval()
    stopping = None
    if not baseline:
        stopping = StoppingCriteriaList([StopStringCriteria(tokenizer, list(manifest["labels"]))])

    def predict(text):
        if device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        with torch.inference_mode():
            label = classify(model, tokenizer, text, manifest["labels"],
                             manifest["system_prompt"], baseline=baseline, stopping=stopping)
        if device == "cuda":
            torch.cuda.synchronize()
        return {"label": label, "latency_ms": (perf_counter() - start) * 1000}

    predict("My mortgage servicer charged an unexpected escrow fee.")
    predictions = []
    for index, row in enumerate(rows, 1):
        predictions.append(predict(row["text"]))
        if index % 100 == 0 or index == len(rows):
            print(f"{source}: {index}/{len(rows)}", flush=True)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return predictions


def summarize(rows, predictions, labels):
    from sklearn.metrics import classification_report, confusion_matrix

    expected = [row["label"] for row in rows]
    actual = [item["label"] if item["label"] in labels else INVALID for item in predictions]
    axis = [*labels, INVALID]
    report = classification_report(expected, actual, labels=axis, output_dict=True, zero_division=0)
    return {
        "accuracy": sum(a == b for a, b in zip(expected, actual)) / len(rows),
        "avg_latency_ms": sum(item["latency_ms"] for item in predictions) / len(rows),
        "invalid_predictions": actual.count(INVALID),
        "classification_report": report,
        "confusion_matrix": {"labels": axis, "counts": confusion_matrix(expected, actual, labels=axis).tolist()},
    }


def exact_match(outputs, reference_outputs):
    return {"key": "exact_match", "score": int(outputs["label"] == reference_outputs["label"])}


def log_experiments(rows, predictions, output):
    from langsmith import Client

    client = Client()
    expected = {f"val-{index:04d}": row["label"] for index, row in enumerate(rows, 1)}
    if client.has_dataset(dataset_name=DATASET):
        dataset = client.read_dataset(dataset_name=DATASET)
    else:
        dataset = client.create_dataset(DATASET, description="Held-out January–February 2026 CFPB complaints")
    examples = list(client.list_examples(dataset_id=dataset.id))
    if not examples:
        client.create_examples(dataset_id=dataset.id, examples=[
            {"inputs": {"row_id": row_id}, "outputs": {"label": label}}
            for row_id, label in expected.items()
        ])
        examples = list(client.list_examples(dataset_id=dataset.id))
    observed = {example.inputs.get("row_id"): example.outputs["label"] for example in examples}
    if observed != expected or len(examples) != len(rows):
        raise ValueError(f"Existing {DATASET} examples differ from the approved validation split.")

    links_path = output / "langsmith_experiments.json"
    links = json.loads(links_path.read_text(encoding="utf-8")) if links_path.exists() else {}
    for variant in VARIANTS:
        if variant in links:
            continue
        by_id = dict(zip(expected, predictions[variant]))

        def target(inputs):
            return by_id[inputs["row_id"]]

        result = client.evaluate(
            target, data=examples, evaluators=[exact_match], experiment_prefix=variant,
            max_concurrency=4, metadata={"validation_sha256": VAL_HASH, "source": "recorded GPU predictions"},
        )
        if len(list(result)) != len(rows):
            raise RuntimeError(f"LangSmith returned an incomplete experiment for {variant}.")
        project = client.read_project(project_name=result.experiment_name)
        links[variant] = {"name": result.experiment_name, "url": project.url}
        write_json(links_path, links)
        print(f"{result.experiment_name}: {project.url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("eval_results"))
    parser.add_argument("--log-only", action="store_true", help="Upload previously recorded predictions to LangSmith.")
    parser.add_argument("--resume-baseline", action="store_true", help="Reuse the baseline JSON from this output directory.")
    args = parser.parse_args()

    if hashlib.sha256(args.val.read_bytes()).hexdigest() != VAL_HASH:
        parser.error("Use the unmodified Phase 1 val.csv.")
    with args.val.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((args.merged.parent / "run_manifest.json").read_text(encoding="utf-8"))
    labels = manifest["labels"]
    if manifest["base_model"] != BASE or Counter(row["label"] for row in rows) != {label: 120 for label in labels}:
        raise ValueError("Model manifest and validation split do not match.")

    args.out.mkdir(parents=True, exist_ok=True)
    predictions = {}
    metrics = {}
    for variant, source in zip(VARIANTS, (BASE, args.merged)):
        path = args.out / f"{variant}.json"
        if args.log_only or (args.resume_baseline and variant == VARIANTS[0]):
            recorded = json.loads(path.read_text(encoding="utf-8"))
            if recorded["validation_sha256"] != VAL_HASH:
                raise ValueError("Recorded predictions refer to a different validation split.")
            predictions[variant] = recorded["predictions"]
            if not args.log_only:
                metrics[variant] = {
                    key: value for key, value in recorded.items()
                    if key not in {"validation_sha256", "predictions"}
                }
        else:
            predictions[variant] = evaluate_model(source, rows, manifest, variant == VARIANTS[0])
            metrics[variant] = summarize(rows, predictions[variant], labels)
            write_json(path, {
                "validation_sha256": VAL_HASH,
                **metrics[variant],
                "predictions": predictions[variant],
            })
        if len(predictions[variant]) != len(rows):
            raise ValueError(f"Prediction count differs from validation row count for {variant}.")

    if not args.log_only:
        baseline, tuned = (metrics[key] for key in VARIANTS)
        deltas = {
            label: {
                metric: tuned["classification_report"][label][metric] - baseline["classification_report"][label][metric]
                for metric in ("precision", "recall", "f1-score")
            }
            for label in labels
        }
        write_json(args.out / "comparison.json", {
            "validation_sha256": VAL_HASH, "examples": len(rows), "labels": labels,
            "high_urgency_labels": ["Mortgage", "Money transfer"],
            "models": metrics, "accuracy_delta": tuned["accuracy"] - baseline["accuracy"],
            "per_class_deltas": deltas,
        })

    from dotenv import load_dotenv

    load_dotenv()
    if os.environ.get("LANGSMITH_API_KEY"):
        log_experiments(rows, predictions, args.out)
    elif args.log_only:
        parser.error("Set LANGSMITH_API_KEY in .env or provide it via Colab Secrets.")
    else:
        print("Metrics saved. Add LANGSMITH_API_KEY and use --log-only to publish both experiments.")


if __name__ == "__main__":
    main()
