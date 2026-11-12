"""Check option order, calibration, and few-shot prompting on the fixed split."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from string import ascii_uppercase

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = (
    "Credit reporting", "Debt collection", "Credit card", "Checking or savings account",
    "Mortgage", "Student loan", "Money transfer",
)
DEMOS = (
    "A credit bureau lists a late payment on my report even though I paid on time.",
    "A collection agency keeps calling and refuses to validate the debt.",
    "My credit card statement shows a purchase I never made.",
    "My checking account was closed and the bank has not returned the balance.",
    "My mortgage servicer scheduled foreclosure while reviewing my modification.",
    "My student loan servicer failed to count my payments toward forgiveness.",
    "Someone used my mobile wallet to send an unauthorized instant transfer.",
)
BASE = "Qwen/Qwen3-1.7B-Base"
LETTERS = ascii_uppercase[:len(LABELS)]
VALIDATION_SHA256 = "4fa1aa1cb1528bdcc21d6b5fb83d0fabbebeb57623eb96b76f3acacb586ad4cc"


def prompt(text, order, few_shot=False):
    choices = "\n".join(f"{letter}. {label}" for letter, label in zip(LETTERS, order))
    demos = "".join(
        f"Complaint: {example}\nAnswer: {LETTERS[order.index(label)]}\n\n"
        for label, example in zip(LABELS, DEMOS)
    ) if few_shot else ""
    return f"Route the bank complaint. Reply with one letter only.\n{choices}\n\n{demos}Complaint: {text}\nAnswer:"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/content/val.csv"))
    parser.add_argument("--sha256", default=VALIDATION_SHA256)
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("/content/baseline_audit.json"))
    args = parser.parse_args()
    path = args.data
    assert hashlib.sha256(path.read_bytes()).hexdigest() == args.sha256
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert Counter(row["label"] for row in rows) == {label: args.per_class for label in LABELS}
    tokenizer = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda").eval()
    tokens = []
    prefix = prompt("N/A", LABELS)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    for letter in LETTERS:
        continuation = tokenizer.encode(prefix + " " + letter, add_special_tokens=False)
        standalone = tokenizer.encode(" " + letter, add_special_tokens=False)
        assert len(standalone) == 1 and continuation == prefix_ids + standalone
        tokens.extend(standalone)

    @torch.inference_mode()
    def scores(texts, order, few_shot=False):
        batch = tokenizer([prompt(text, order, few_shot) for text in texts],
                          padding=True, return_tensors="pt").to(model.device)
        return model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, tokens].float().cpu()

    results = {}
    for name, order, few_shot in (
        ("original", LABELS, False),
        ("rotated", LABELS[3:] + LABELS[:3], False),
        ("few_shot", LABELS, True),
    ):
        bias = scores(["N/A"], order, few_shot)[0]
        logits = torch.cat([
            scores([row["text"] for row in rows[index:index + 8]], order, few_shot)
            for index in range(0, len(rows), 8)
        ])
        for variant, values in ((name, logits), (name + "_calibrated", logits - bias)):
            predictions = [order[index] for index in values.argmax(dim=1).tolist()]
            counts = Counter(predictions)
            correct = sum(label == row["label"] for label, row in zip(predictions, rows))
            results[variant] = {
                "correct": correct, "accuracy": correct / len(rows),
                "option_order": order, "prediction_counts": {label: counts[label] for label in LABELS},
                "predictions": predictions,
            }
            print(f"{variant}: {correct}/{len(rows)} ({correct / len(rows):.2%}) {dict(counts)}", flush=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
