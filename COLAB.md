# Colab runbook

## Start a T4 session

1. Open [Google Colab](https://colab.research.google.com/) and confirm the profile is the Algo Shank account used for the project.
2. Use **File → Upload notebook** to open the appropriate local notebook below. Save a Drive copy if you want to retain the notebook and its cell outputs.
3. Choose **Runtime → Change runtime type → T4 GPU** and connect. Run `!nvidia-smi -L` in a code cell and check that the output identifies a T4. Colab assigns accelerators subject to quota and availability. Files in `/content` are temporary, so download each output ZIP before ending the session. [Colab FAQ](https://research.google.com/colaboratory/faq.html) · [GPU settings and availability](https://colab.research.google.com/notebooks/pro.ipynb)

## Train, merge, and smoke-test

Upload [finetune_complaint_router.ipynb](notebooks/finetune_complaint_router.ipynb). Run its numbered sections in order.

| Section | What to check |
| --- | --- |
| 1. Install and set up | LLaMA-Factory v0.9.4 installs and the notebook creates a run directory under `/content/complaint_router/`. |
| 2. Prepare ShareGPT | When the upload widget appears, select only `data/processed/train.csv`. The approved file contains 3,360 complaints, 480 per queue. The cell checks the file hash and class balance. |
| 3. Train | Qwen3-1.7B-Base trains with rank-8 LoRA for two epochs, FP16, 512-token cutoff, learning rate 0.0002, micro-batch 2, and gradient accumulation 8. |
| 4–5. Plot and merge | The notebook saves the training loss data and plot, then merges the adapter into the base checkpoint. |
| 6. Smoke-test | Check that all seven queue examples pass before using the model in the demo. |
| 7. Bundle | From the Files pane, download the printed `/content/complaint-router-<run-id>.zip` to Week5. The bundle includes the merged model, training and merge configs, manifest, loss artifacts, and smoke-test results. |

Keep `val.csv` and `april_prompt_holdout.csv` for evaluation. The [executed training notebook](https://colab.research.google.com/drive/1uwWeV3jwd7UX69CyeYMDRUH_03QV9Wdl) and `complaint-router-20261106-035015-773607.zip` are the evidence and checkpoint from the completed run. Use that bundle to reproduce the reported results.

## Evaluate the January–February holdout

Open [evaluate_complaint_router.ipynb](notebooks/evaluate_complaint_router.ipynb) in a T4 session. In Colab's Files pane, upload these three files directly to `/content`:

| Local file | Purpose |
| --- | --- |
| `complaint-router-20261106-035015-773607.zip` | Exact model bundle expected by the evaluation notebook, 3,457,143,110 bytes |
| `data/processed/val.csv` | Unmodified 840-row holdout, 120 per queue |
| `backend/eval.py` | Evaluates both models and optionally publishes their predictions |

In Colab **Secrets**, add `LANGSMITH_API_KEY` and enable notebook access if publishing experiments. Run the setup cell after uploading the files, then run the evaluation cell. The original baseline selects the highest-scoring A–G letter. The fine-tuned model generates a queue label. Download `/content/phase3-eval-results.zip` and keep the comparison JSON, both model JSONs, and experiment links. The saved run scored **18.93%** for the original baseline and **85.48%** for the tuned model.

If you evaluated before adding the secret, enable access and run the final **publish saved predictions** cell in the same runtime. It reuses the saved predictions. LangSmith gets row IDs, labels and latency for these experiments. The live FastAPI tracing settings are separate. [Baseline experiment](https://smith.langchain.com/o/44990052-f9fe-499b-b7da-2d2fc85355e7/projects/p/4d657dab-7bb8-4acb-b1c2-6d6dd734297c) · [Fine-tuned experiment](https://smith.langchain.com/o/44990052-f9fe-499b-b7da-2d2fc85355e7/projects/p/3fa82a6d-ea5c-423c-bc6d-6e46e66a1384)

If the bundle upload stalls, check that the filename and byte count match the table. The existing setup cell also accepts 13 raw parts named `router-part-00` through `router-part-12`, with the first 12 at 256 MiB each. The notebook waits for the upload to finish before extracting. If the runtime is replaced, upload the inputs again and restart from the setup cell.

## Check the April results

April is a separate 2,800-row holdout with 400 complaints per queue. The published comparison uses the same January–February checkpoint and the few-shot Qwen baseline. The tuned model scored **84.00%** versus **33.61%** for the baseline, a **50.39 percentage-point gain**. The tuned score was 1.48 points below its January–February result. See [saved April metrics](eval_results/april_finetuned_lora.json) and the [slide brief](demo_slide_brief.md) for the evidence and prompt audit.

To rerun the April prompt audit, upload `data/processed/april_prompt_holdout.csv` and `backend/audit_baseline.py` to the evaluation runtime after setup. Run this in a separate code cell:

```python
!python /content/audit_baseline.py --data /content/april_prompt_holdout.csv --sha256 50d863cb0ac71866593e76ce7abe35e31139f26d1341ab40b925890c52659d35 --per-class 400 --out /content/april_baseline_audit.json
```

Read `few_shot.accuracy` from the output JSON. The January–February evaluation notebook's main scoring cell expects the original `val.csv` hash and class balance, so keep the two runs separate.
To score the merged model on the same April rows, run this in another code cell after the setup cell. The command writes a separate JSON and does not publish to LangSmith.

```python
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, "/content")
from eval import evaluate_model, summarize, write_json

april = Path("/content/april_prompt_holdout.csv")
digest = hashlib.sha256(april.read_bytes()).hexdigest()
assert digest == "50d863cb0ac71866593e76ce7abe35e31139f26d1341ab40b925890c52659d35"
artifacts = Path("/content/phase2_artifacts")
manifest = json.loads((artifacts / "run_manifest.json").read_text(encoding="utf-8"))
with april.open(encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))
labels = manifest["labels"]
assert Counter(row["label"] for row in rows) == {label: 400 for label in labels}
predictions = evaluate_model(artifacts / "merged_model", rows, manifest, baseline=False)
metrics = summarize(rows, predictions, labels)
write_json(Path("/content/april_finetuned_lora.json"), {
    "validation_sha256": digest,
    **metrics,
    "predictions": predictions,
})
print(f"April tuned accuracy: {metrics['accuracy']:.2%}")
```

Download `april_finetuned_lora.json` and `april_baseline_audit.json` from the Files pane before disconnecting.
