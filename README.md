# Bank Complaint Router

This internal routing prototype classifies free-text complaints into seven specialist queues. The React demo shows the recommended queue and flags Mortgage and Money transfer for urgent review.

The training and first validation split came from the January–February 2026 CFPB complaint export. The separate April holdout checks how the same checkpoint performs on later complaints. For these historical exports, CFPB obtained consumer consent before publication and took steps to remove personal information. Treat pasted narratives as potentially sensitive and use synthetic examples for demos. [CFPB narrative publication standard](https://files.consumerfinance.gov/f/documents/cfpb_narrative-scrubbing-standard_2023-05.pdf)

## Colab and demo

- [Colab runbook](COLAB.md) covers the T4 setup, training and evaluation inputs, output downloads, and LangSmith publishing.
- [Five-minute demo script](DEMO_SCRIPT.md) follows the executed notebook, live React comparison, evaluation charts, and trace links.
- [Slide brief](demo_slide_brief.md) includes April results and evidence for the presentation.

## FastAPI backend

From `C:\Maven\Projects\Week5`, create a Python virtual environment and install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend/requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The local checkpoints are under `models/merged_model` and `models/qwen_base`. Set `MERGED_MODEL_PATH` or `BASE_MODEL_PATH` in `.env` to use different locations. If the base checkpoint is absent, the first baseline request downloads it from Hugging Face. Add `LANGSMITH_API_KEY` to `.env` for project `bank-complaint-router`. Open `http://127.0.0.1:8000/docs` to try the API.

| Endpoint | Behavior |
| --- | --- |
| `POST /route` | Accepts `{"text": "..."}` and returns the fine-tuned label, queue, urgency, latency, model variant, and trace URL. Use `?model=baseline` for the base model. |
| `POST /compare` | Accepts the same body and returns `baseline` and `finetuned` results. |
| `GET /eval` | Returns saved metrics, confusion matrices, audit summaries, and LangSmith experiment links. |
| `GET /health` | Reports service status, inference engine, resident model, device, and tracing configuration. |

Input cleaning matches the training preparation, with whitespace collapsed and a 1,200-character limit at a word boundary. Bodies containing more than 20,000 characters are rejected. Mortgage and Money transfer results carry `urgent: true`.

Transformers uses CUDA when available and float32 on CPU. Models load lazily, and one model is resident at a time. Requests are serialized around inference and model switching. The first request and model switches include loading time beyond the reported inference latency. The comparison endpoint uses the original zero-shot letter-choice baseline associated with the 18.93% experiment. The 35.60% result refers to the separate few-shot audit.

Traces include the endpoint, model variant, character and word counts of the cleaned model input, and a request ID shared by both runs from a comparison. Outputs include the prediction, queue, urgency, and inference latency. Complaint text is omitted by default. For a controlled demo with synthetic complaints, set `LANGSMITH_TRACE_INPUT_TEXT=true` in the backend environment and restart it. This sends the cleaned, truncated narrative for every routing or comparison request to LangSmith. Leave it off for real complaints.

Missing credentials disable tracing. Tracing setup or URL lookup failures return the inference result with `trace_url: null`. A trace URL can be created before LangSmith confirms ingestion.

For optional CPU GGUF inference, install `llama-cpp-python`, set `MODEL_ENGINE=llama_cpp`, and provide `GGUF_MODEL_PATH` and `GGUF_BASELINE_PATH`. Use the Qwen tokenizer and ChatML template when converting the merged model. Quantization can reduce memory requirements and change accuracy. This path constrains label/letter generation with a grammar and requires its own evaluation before comparing its results with the Transformers metrics.

## React UI

Start FastAPI using the command above. In a second PowerShell terminal from Week5, run:

```powershell
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The app offers routing with seven synthetic samples, side-by-side comparison, and an evaluation dashboard. Vite proxies `/api` to FastAPI on port 8000. Run `npm run build` to generate `frontend/dist`. For a separately hosted build, configure an `/api` reverse proxy or set `VITE_API_BASE_URL` before building and allow the frontend origin in FastAPI's `CORS_ORIGINS`.

## Held-out evaluation

Both models scored the same 840 January–February 2026 CFPB narratives, with 120 per class. The Qwen3-1.7B-Base baseline selected the highest-scoring letter from A–G. The merged LoRA model generated a routing label with generation stopping at a complete label. Latency was measured one complaint at a time on the same Colab T4 after a warm-up call, including tokenization and decoding where applicable.

| Metric | Qwen base | Merged LoRA |
| --- | ---: | ---: |
| Accuracy | 18.93% | 85.48% |
| Mortgage recall | 0.00% | 93.33% |
| Money transfer recall | 0.00% | 82.50% |
| Mean latency per complaint | 68.2 ms | 143.4 ms |

Credit reporting complaints routed to Debt collection fell from 25 to 5. Debt collection complaints routed to Credit reporting fell from 58 to 14. The two directions together accounted for 83 baseline errors and 19 merged-model errors.

The original zero-shot prompt sent 838 of 840 complaints to Credit reporting or Debt collection. A follow-up audit on the same split verified the answer-token encoding. The batched repeat scored 19.05%, rotating the choices scored 20.36%, and adding one handwritten example per class scored 35.60% with the same letter scoring. Subtracting the logits for a content-free complaint reduced accuracy in all three setups. The merged model led the strongest tested prompt by 49.88 percentage points. The stronger prompt was identified during an exploratory audit of this validation split. See `eval_results/baseline_audit.json` for all six runs.

`eval_results/comparison.json` contains both confusion matrices, full classification reports, and per-class changes. The two model JSON files also contain predictions and latency for each validation row in CSV order.

Both exact-match experiments were published from the saved predictions on `cfpb-complaints-val`. The LangSmith dataset uses validation row IDs and reference labels. Each experiment includes the predicted label and measured inference latency. Links are saved in `eval_results/langsmith_experiments.json`.

## April Money transfer augmentation experiment

We added 240 unused January–February Money transfer complaints with bank-account wording, increasing that class from 480 to 720 training examples. Each other class stayed at 480. We used the original LoRA settings and compared the checkpoints on the same 2,800 April complaints, with 400 per class.

| April metric | Original LoRA | Augmented LoRA | Change |
| --- | ---: | ---: | ---: |
| Money transfer recall | 77.00% | 87.75% | +10.75 pp |
| Checking or savings account recall | 74.00% | 60.00% | −14.00 pp |
| Money transfer precision | 79.18% | 70.06% | −9.12 pp |
| Overall accuracy | 84.00% | 83.57% | −0.43 pp |

Money transfer complaints routed to Checking or savings fell from 70 to 33. In the reverse direction, Checking or savings complaints routed to Money transfer rose from 63 to 119. The augmented model caught more transfers and also assigned the transfer label to more account complaints. Recall changes in the other five queues ranged from −2.00 to +1.75 percentage points.

This single run changed both the wording and the number of Money transfer training examples. An equal-size random augmentation would be needed to isolate the effect of selecting account-like wording. [Colab run](https://colab.research.google.com/drive/19PVe5p6cfQLhT2_39KSWX2HY9ehCKAIQ) · [Paired results and confusion matrix](eval_results/april_augmentation_comparison.json)

## Limitations

The holdouts are balanced across queues. Production complaint volumes and label mix can differ. Queue boundaries can also be ambiguous, and the CFPB narratives represent consumers who opted in to publication. The original January–February checkpoint achieved 77% Money transfer recall on the April holdout, so urgent complaints still need human review. This prototype is for internal triage and has no customer-facing output.

See [submission links](SUBMISSION_LINKS.md) for the executed Colab notebooks and the verified Mortgage traces.
