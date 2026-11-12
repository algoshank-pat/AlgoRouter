# Bank Complaint Router — Results and Status

**Suggested slide title**

Fine-tuned Qwen routed 84% of 2,800 new bank complaints correctly

**Main comparison**

**33.61% → 84.00% accuracy on April complaints**

**+50.39 percentage points** over the strongest Qwen base prompt tested on the same April set

| Evaluation set | Qwen base with examples | Fine-tuned Qwen | Accuracy gain |
| --- | ---: | ---: | ---: |
| January–February holdout, 840 complaints | 35.60% (299 correct) | **85.48% (718 correct)** | **+49.88 pp** |
| April holdout, 2,800 complaints | 33.61% (941 correct) | **84.00% (2,352 correct)** | **+50.39 pp** |

**How performance changed on new data**

The same January–February checkpoint scored **1.48 percentage points lower** on April. Both sets were balanced across seven queues. The April set contained 400 complaints per queue and excluded exact cleaned narratives already used in training or validation.

**What the audit found**

The original zero-shot A–G prompt scored 18.93% on the first holdout and sent 838 of 840 complaints to Credit reporting or Debt collection. Adding one handwritten example per queue raised accuracy to 35.60%. On the fresh April set, the fixed zero-shot prompt scored 20.00% and the same prompt with examples scored 33.61%. The prompt improvement held on new data.

**Performance for urgent queues**

| Fine-tuned model metric | January–February | April |
| --- | ---: | ---: |
| Mortgage recall | 93.33% (112 of 120) | **95.75% (383 of 400)** |
| Money transfer recall | 82.50% (99 of 120) | **77.00% (308 of 400)** |

Money transfer recall fell by **5.50 percentage points**, with 92 of 400 April complaints missed. This remains a limitation despite the overall accuracy result.

**Project status**

Data preparation, LoRA training, merged-model export, and all seven smoke tests are complete. Both held-out evaluations and the prompt audit are complete, with the model bundle and April metrics saved locally. The original zero-shot baseline and fine-tuned January–February experiments are published in LangSmith with 840 scored rows each, using row IDs and labels. The April results are saved locally. FastAPI and the React interface are complete. The browser demo verified live routing, side-by-side comparison, evaluation charts, and LangSmith traces. The Colab runbook and five-minute demo script are ready for review.

**Suggested speaker note**

“We trained a 1.7-billion-parameter model to route complaints across seven bank queues. It reached 85.5% accuracy on the first holdout. We then tested the same checkpoint on 2,800 new April complaints, where it scored 84%. A base model given one example per queue scored 33.6% on those same complaints. Mortgage recall improved, while Money transfer recall fell to 77%, identifying a specific area for further work.”

**Slide footnote**

Public CFPB narratives. Training used 3,360 January–February 2026 complaints. The first holdout contained 840 complaints, balanced at 120 per queue. The few-shot prompt was selected during an exploratory audit on that holdout, then reused on April. April contained 2,800 complaints, balanced at 400 per queue, with no exact cleaned narrative overlap with the earlier data. The checkpoint was evaluated on April without additional training.

**Source files in Week5**

`eval_results/april_finetuned_lora.json` · `eval_results/comparison.json` · `eval_results/baseline_audit.json` · `eval_results/langsmith_experiments.json` · `data/processed/april_prompt_holdout.csv`

The April baseline audit and its 941-of-2,800 result are recorded in the Colab evaluation notebook's April audit cell. The April LoRA JSON contains all 2,800 predictions, per-class metrics, and the confusion matrix.
