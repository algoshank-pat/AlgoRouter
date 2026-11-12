# Five-minute demo script

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:40 | React landing page and architecture | “A complaint can arrive as free text. This prototype routes it to one of seven specialist queues and flags two queues for urgent review.” |
| 0:40–1:30 | Executed Colab notebook and loss curve | “We fine-tuned Qwen3-1.7B-Base on 3,360 January–February CFPB complaints. The T4 run used rank-8 LoRA and two epochs. After merging the adapter, we checked all seven queues with smoke tests.” |
| 1:30–2:25 | Compare models with the Mortgage sample | “Both models see the same foreclosure complaint. The base model picks Debt collection and the tuned model picks Mortgage and marks it urgent. This is one example. The held-out results tell us how often the model gets routing right.” |
| 2:25–3:25 | Evaluation dashboard, recall chart, and confusion matrix | “On 840 January–February complaints, the tuned model reached 85.48%. The original zero-shot baseline got 18.93%, and a prompt audit lifted it to 35.60%. On 2,800 separate April complaints, that stronger baseline reached 33.61%, compared with 84.00% from the same tuned checkpoint.” |
| 3:25–4:15 | Open both LangSmith trace links | “The two predictions share a request ID, so we can connect the model decisions. Each trace records the input size, prediction, and latency. Complaint text stays out of traces by default.” |
| 4:15–5:00 | April urgent-class results and close | “April mortgage recall was 95.75%. Money transfer recall was 77%, so 92 of 400 money transfer complaints were missed. A pilot could measure review time, rerouting, and infrastructure cost per complaint. This is an internal routing aid that still needs human review.” |

Before recording, start FastAPI and Vite from the [README](README.md), confirm the LangSmith links are available, and use synthetic samples. Keep the [Colab runbook](COLAB.md) and [slide brief](demo_slide_brief.md) open for reference.
