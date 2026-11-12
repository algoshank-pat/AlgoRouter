# Submission links

- [Executed training notebook](https://colab.research.google.com/drive/1uwWeV3jwd7UX69CyeYMDRUH_03QV9Wdl)
- [Executed evaluation notebook](https://colab.research.google.com/drive/1jZiiHQjH3lr28Ad52O10yNSwFQxQeqN0)

Both notebooks are stored under the Algo Shank account. The notebook templates are included in the ZIP. Reviewer access depends on the sharing settings of the original notebooks and LangSmith project.

The synthetic Mortgage comparison produced two verified LangSmith runs with the shared request ID `934e39da-fda5-475c-a269-a6887d87a336`.

- Qwen base predicted Debt collection. [Baseline trace](https://smith.langchain.com/o/44990052-f9fe-499b-b7da-2d2fc85355e7/projects/p/0c56967f-d928-443c-8a00-cd00efeb4c79/r/01a1bc0b-7c1b-7c1e-9870-7254d2154d4b)
- Fine-tuned LoRA predicted Mortgage and flagged it as urgent. [Fine-tuned trace](https://smith.langchain.com/o/44990052-f9fe-499b-b7da-2d2fc85355e7/projects/p/0c56967f-d928-443c-8a00-cd00efeb4c79/r/01a1bc0b-97b1-7fe1-a764-5303231ecc7a)

The separate checkpoint export is `complaint-router-20261106-035015-773607.zip`. Follow the [README](README.md) for local setup and the [Colab runbook](COLAB.md) for dataset requirements. The evaluation JSONs include metrics and predictions without complaint narratives.
