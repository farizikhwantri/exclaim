# EXCLAIM
**EX**plainable Comp**L**iance detection with **A**rgumentative **I**nference of **M**ulti-hop reasoning

Overview
EXCLAIM is a framework that adapts assurance case structures into a multi-hop natural language inference (NLI) formulation to support transparent, scalable compliance detection with Large Language Models (LLMs). It decomposes regulatory requirements into Claim–Argument–Evidence (CAE) assurance cases using LLMs, evaluates them as multi-step premise–hypothesis chains, and measures faithfulness and structural quality of the reasoning.

Key Contributions
- Faithfulness analysis: First systematic study of four interpretation methods in NLI-based compliance detection—Gradient, Integrated Gradients, LIME, and SHAP.
- Assurance-case adaptation: Adapts CAE as a discourse framework for compliance detection, enabling traceable, explainable multi-hop reasoning across steps.
- Data generation and evaluation: Generates CAE-structured assurance cases from GDPR requirements using open-source and proprietary LLMs; proposes metrics for instance-level consistency and structural coherence of LLM-generated assurance cases.
- Empirical results: Demonstrates improved explanation F1-score and faithfulness versus single-step baselines across encoder-based and decoder-only models, generalising to unseen requirements and public single-step NLI datasets.

Repository Structure
- src/
  - eval_llm_nli.py: Evaluate causal LLMs on NLI-style compliance (zero-shot, in-context, multi-hop).
  - attributions_llm.py: LLM faithfulness via post-hoc attribution and AOPC metrics on generated sequences.
  - attributions.py: Token attribution pipelines (Captum + Ferret) for encoder models; SHAP/LIME/IG/Gradient.
  - train.py: Supervised classifier training on CSV/GLUE datasets; supports auxiliary datasets and checkpoints.
  - train_prob_finetune.py: Probe-then-finetune training protocol for sequence classifiers.
  - train_lora.py: LoRA training for targeted modules in sequence classifiers.
  - pipeline.py: Model construction utilities and dataset loaders (CSV, GLUE).
- script/
  - llm_gen_claim_arg.py: Generate CAE-structured assurance cases from requirements with chat LLMs.
- data/
  - dpa/README.md: Source references for DPA and GDPR-related datasets.
- results/
  - llm_gen_data/slope_aopc_compr_llm.pdf: Sample figure for comprehensiveness slopes.

Installation
- Python >= 3.12
- Recommended: conda
- macOS (Apple Silicon supported), CUDA optional

````bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r [requirements.txt](http://_vscodecontentref_/0)
