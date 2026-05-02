# Instruction-Conditioned LLMs for Grammatical Error Correction

This repository contains the official implementation and evaluation framework for the paper:

> Türker, T., Eryiğit, G. (2026). *Instruction-Following LLMs for Grammatical Error Correction: Analyzing Neutral-Anchored Instructional Sensitivity Across Editing Modes*. [Conference Name TBD].

## Overview

In this work, we probe whether models are sensitive to **editing mode** — the degree of correction aggressiveness encoded in the system prompt — by evaluating them under three zero-shot instruction variants: **Neutral**, **Min-Edit**, and **Fluency**. All three instructions target the same grammatical correction task but differ in how explicitly they constrain the model's editing behavior.

We benchmark seven instruction-following LLMs across three standard GEC datasets (CoNLL-2014, JFLEG, W&I+LOCNESS) and measure performance using the metrics established by each shared task.

If you use this framework or our findings in your research, please cite our paper (BibTeX entry will be added upon publication).

---

## Repository Structure

```
gec-llm-eval/
├── main.py                  # Inference entry point
├── eval.py                  # Evaluation entry point
├── api_caller.py            # LLM API wrappers (OpenAI, Claude, Gemini, Azure)
├── loader.py                # Data loading utilities
├── reporter.py              # Output writing (.jsonl, .txt, .m2)
├── globals.py               # Shared runtime state
├── tokenizer.py             # Tokenization utilities
├── sent_level_gleu.py       # Sentence-level GLEU scoring
├── config.yaml              # Runtime configuration (API keys, paths, model list)
├── requirements.txt
├── instructions/
│   ├── new/                 # Active instruction files used in experiments
│   │   ├── zs-neutral.txt
│   │   ├── zs-minedit.txt
│   │   └── zs-fluency.txt
├── results/                 # Model output files (auto-generated)
├── evaluation/
│   └── results.md           # Raw evaluation outputs, per model and instruction
```

---

## Setup

### 1. Environment

Python 3.9+ is required. A virtual environment is strongly recommended.

```bash
git clone https://github.com/tolgahanturker/gec-llm-eval.git
cd gec-llm-eval

python -m venv venv
source venv/bin/activate        # Linux/macOS
# .\venv\Scripts\activate       # Windows

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Python 2.7 for M2Scorer

The M2Scorer (used for CoNLL-2014 evaluation) requires **Python 2.x**. Install Python 2.7 via pyenv:

```bash
pyenv install 2.7.18
```

Then set the full path to the Python 2 interpreter in `config.yaml` (see below).

### 3. Configuration

All runtime settings live in `config.yaml` at the project root. Fill in your API credentials before running experiments:

```yaml
GENERAL:
  DEBUG: True
  ALLOWED_LLMS: [...]   # do not modify unless adding a new model

EVALUATION:
  PYTHON2_FULL_PATH_FOR_M2SCORER: "/path/to/.pyenv/versions/2.7.18/bin/python"
  M2SCORER_PATH: "./data/conll2014/m2scorer/scripts/m2scorer.py"
  GLEU_PATH: "./data/jfleg/eval/gleu.py"

OPENAI_API:
  KEY: <your-openai-api-key>

GEMINI_API:
  KEY: <your-gemini-api-key>

CLAUDE_API:
  KEY: <your-anthropic-api-key>

AZURE_API:
  KEY: <your-azure-api-key>
  ENDPOINT: <your-azure-endpoint>
```

> Mistral and LLaMA models are served via Azure AI Inference and require the `AZURE_API` block.

---

## Usage

### Step 1 — Run Inference

```bash
# python main.py <model> <data_path> <instruction_path>

# CoNLL-2014
python main.py gpt-4.1-2025-04-14 \
    "data/conll2014/official-2014.combined-withalt.m2" \
    "instructions/new/zs-neutral.txt"

# JFLEG
python main.py claude-sonnet-4-5 \
    "data/jfleg/test.src" \
    "instructions/new/zs-fluency.txt"

# W&I+LOCNESS (BEA-2019)
python main.py Mistral-Large-3 \
    "data/wandilocness/ABCN.dev.gold.bea19.m2" \
    "instructions/new/zs-minedit.txt"
```

Each run writes three files to `results/`:

| File | Format | Used for |
|------|--------|---------|
| `<model>_<data>_<instruction>_<timestamp>.jsonl` | JSON Lines | Raw responses (intermediate) |
| `<model>_<data>_<instruction>_<timestamp>.txt` | Plain text, one sentence per line | M2Scorer, GLEU |
| `<model>_<data>_<instruction>_<timestamp>.m2` | M2 format | ERRANT |

### Step 2 — Evaluate

Use the `.txt` output for M2Scorer and GLEU, and the `.m2` output for ERRANT.

```bash
# python eval.py <metric> <system_output> <gold_data>

# CoNLL-2014 → M2Scorer (F0.5)
python eval.py m2scorer \
    "results/gpt-4.1-2025-04-14_official-2014.combined-withalt.m2_zs-neutral.txt_202601010000.txt" \
    "data/conll2014/official-2014.combined-withalt.m2"

# JFLEG → GLEU
python eval.py gleu \
    "results/claude-sonnet-4-5_test.src_zs-fluency.txt_202601010000.txt" \
    "./data/jfleg/test"

# W&I+LOCNESS → ERRANT (F0.5)
python eval.py errant \
    "results/Mistral-Large-3_ABCN.dev.gold.bea19.m2_zs-minedit.txt_202601010000.m2" \
    "./data/wandilocness/ABCN.dev.gold.bea19.m2"
```

---

## Supported Models

| Model | Provider | API Block |
|-------|----------|-----------|
| `gpt-3.5-turbo` | OpenAI | `OPENAI_API` |
| `gpt-4.1-2025-04-14` | OpenAI | `OPENAI_API` |
| `gpt-5-mini-2025-08-07` | OpenAI | `OPENAI_API` |
| `claude-sonnet-4-5` | Anthropic | `CLAUDE_API` |
| `gemini-2.5-pro` | Google | `GEMINI_API` |
| `Mistral-Large-3` | Azure AI Inference | `AZURE_API` |
| `Llama-3.3-70B-Instruct` | Azure AI Inference | `AZURE_API` |
| `Llama-4-Scout-17B-16E-Instruct` | Azure AI Inference | `AZURE_API` |

---

## Instruction Modes

Three zero-shot instruction variants are evaluated. All are located in `instructions/new/`.

| File | Mode | Description |
|------|------|-------------|
| `zs-neutral.txt` | **Neutral** | Generic correction instruction with no constraints on editing style. Serves as the anchor condition. |
| `zs-minedit.txt` | **Min-Edit** | Explicitly instructs the model to apply only necessary edits, preserving the original structure and tone. |
| `zs-fluency.txt` | **Fluency** | Instructs the model to correct errors and improve fluency to match native speaker norms. |

---

## Contact

For questions or feedback, feel free to reach out:

**Tolgahan Türker** — turkert21@itu.edu.tr
