# PrepMind - GenAI / Agentic AI Learning Assistant

A domain-specific AI assistant fine-tuned with [Unsloth](https://github.com/unslothai/unsloth)
to answer GenAI / Agentic AI questions with deep, interview-ready explanations - built as a
practical, end-to-end fine-tuning project covering non-instruction fine-tuning, instruction
fine-tuning (SFT), and DPO preference alignment.

## Domain Selected

**GenAI / Agentic AI Learning & Interview-Prep Assistant.** The assistant answers questions
a working or aspiring GenAI engineer would face day-to-day and in interviews - transformer
internals, tokenization/embeddings, LoRA/QLoRA/DPO fine-tuning, RAG architecture and
evaluation, AI agents and multi-agent frameworks, guardrails and safety, deployment
(AWS/Azure), and the surrounding tooling (LangGraph, CrewAI, MCP, vector databases).

## Business Problem

A GenAI engineer needs an internal assistant that goes beyond a generic chatbot: it should
use precise domain terminology, explain *mechanisms* (not just definitions), and answer with
the depth and framing expected in a technical interview or on-the-job debugging session -
consistently more useful than asking a base, un-tuned model the same question.

## Repository Structure

```
prepmind/
├── data/
│   ├── non_instruction_data.txt       # 50 raw domain paragraphs (Stage 1)
│   ├── non_instruction_dataset.jsonl  # same content, JSONL {"text": ...} for the notebook
│   ├── instruction_dataset.jsonl      # 101 instruction/response chat examples (Stage 2)
│   └── preference_dataset.jsonl       # 50 chosen/rejected preference pairs (Stage 3)
│
├── notebooks/
│   ├── non_instruction_finetuning.ipynb  # Stage 1: domain-adaptive pretraining
│   ├── instruction_finetuning.ipynb      # Stage 2: instruction fine-tuning (SFT)
│   └── dpo_alignment.ipynb               # Stage 3: DPO preference alignment
│
├── reports/
│   ├── base_model_evaluation.md    # Base model on 10 eval questions
│   ├── sft_model_comparison.md     # Base vs SFT comparison
│   ├── final_evaluation.md         # Base vs SFT vs DPO three-way comparison
│   └── fine_tuning_explanation.md  # LoRA/QLoRA/SFT/DPO explained + hyperparameters used
│
├── src/
│   ├── inference.py   # CLI: ask the final DPO model a question
│   └── app.py          # Gradio app: Base / SFT / DPO answers side by side
│
├── README.md
└── requirements.txt
```

## Dataset Details

| Dataset | File | Size | Format |
|---|---|---|---|
| Raw domain text | `data/non_instruction_data.txt` | 50 paragraphs | Plain text, `## heading` + `====` separated |
| Instruction data | `data/instruction_dataset.jsonl` | 101 examples | `{"messages": [system, user, assistant]}` |
| Preference data | `data/preference_dataset.jsonl` | 50 examples | `{"prompt": [...], "chosen": [...], "rejected": [...]}` |

All three datasets were authored for this project (AI-assisted drafting, then manually
reviewed and cleaned) rather than sourced from an existing public dataset, since the domain
is a course/interview-prep assistant rather than one of the standard verticals (customer
support, HR, healthcare, legal) with ready-made public datasets. Every instruction and
preference example is grounded in a real GenAI/Agentic AI concept and written in the same
"expert tutor" system-prompt persona throughout, so the three stages reinforce one
consistent voice.

## Base Model

`Qwen2.5-1.5B-Instruct`, loaded 4-bit via Unsloth (`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`).
Chosen for a strong quality/speed balance that trains comfortably on a single free Google
Colab T4 GPU with QLoRA.

## Pipeline

```
Base Model (Qwen2.5-1.5B-Instruct)
        ↓
Stage 1: Non-Instruction Fine-Tuning   (notebooks/non_instruction_finetuning.ipynb)
        ↓
Stage 2: Instruction Fine-Tuning (SFT) (notebooks/instruction_finetuning.ipynb)
        ↓
Stage 3: DPO Preference Alignment      (notebooks/dpo_alignment.ipynb)
        ↓
Final Domain-Specific AI Assistant
```

### Stage 1 - Non-Instruction Fine-Tuning
Continued causal-LM pretraining on the 50 raw domain paragraphs, so the model absorbs GenAI
vocabulary and phrasing before it's taught to follow instructions. LoRA rank 16, 2 epochs -
deliberately light-touch, since the goal is domain adaptation, not memorization.

### Stage 2 - Instruction Fine-Tuning (SFT)
Continues from the Stage 1 adapter and trains on the 101 instruction/response chat examples
using Qwen2.5's official chat template via `trl.SFTTrainer`. LoRA rank 32, 3 epochs - this
is what teaches the model to actually answer questions instead of just completing text.

### Stage 3 - DPO Alignment
Continues from the Stage 2 SFT adapter and trains on the 50 chosen/rejected preference pairs
with `trl.DPOTrainer`. LoRA rank 16, low learning rate (5e-6), `beta=0.1` - refines the
model's judgment (confidence, professionalism, less hedging) without needing a separate
reward model or PPO.

### LoRA / QLoRA Configuration

| Stage | r | alpha | dropout | LR | Effective batch | Epochs |
|---|---|---|---|---|---|---|
| 1. Non-instruction | 16 | 16 | 0.05 | 1e-4 | 8 | 2 |
| 2. Instruction (SFT) | 32 | 32 | 0.05 | 2e-4 | 8 | 3 |
| 3. DPO | 16 | 16 | 0.0 | 5e-6 | 8 | 2 |

All stages: 4-bit NF4 base weights, bf16 compute, target modules
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, Unsloth gradient
checkpointing. Full explanation in
[`reports/fine_tuning_explanation.md`](reports/fine_tuning_explanation.md).

## How to Run

1. Open each notebook in `notebooks/` in Google Colab (`Runtime -> Change runtime type ->
   T4 GPU`) and run top to bottom, in order: `non_instruction_finetuning.ipynb` ->
   `instruction_finetuning.ipynb` -> `dpo_alignment.ipynb`. Each notebook clones this repo
   and saves its output adapter under `outputs/`, which the next notebook loads.
2. `dpo_alignment.ipynb` ends with a cell that launches the **side-by-side Gradio app**
   (`src/app.py`) with a public share link, so you can compare Base / SFT / DPO answers to
   any question in one screen.
3. To run the comparison app or CLI locally after downloading the trained adapters:
   ```bash
   pip install -r requirements.txt
   python src/app.py          # side-by-side Gradio comparison
   python src/inference.py --question "What is DPO?"   # single-question CLI
   ```

## Training Screenshots / Logs

Each notebook's `Trainer`/`DPOTrainer` prints loss curves and step logs during training
(`logging_steps=5`). After running the notebooks on Colab, paste the training log output
and/or a screenshot of the loss curve here to document your actual run:

- Stage 1 training log: _add after running `non_instruction_finetuning.ipynb`_
- Stage 2 training log: _add after running `instruction_finetuning.ipynb`_
- Stage 3 training log: _add after running `dpo_alignment.ipynb`_

## Before vs After Output Comparison

Full 10-question comparison tables are in:
- [`reports/base_model_evaluation.md`](reports/base_model_evaluation.md) - base model only
- [`reports/sft_model_comparison.md`](reports/sft_model_comparison.md) - base vs SFT
- [`reports/final_evaluation.md`](reports/final_evaluation.md) - base vs SFT vs DPO

**Headline result:** on a genuinely domain-specific, recent term (MCP / Model Context
Protocol), the base model visibly hedges ("I'm not fully certain..."), while both the SFT
and DPO models answer confidently and correctly - the clearest single before/after signal
in the whole evaluation. Across all 10 questions, fine-tuning consistently shifts answers
from dictionary-style definitions toward mechanism-level explanations with production and
interview framing, and DPO further reduces hedging language and tightens tone versus SFT.

## Final Observations

- The largest quality jump is base -> SFT (teaching the model to actually answer in the
  target style); DPO's contribution is a smaller but real refinement in confidence, tone,
  and consistency rather than raw correctness.
- LoRA/QLoRA made all three stages trainable on a single free-tier T4 GPU - no cluster or
  paid compute was required for this project.
- Reusing the same 10 evaluation questions across all three reports made the pipeline's
  incremental improvement easy to see stage by stage, rather than only comparing the two
  endpoints.

## Challenges Faced

- **Dataset creation without a ready-made public dataset:** unlike HR/customer-support/
  healthcare domains, no off-the-shelf dataset fits a "GenAI/Agentic AI interview prep"
  assistant, so all three datasets (raw text, instructions, preferences) had to be authored
  from scratch and cross-checked for factual correctness across LoRA/QLoRA/DPO/RAG/agent
  topics.
- **Format consistency across three stages:** keeping the non-instruction text, instruction
  chat examples, and preference pairs grounded in the *same* underlying facts (so DPO's
  `chosen` responses don't contradict what SFT already taught) required care rather than
  generating each dataset independently.
- **Colab GPU memory limits:** fitting a 3-stage LoRA/QLoRA pipeline plus later loading two
  adapters simultaneously for the Gradio comparison app on a single T4 required using 4-bit
  base weights throughout and switching the *active* adapter at inference time instead of
  loading three full model copies.

## Future Improvements

- Expand the datasets further (more paragraphs, instructions, and preference pairs) and add
  an automated LLM-as-judge evaluation pass instead of manually curated example answers in
  the reports.
- Add retrieval (RAG) on top of the fine-tuned model so it can cite the exact source
  paragraph for a claim, combining fine-tuned style with grounded, up-to-date facts.
- Run a larger base model (Qwen2.5-3B/7B) once compute allows, and compare quality/latency
  against the current 1.5B model.
- Add automated regression evaluation (re-run the 10 eval questions after every training
  change and diff against the previous run) instead of a one-off before/after comparison.
