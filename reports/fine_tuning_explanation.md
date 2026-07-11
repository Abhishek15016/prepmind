# Fine-Tuning Explanation: LoRA, QLoRA, Non-Instruction FT, SFT, and DPO

This note explains, in plain language, every technique used in this project's three-stage
pipeline (`Base -> Non-Instruction FT -> Instruction FT (SFT) -> DPO`), plus the exact
hyperparameters used for each stage.

## Why is full fine-tuning expensive?

Full fine-tuning updates every single parameter in the model. For a 1.5B-parameter model
that means storing and updating 1.5 billion weights, plus (for Adam-family optimizers) two
extra optimizer states per weight - roughly 4x the model's own size in optimizer memory
alone, on top of activations and gradients. On larger models (7B, 13B, 70B) this quickly
requires multiple high-end GPUs or a full training cluster, which is out of reach for most
individual projects and even many companies for routine fine-tuning experiments. It's also
riskier: with every weight free to move, a small or narrow dataset can drag the model far
from its original, broadly capable state (see "catastrophic forgetting" below).

## What does LoRA do?

LoRA (Low-Rank Adaptation) freezes 100% of the original model weights and instead injects
small trainable "adapter" matrices into specific layers (typically the attention
projections and MLP layers). Each adapter is a pair of low-rank matrices `A` and `B` whose
product approximates the *change* the model would need to specialize for the new task,
rather than storing a full weight update. Because `A` and `B` are tiny compared to the
original weight matrix (rank 16-32 vs. a full dense matrix of thousands of dimensions),
the number of trainable parameters drops from billions to a few million - enough to
fine-tune a 7B+ model on a single consumer or free-tier GPU, in minutes to hours instead
of requiring a cluster.

## What does QLoRA do, and why is it useful on a limited GPU?

QLoRA combines LoRA with quantization of the *frozen base model*. The base weights are
loaded in 4-bit precision (NF4 - a data type whose quantization levels are chosen to match
the normal distribution neural network weights actually follow), cutting the base model's
memory footprint roughly 4x compared to fp16/bf16. The LoRA adapters themselves are still
trained in higher precision (bf16), and during each forward pass the 4-bit weights are
dequantized on the fly for computation - only the small adapters receive gradient updates,
so the precision loss from 4-bit storage barely affects final quality. Two further tricks
make this practical: **double quantization** (quantizing the quantization constants
themselves for extra memory savings) and **paged optimizers** (spilling optimizer state to
CPU RAM during memory spikes instead of crashing). Together, these are exactly what let
this project fine-tune `Qwen2.5-1.5B-Instruct` on a single free Google Colab T4 GPU (16GB
VRAM) - without QLoRA, even a comfortably small 1.5B model in fp16 plus optimizer state
and activations would run uncomfortably close to that GPU's memory ceiling.

## What is non-instruction fine-tuning?

Non-instruction fine-tuning (Stage 1, `notebooks/non_instruction_finetuning.ipynb`) is
continued causal-language-model pretraining on raw, unstructured domain text - in this
project, 50+ paragraphs of GenAI/Agentic AI explanations with no question/answer
structure. The model is simply trained to predict the next token, the same objective used
in original pretraining, just on domain-specific text instead of general web text. The
goal is **not** to teach the model to follow instructions - it's to shift the model's
internal language distribution toward domain vocabulary, phrasing, and background
knowledge, so that the instruction-tuning stage that follows has less distance to cover.

## What is instruction fine-tuning?

Instruction fine-tuning (Stage 2, `notebooks/instruction_finetuning.ipynb`) is supervised
fine-tuning (SFT) on `(instruction, response)` or multi-turn `(system, user, assistant)`
examples - in this project, 100+ GenAI/Agentic AI question-answer pairs. Unlike Stage 1,
the training objective is still next-token prediction, but the *data* now explicitly
demonstrates the desired behavior: given a question, produce a specific kind of answer.
This is what turns a text-completion model into something that reliably follows
instructions and answers questions in a consistent persona/style, rather than just
continuing text plausibly.

## What is DPO?

DPO (Direct Preference Optimization, Stage 3, `notebooks/dpo_alignment.ipynb`) further
aligns the SFT model using preference data - pairs of `(chosen, rejected)` responses to
the same prompt, where `chosen` is correct/helpful/safe/professional and `rejected` is
wrong/generic/unsafe/rude. Instead of training a separate reward model and running
unstable on-policy reinforcement learning (as classic RLHF/PPO does), DPO re-derives the
RLHF objective's closed-form optimal policy and trains the policy model directly on the
preference pairs, using its own log-probabilities relative to a frozen reference model.
The practical effect: the model is pushed to increase the probability of chosen-style
responses and decrease the probability of rejected-style responses, in relative terms
anchored against what it already believed before DPO started.

## Difference between SFT and DPO

| | SFT | DPO |
|---|---|---|
| Data shape | `(prompt, response)` - one correct answer per example | `(prompt, chosen, rejected)` - two responses per example, one better |
| What it teaches | *How to answer at all* - format, persona, task-following | *Which of two plausible answers is better* - quality, tone, safety, confidence |
| Objective | Maximize likelihood of the target response | Increase relative probability of chosen vs rejected response |
| Typical use | Turn a base/pretrained model into an instruction-follower | Polish an already-instruction-tuned model's judgment and tone |
| Failure mode if skipped | Model may ignore instructions or ramble like a raw LM | Model may follow instructions but stay generic, hedge, or occasionally give a technically-plausible-but-worse answer |

In short: **SFT teaches capability, DPO refines judgment.** This project runs them in that
order deliberately - DPO needs a model that can already produce reasonable answers before
it can usefully learn to prefer the better one.

## Hyperparameters used in this project

| Stage | Rank (r) | Alpha | Dropout | Learning Rate | Batch Size (effective) | Epochs | Notes |
|---|---|---|---|---|---|---|---|
| 1. Non-instruction FT | 16 | 16 | 0.05 | 1e-4 | 2 x 4 grad-accum = 8 | 2 | Small rank/short run - goal is gentle domain adaptation, not memorization |
| 2. Instruction FT (SFT) | 32 | 32 | 0.05 | 2e-4 | 2 x 4 grad-accum = 8 | 3 | Higher rank - teaching a new behavior (instruction-following) needs more adapter capacity |
| 3. DPO alignment | 16 | 16 | 0.0 | 5e-6 | 2 x 4 grad-accum = 8 | 2 | Much lower LR (typical for DPO) and beta=0.1 to avoid over-correcting away from the SFT model |

`lora_alpha` is set equal to `r` in every stage (a common default that keeps the
effective adapter scaling factor - `alpha / r` - at 1.0); dropout is disabled for DPO
since preference training benefits from a more deterministic policy during optimization.
All three stages target the same LoRA modules: `q_proj, k_proj, v_proj, o_proj, gate_proj,
up_proj, down_proj`.
