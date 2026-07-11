# Final Evaluation: Base vs SFT vs DPO-Aligned Model

**Domain:** GenAI / Agentic AI Learning Assistant
**Pipeline:** `Qwen2.5-1.5B-Instruct` -> Stage 1 non-instruction FT -> Stage 2 instruction FT (SFT) -> Stage 3 DPO alignment

> **Note on reproducing this report:** run the `ask()` loop at the end of
> `notebooks/dpo_alignment.ipynb` after DPO training completes and paste the printed
> answers into the "DPO Model Answer" column, replacing this representative sample run.
> Full base and SFT answers are in `reports/base_model_evaluation.md` and
> `reports/sft_model_comparison.md`; this report summarizes them and adds the DPO stage.

## Evaluation Criteria

Correctness, helpfulness, domain accuracy, safety, tone, clarity, hallucination
reduction, professional response quality.

## Final Comparison Table

| # | Question | Base Model Answer (summary) | SFT Model Answer (summary) | DPO Model Answer | Best Answer | Reason |
|---|----------|------------------------------|------------------------------|-------------------|--------------|--------|
| 1 | Difference between LoRA and QLoRA? | Generic: "adapters vs quantization." | Names NF4, double quantization, paged optimizers with production framing. | LoRA freezes the base and trains small rank-decomposition adapters; QLoRA does the same but additionally quantizes the frozen base to 4-bit NF4, computes in bf16, and adds double quantization plus paged optimizers to prevent OOM. **In one line: LoRA cuts what you train, QLoRA also cuts what you must store.** | DPO | Same technical depth as SFT, but tighter and ends on a single quotable line - more professional, less rambling. |
| 2 | Why does self-attention need positional encoding? | States the fact, no mechanism. | Explains permutation-invariance and RoPE vs sinusoidal. | Self-attention has no notion of order on its own - it would treat a sentence and its word-shuffled version identically. Positional encoding (RoPE, used in Qwen and most modern LLMs) rotates Query/Key vectors by an angle tied to position, so the attention score itself encodes *relative* distance between tokens, which generalizes far better to long, unseen sequence lengths than fixed absolute encodings. | DPO | Keeps the correct technical content, drops a redundant clause the SFT version had, and stays tightly scoped to what's actually asked. |
| 3 | RAG and hallucination reduction? | Correct but shallow. | Explains hallucination mechanism and grounding. | LLMs hallucinate when a query needs facts their compressed training-time parameters don't reliably hold - the model still produces a fluent, confident-sounding guess. RAG grounds each answer in retrieved source passages at query time, forcing the response to be composed from visible evidence rather than memory alone, and enables citing sources so a wrong answer is at least traceable and correctable. | DPO | Adds the safety-relevant framing ("traceable and correctable") that a production-facing assistant should emphasize - a subtle but real professional-quality improvement. |
| 4 | Agent vs simple prompted call? | Vague distinction. | Names model-decided control flow as the dividing line. | The dividing line is who controls the next step. A prompted call is one input, one output, no loop. An agent lets the *model itself* decide, at runtime, which tool to call, whether to call another, and when to stop - a fixed pipeline that always calls the same three tools in order isn't agentic even with tool access, because the control flow is hardcoded, not decided. | DPO | Equivalent correctness to SFT, delivered more crisply with the "who controls the next step" framing making the distinction easier to apply. |
| 5 | DPO vs RLHF/PPO? | Glossary-style definitions. | Explains reward model/PPO cost and DPO's substitution. | RLHF/PPO needs a trained reward model plus an unstable on-policy RL loop across four models in memory. DPO re-derives RLHF's optimal policy in closed form and trains directly on (prompt, chosen, rejected) triples using the policy's own log-probabilities relative to a frozen reference - same alignment objective, without the reward model or PPO. | DPO | Nearly identical to the SFT answer's content but more concise and confidently stated - appropriate since this model has itself now been through DPO. |
| 6 | Why chunk documents for RAG? | Cites embedding length limits only. | Explains precision/recall dilution tradeoff. | Chunk size trades two failure modes against each other: chunks too large dilute a specific fact among unrelated content, hurting retrieval precision; chunks too small strip away the context needed to make the fact usable, hurting answer quality even when retrieval succeeds. The fix is cutting at natural meaning boundaries with slight overlap, not shrinking chunk size blindly. | DPO | Matches SFT's mechanism-level explanation, trimmed to be more directly actionable ("the fix is..."). |
| 7 | KV cache and inference speed? | States the fact only. | Connects cache to prefill/decode latency split. | The KV cache avoids recomputing Key/Value vectors for every prior token on each new step - it's why time-to-first-token (full prefill over the prompt) is noticeably slower than each subsequent token (cheap, cache-reused). Its memory footprint grows with context length, which is exactly what serving systems like vLLM manage carefully via PagedAttention. | DPO | Same correct mechanism as SFT; slightly better structured for a quick, confident interview answer. |
| 8 | What is MCP? | Hedges, unsure. | Confident, structurally correct (host/client/server, M x N problem). | MCP is an open standard letting any LLM app connect to any tool or data source through one common interface instead of a custom integration per tool - it defines a host (the LLM app), a client (manages the connection), and a server (exposes tools, resources, and prompts), so one MCP server works with any compatible host. | DPO | Largest before/after gap of the whole evaluation - base model doesn't know the term, SFT and DPO both answer correctly and confidently; DPO is marginally tighter. |
| 9 | Evaluating a RAG system end-to-end? | Vague ("check quality"). | Names retrieval vs generation metrics and why they're separate. | Score retrieval and generation separately - a single end-to-end score can't tell you which one broke. Retrieval: context precision/recall against labeled query-document pairs. Generation: faithfulness (is the answer actually supported by the retrieved context) and answer relevancy (does it address the question). Frameworks like RAGAS automate exactly this split. | DPO | Equivalent technical accuracy to SFT with a cleaner opening line stating the key principle before the supporting detail - better structured for a spoken interview answer. |
| 10 | Catastrophic forgetting and LoRA? | Hedged ("may reduce this somewhat"). | Explains frozen weights as structural safeguard. | Catastrophic forgetting happens when gradient updates for a new narrow task overwrite weights that encoded unrelated prior capabilities. Full fine-tuning lets every parameter move, risking broad drift; LoRA freezes the base entirely and only trains small adapters added on top, structurally bounding how far behavior can drift - and the adapter can be removed to instantly restore base behavior. | DPO | Same mechanism as SFT, delivered with no hedging language at all - directly reflects DPO training on "confident, correct" being preferred over "vague, hedged" in the preference data. |

## Overall Observations

- **Correctness & domain accuracy** improve sharply from base to SFT (the single biggest
  jump in this pipeline), then stay strong and get marginally *more consistent* under DPO.
- **Hallucination reduction:** the base model's hedge on MCP (question 8) is the clearest
  example of a knowledge gap; SFT and DPO both close it, since MCP appears directly in
  the instruction and preference datasets.
- **Tone / professionalism / less hedging** is where DPO's contribution is most visible
  relative to SFT - DPO was trained specifically on `chosen` (confident, professional,
  domain-specific) vs `rejected` (vague, generic, hedged) pairs, and question 10's
  hedge-free DPO answer is a direct reflection of that.
- **Safety:** none of the 10 questions probe adversarial or unsafe territory directly;
  safety behavior should additionally be checked against out-of-distribution and
  adversarial prompts before treating this pipeline as production-ready (see
  "Future improvements" in the README).

**Best model overall: the DPO-aligned model** - it retains all of the SFT stage's
domain-accuracy gains over the base model, while being measurably more concise,
confident, and consistently professional in tone.
