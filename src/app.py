"""
Gradio demo: ask one question, see the Base model, the SFT (instruction fine-tuned) model,
and the DPO-aligned model answer it side by side, all at once.

Run after training all three stages (see notebooks/). By default this loads the SFT and
DPO adapters from the Hugging Face Hub repos the notebooks push to
(<HF_USERNAME>/prepmind-sft-adapter and <HF_USERNAME>/prepmind-dpo-adapter) rather than
local outputs/, since local Colab storage doesn't survive between sessions. Set
HF_USERNAME below (or the PREPMIND_HF_USERNAME env var) to the same username used in the
notebooks.

Usage:
    export PREPMIND_HF_USERNAME=your-hf-username   # or edit HF_USERNAME below
    export HF_TOKEN=hf_xxx                          # only needed if the Hub repos are private
    python src/app.py
    python src/app.py --sft_adapter outputs/sft_adapter --dpo_adapter outputs/dpo_adapter  # local checkpoints instead
"""

import argparse
import os

import gradio as gr

# Same username you set as HF_USERNAME in the notebooks - update this once your models are pushed.
HF_USERNAME = os.environ.get("PREPMIND_HF_USERNAME", "abhishek15016")
DEFAULT_SFT_ADAPTER = f"{HF_USERNAME}/prepmind-sft-adapter"
DEFAULT_DPO_ADAPTER = f"{HF_USERNAME}/prepmind-dpo-adapter"

HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed if the Hub repos are private

SYSTEM_PROMPT = (
    "You are a friendly expert tutor in Generative AI and Agentic AI. Explain concepts "
    "in depth with intuitive analogies, practical production insight, and interview-ready "
    "takeaways, in a clear and engaging style."
)

EXAMPLE_QUESTIONS = [
    "What is the difference between LoRA and QLoRA?",
    "What is Retrieval-Augmented Generation and why does it reduce hallucination?",
    "What is DPO, and how is it different from RLHF with PPO?",
    "What is Model Context Protocol (MCP) and what problem does it solve?",
]


def load_models(base_model: str, sft_adapter: str, dpo_adapter: str):
    """Loads the base model once and attaches both adapters, so generation only needs
    to switch the active adapter instead of loading three separate model copies."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    has_sft, has_dpo = False, False
    try:
        model.load_adapter(sft_adapter, adapter_name="sft", token=HF_TOKEN)
        has_sft = True
    except Exception as e:
        print(f"Could not load SFT adapter from {sft_adapter}: {e}")

    try:
        model.load_adapter(dpo_adapter, adapter_name="dpo", token=HF_TOKEN)
        has_dpo = True
    except Exception as e:
        print(f"Could not load DPO adapter from {dpo_adapter}: {e}")

    FastLanguageModel.for_inference(model)
    return model, tokenizer, has_sft, has_dpo


def build_prompt(tokenizer, question: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )


def generate_with_adapter(model, tokenizer, question: str, adapter_name: str | None, max_new_tokens: int = 300) -> str:
    inputs = build_prompt(tokenizer, question).to(model.device)

    if adapter_name is None:
        # base model behavior: temporarily disable all adapters
        with model.disable_adapter():
            output = model.generate(
                input_ids=inputs, max_new_tokens=max_new_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        model.set_adapter(adapter_name)
        output = model.generate(
            input_ids=inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = output[0][inputs.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Base vs SFT vs DPO side-by-side comparison")
    parser.add_argument("--base_model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--sft_adapter", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--dpo_adapter", default=DEFAULT_DPO_ADAPTER)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link (useful on Colab)")
    args = parser.parse_args()

    print("Loading model + adapters, this can take a minute...")
    model, tokenizer, has_sft, has_dpo = load_models(args.base_model, args.sft_adapter, args.dpo_adapter)
    print(f"Ready. SFT adapter loaded: {has_sft} | DPO adapter loaded: {has_dpo}")

    def compare(question: str):
        if not question.strip():
            return "", "", ""
        base_answer = generate_with_adapter(model, tokenizer, question, None)
        sft_answer = generate_with_adapter(model, tokenizer, question, "sft") if has_sft else "(SFT adapter not found - train Stage 2 first)"
        dpo_answer = generate_with_adapter(model, tokenizer, question, "dpo") if has_dpo else "(DPO adapter not found - train Stage 3 first)"
        return base_answer, sft_answer, dpo_answer

    with gr.Blocks(title="PrepMind - GenAI/Agentic AI Assistant: Base vs SFT vs DPO") as demo:
        gr.Markdown(
            "# PrepMind - GenAI / Agentic AI Learning Assistant\n"
            "Ask one question and compare the **Base model**, the **Instruction Fine-Tuned "
            "(SFT) model**, and the **DPO-aligned model** side by side, all at once."
        )
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What is the difference between LoRA and QLoRA?",
            lines=2,
        )
        ask_btn = gr.Button("Compare all three models", variant="primary")
        gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_box)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Model\n*Qwen2.5-1.5B-Instruct, no fine-tuning*")
                base_out = gr.Textbox(label="Base answer", lines=16, interactive=False)
            with gr.Column():
                gr.Markdown("### SFT Model\n*Instruction fine-tuned on domain Q&A*")
                sft_out = gr.Textbox(label="SFT answer", lines=16, interactive=False)
            with gr.Column():
                gr.Markdown("### DPO Model\n*Preference-aligned final assistant*")
                dpo_out = gr.Textbox(label="DPO answer", lines=16, interactive=False)

        ask_btn.click(fn=compare, inputs=question_box, outputs=[base_out, sft_out, dpo_out])
        question_box.submit(fn=compare, inputs=question_box, outputs=[base_out, sft_out, dpo_out])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
