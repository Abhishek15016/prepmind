"""
Simple inference script for the final DPO-aligned PrepMind GenAI/Agentic AI assistant.

The training notebooks (see notebooks/) push every stage's model to the Hugging Face Hub
rather than only saving to Colab's local disk, since local Colab storage is wiped between
sessions. By default this script loads the final merged DPO model from the Hub repo
<HF_USERNAME>/prepmind-dpo-qwen2.5-1.5b - set HF_USERNAME below (or via the
PREPMIND_HF_USERNAME env var) to the same username used in the notebooks.

Usage:
    export PREPMIND_HF_USERNAME=your-hf-username   # or edit HF_USERNAME below
    export HF_TOKEN=hf_xxx                          # only needed if the Hub repo is private
    python src/inference.py
    python src/inference.py --question "What is DPO?"
    python src/inference.py --model_path outputs/dpo_merged   # load a local checkpoint instead

With no --question, drops into an interactive loop. Works with either a merged model
directory/repo, or a base model + LoRA adapter directory (pass --adapter_path).
"""

import argparse
import os

# Same username you set as HF_USERNAME in the notebooks - update this once your models are pushed.
HF_USERNAME = os.environ.get("PREPMIND_HF_USERNAME", "abhishek15016")
DEFAULT_MODEL_PATH = f"{HF_USERNAME}/prepmind-dpo-qwen2.5-1.5b"

HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed if the Hub repo is private

SYSTEM_PROMPT = (
    "You are a friendly expert tutor in Generative AI and Agentic AI. Explain concepts "
    "in depth with intuitive analogies, practical production insight, and interview-ready "
    "takeaways, in a clear and engaging style."
)

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"


def load_model(model_path: str, adapter_path: str | None, max_seq_length: int = 2048):
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            token=HF_TOKEN,
        )
        if adapter_path:
            model.load_adapter(adapter_path, token=HF_TOKEN)
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, token=HF_TOKEN)
        model.eval()
        return model, tokenizer


def generate_answer(question: str, model, tokenizer, max_new_tokens: int = 400) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_ids = output[0][inputs.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="PrepMind GenAI/Agentic AI assistant inference")
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Path or HF Hub repo id of the final (merged) DPO model, or the base model if using --adapter_path",
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Optional path/repo id of a LoRA adapter to load on top of --model_path",
    )
    parser.add_argument("--question", default=None, help="Ask a single question and exit")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model(args.model_path, args.adapter_path)
    print("Model loaded.\n")

    if args.question:
        answer = generate_answer(args.question, model, tokenizer, args.max_new_tokens)
        print(f"Q: {args.question}\nA: {answer}")
        return

    print("Interactive mode - type a question, or 'quit' to exit.\n")
    while True:
        question = input("question = ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        answer = generate_answer(question, model, tokenizer, args.max_new_tokens)
        print(f"answer = {answer}\n")


if __name__ == "__main__":
    main()
