# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quick HF-side sanity runner for imu_1 checkpoints.

Supports base (no chat markers) and chat-style prompts. For base checkpoints,
use BOS=<|bos|> and plain text prompts. For chat checkpoints, include markers
like <|user_start|>…<|assistant_start|>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()


def format_prompt(prompt: str, chat: bool, bos_token: str) -> str:
    if chat:
        return f"{bos_token}<|user_start|>{prompt}<|user_end|><|assistant_start|>"
    return f"{bos_token}{prompt}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--bos-token", type=str, default="", help="Use '' to disable BOS.")
    parser.add_argument(
        "--eos-token",
        type=str,
        default="<|endoftext|>",
        help="Use '' to disable EOS stopping.",
    )
    parser.add_argument("--chat", action="store_true", help="Use chat markers.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt(s) to test. If omitted, a default prompt is used.",
    )
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tok.bos_token_id = tok.convert_tokens_to_ids(args.bos_token) if args.bos_token else None
    tok.eos_token_id = tok.convert_tokens_to_ids(args.eos_token) if args.eos_token else None
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_safetensors=True,
        device_map="cpu",
    )
    if tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

    prompts = args.prompt or ["The capital of France is"]
    for p in prompts:
        formatted = format_prompt(p, chat=args.chat, bos_token=args.bos_token or "")
        input_ids = tok.encode(formatted, add_special_tokens=False, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)
        print(f"Prompt: {p}")
        print(f"Encoded ids: {input_ids.tolist()}")

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tok.eos_token_id,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        decoded = tok.decode(outputs[0], skip_special_tokens=False)
        print(f"Generated: {decoded}")
        print("---")


if __name__ == "__main__":
    main()
