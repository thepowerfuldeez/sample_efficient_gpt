"""Generate K chat completions with vLLM for a small dataset slice.

Run with python infer_vllm_k.py
python scripts/infer_vllm_k.py \
    --limit 10 \
    --model-dir /mnt/harddrive/checkpoints/1121_main_run_mix_midtrain/hf_50000 \
    --tokenizer /mnt/harddrive/checkpoints/1121_main_run_mix_midtrain/hf_50000 \
    --data-file /home/george/sample_efficient_gpt/sample_efficient_gpt/evals/train0_filtered.jsonl
"""

from __future__ import annotations

import argparse
import time
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port

GPUs_per_dp_rank = 1
DP_size = 4


def load_chat_template(tokenizer_dir: Path) -> str | None:
    """Load the chat template if present so vLLM matches HF tokenization."""
    tok_path = Path(tokenizer_dir)
    tok = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True, trust_remote_code=False)
    if getattr(tok, "chat_template", None):
        return tok.chat_template
    template_path = tok_path / "chat_template.jinja"
    if template_path.is_file():
        return template_path.read_text()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mnt/harddrive/checkpoints/1121_main_run_mix_midtrain/hf_50000"),
        help="Path to the HF-converted checkpoint.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("/home/george/axolotl/train1_fla_select.jsonl"),
        help="JSONL dataset with messages under ['input']['messages'].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_sft_stage2"),
        help="Where to write the generated dataset.",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path("train2_k_vllm_subset.jsonl"),
        help="Where to write the generated dataset.",
    )
    parser.add_argument(
        "--start_idx", type=int, default=-1, help="Number of rows to generate (uses ds.select(range(limit)))."
    )
    parser.add_argument(
        "--end_idx", type=int, default=-1, help="Number of rows to generate (uses ds.select(range(limit)))."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Number of rows to generate (uses ds.select(range(limit)))."
    )
    parser.add_argument("--k", type=int, default=8, help="Samples to generate per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def truncate_messages(messages: list[dict], limit: int = 3456) -> list[dict]:
    c = 0
    limit_chars = limit
    new_messages: list[dict] = []

    # Process in user–assistant pairs: (0,1), (2,3), ..., last user
    for i in range(0, len(messages), 2):
        is_full_turn = i < len(messages) - 1 and messages[i + 1]["role"] == "assistant"

        if is_full_turn:
            user_msg = messages[i]
            asst_msg = messages[i + 1]
            l = len(user_msg["content"]) + len(asst_msg["content"])

            if c + l > limit_chars:
                break

            new_messages.extend((user_msg, asst_msg))
            c += l

        else:
            # Single last user message
            user_msg = messages[i]
            l = len(user_msg["content"])

            if c + l > limit_chars:
                break

            new_messages.append(user_msg)
            c += l

    # --- Ensure the last message is always a user role ---
    if new_messages and new_messages[-1]["role"] != "user":
        # Drop the last assistant if present
        if new_messages[-1]["role"] == "assistant":
            new_messages = new_messages[:-1]

    return new_messages


# def main(dp_size, dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank) -> None:
def main() -> None:
    args = parse_args()
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # os.environ["VLLM_DP_RANK"] = str(dp_rank)
    # os.environ["VLLM_DP_SIZE"] = str(dp_size)
    # os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    # os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    chat_template = load_chat_template(args.model_dir)

    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)
    if "<|assistant_end|>" in tok.get_vocab():
        eos_ids.append(tok.convert_tokens_to_ids("<|assistant_end|>"))
    eos_ids = list(dict.fromkeys(eos_ids))

    sampling_params = SamplingParams(
        n=args.k,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_token_ids=eos_ids,
    )

    llm = LLM(
        model=str(args.model_dir),
        tokenizer=str(args.model_dir),
        dtype="auto",
        max_model_len=8192,
        distributed_executor_backend="external_launcher",
        data_parallel_size=world_size,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        seed=42,
    )
    dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
    dp_size = llm.llm_engine.vllm_config.parallel_config.data_parallel_size

    ds = load_dataset("json", data_files=str(args.data_file))["train"]
    if args.limit > 0:
        ds = ds.select(range(args.limit))
    if args.start_idx != -1 and args.end_idx != -1:
        ds = ds.select(range(args.start_idx, args.end_idx))

    # promts_per_rank = len(ds) // dp_size
    # start = dp_rank * promts_per_rank
    # end = start + promts_per_rank
    # ds = ds.select(range(start, end))
    # print(f"DP rank {dp_rank} needs to process {len(ds)} prompts")

    num_problems = len(ds)
    indices = list(range(dp_rank, num_problems, dp_size))
    ds = ds.select(indices)

    conversations = []
    meta = []
    for row in ds:
        messages = truncate_messages(row["input"]["messages"], 4608 - args.max_new_tokens)
        if sum(len(m['content']) for m in messages) >= 4608 - args.max_new_tokens:
            continue
        if messages:
            conversations.append(messages)
            meta.append(
                {
                    "index": row.get("index"),
                    "input": row["input"],
                    "output": row.get("output"),
                }
            )
    print(f"DP rank {dp_rank} needs to process {len(conversations)} prompts")

    start_time = time.perf_counter()
    outputs = llm.chat(
        conversations, sampling_params=sampling_params, **({"chat_template": chat_template} if chat_template else {})
    )
    elapsed = time.perf_counter() - start_time

    results = []
    total_prompt = 0
    total_generated = 0
    for record, out in zip(meta, outputs):
        if getattr(out, "prompt_token_ids", None) is not None:
            total_prompt += len(out.prompt_token_ids)
        generated = []
        for candidate in out.outputs:
            total_generated += len(candidate.token_ids)
            generated.append(candidate.text)
        record["generated_outputs"] = generated
        results.append(record)

    out_path = Path(args.output_dir) / f"{dp_rank}_{args.prefix}"
    Dataset.from_list(results).to_json(out_path)

    total_tokens = total_prompt + total_generated
    toks_per_s = total_generated / elapsed if elapsed > 0 else 0.0
    print(
        f"Completed {len(results)} prompts "
        f"({total_generated} new tokens, {total_tokens} total) "
        f"in {elapsed:.2f}s -> {toks_per_s:.1f} toks/s"
    )


if __name__ == "__main__":
    main()
    # torch.multiprocessing.set_start_method("spawn")
    # from multiprocessing import Process

    # dp_master_ip = "127.0.0.1"
    # dp_master_port = get_open_port()
    # procs = []
    # for i in range(DP_size):
    #     proc = Process(target=main, args=(DP_size, i, dp_master_ip, dp_master_port, GPUs_per_dp_rank))
    #     proc.start()
    #     procs.append(proc)
    # exit_code = 0
    # for proc in procs:
    #     proc.join()
    #     if proc.exitcode:
    #         exit_code = proc.exitcode

    # exit(exit_code)
