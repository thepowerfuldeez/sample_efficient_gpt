"""
Evaluate chat models using a Hugging Face checkpoint served through vLLM.

Recommended: run with torchrun for data-parallel inference, e.g.:
torchrun --nproc-per-node=4 -m sample_efficient_gpt.evals.chat_eval --checkpoint /path/to/hf_checkpoint
"""

import argparse
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import get_world_group

from sample_efficient_gpt.evals.tasks.arc import ARC
from sample_efficient_gpt.evals.tasks.gsm8k import GSM8K
from sample_efficient_gpt.evals.tasks.humaneval import HumanEval
from sample_efficient_gpt.evals.tasks.mmlu import MMLU


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


def run_generative_eval(
    task_object,
    llm: LLM,
    sampling_params: SamplingParams,
    chat_template: str | None,
    batch_size: int,
    rank: int,
    world_size: int,
    reduce_device,
    max_problems: int | None = None,
    log_file: None | Path = None,
):
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_passed = 0
    total = 0
    indices = list(range(rank, num_problems, world_size))

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        conv_batch = []
        meta = []
        for local_idx in range(start, end):
            idx = indices[local_idx]
            conversation = task_object[idx]
            conv_batch.append(conversation["messages"][:-1])  # drop the gold response
            meta.append((idx, conversation))

        outputs = llm.chat(
            conv_batch,
            sampling_params=sampling_params,
            **({"chat_template": chat_template} if chat_template else {}),
        )

        for (idx, conversation), out in zip(meta, outputs):
            completions = [candidate.text for candidate in out.outputs]
            outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
            passed = any(outcomes)
            total += 1
            num_passed += int(passed)

            if log_file is not None:
                Path(log_file).parent.mkdir(exist_ok=True, parents=True)
                with Path(log_file).open("a") as f:
                    f.write(f"{task_object},{idx},{completions[0]},{outcomes[0]}")
                    f.write("\n")

        print(
            f"\r\033[KRank {rank} processed {total}/{len(indices)} ({100 * num_passed / total:.2f}%)",
            end="",
            flush=True,
        )

    print()

    num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=reduce_device)
    total_tensor = torch.tensor([total], dtype=torch.long, device=reduce_device)
    if dist.is_initialized():
        world_group = get_world_group()
        group = world_group.device_group if reduce_device.type == "cuda" else world_group.cpu_group
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM, group=group)
    num_passed = num_passed_tensor.item()
    total = total_tensor.item()

    if rank == 0:
        print("=" * 50)
        print(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")
    return num_passed / total if total > 0 else 0.0


def run_categorical_eval(
    task_object,
    llm: LLM,
    sampling_params: SamplingParams,
    chat_template: str | None,
    batch_size: int,
    rank: int,
    world_size: int,
    reduce_device,
    max_problems=None,
    log_file: None | Path = None,
):
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    indices = list(range(rank, num_problems, world_size))
    num_passed = 0
    total = 0

    def pick_letter(text, letters):
        for ch in text.strip():
            upper = ch.upper()
            if upper in letters:
                return upper
        return None

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        conversations = [task_object[indices[ii]] for ii in range(start, end)]
        prompts = [conversation["messages"][:-1] for conversation in conversations]

        outputs = llm.chat(
            prompts,
            sampling_params=sampling_params,
            **({"chat_template": chat_template} if chat_template else {}),
        )

        for idx, (conversation, out) in enumerate(zip(conversations, outputs)):
            letters = conversation["letters"]
            generated = out.outputs[0].text
            predicted_letter = pick_letter(generated, letters)
            outcome = task_object.evaluate(conversation, predicted_letter) if predicted_letter else False

            if log_file is not None:
                Path(log_file).parent.mkdir(exist_ok=True, parents=True)
                with Path(log_file).open("a") as f:
                    f.write(f"{task_object},{indices[start + idx]},{predicted_letter},{outcome}")
                    f.write("\n")

            num_passed += int(outcome)
            total += 1

        print(
            f"\r\033[KRank {rank} processed {total}/{len(indices)} ({100 * num_passed / total:.2f}%)",
            end="",
            flush=True,
        )

    print()

    num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=reduce_device)
    total_tensor = torch.tensor([total], dtype=torch.long, device=reduce_device)
    if dist.is_initialized():
        world_group = get_world_group()
        group = world_group.device_group if reduce_device.type == "cuda" else world_group.cpu_group
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM, group=group)
    num_passed = num_passed_tensor.item()
    total = total_tensor.item()

    average = num_passed / total if total > 0 else 0.0
    if rank == 0:
        print(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


def run_chat_eval(
    task_name,
    tokenizer,
    llm,
    chat_template,
    rank,
    world_size,
    reduce_device,
    batch_size=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    num_samples=1,
    max_problems=None,
    log_file=None,
):
    task_module = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test", few_shot=False),
        "GSM8K_5shot": partial(GSM8K, subset="main", split="test", few_shot=True),
        "GSM8K_5shot_cot": partial(GSM8K, subset="main", split="test", few_shot=True, use_llama_system_prompt=True),
    }[task_name]
    task_object = task_module()

    eos_list = tokenizer.encode("<|endoftext|>") + tokenizer.encode("<|assistant_end|>")

    if task_object.eval_type == "generative":
        sampling_params = SamplingParams(
            n=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token_ids=eos_list,
        )
        acc = run_generative_eval(
            task_object,
            llm,
            sampling_params,
            chat_template,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            reduce_device=reduce_device,
            max_problems=max_problems,
            log_file=log_file,
        )
    elif task_object.eval_type == "categorical":
        sampling_params = SamplingParams(
            n=1,
            max_tokens=8,
            temperature=0.0,
            top_k=top_k,
            stop_token_ids=eos_list,
        )
        acc = run_categorical_eval(
            task_object,
            llm,
            sampling_params,
            chat_template,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            reduce_device=reduce_device,
            max_problems=max_problems,
            log_file=log_file,
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="HF converted checkpoint to load with vLLM")
    parser.add_argument(
        "--base-dir",
        type=str,
        help="directory with csv results",
        default="/home/george/sample_efficient_gpt/sample_efficient_gpt/evals/results/",
    )
    parser.add_argument(
        "-a",
        "--task-name",
        type=str,
        default=None,
        help="Task name. Default = all tasks. Use | to split multiple tasks.",
    )
    parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "auto"])
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=1024)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=30)
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size for categorical evaluation")
    parser.add_argument("-x", "--max-problems", type=int, default=None, help="Max problems to evaluate")
    parser.add_argument("--log-file", type=Path, default=None, help="log file for logs")
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["cuda", "cpu", "mps"],
        help="Device type for evaluation: cuda|cpu|mps. empty => autodetect",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    model_slug = "_".join(Path(args.checkpoint).parts[-3:])
    output_csv_path = base_dir / "chat_eval" / f"{model_slug}.csv"
    output_csv_path.parent.mkdir(exist_ok=True, parents=True)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    reduce_device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    chat_template = load_chat_template(Path(args.checkpoint))
    llm = LLM(
        model=args.checkpoint,
        tokenizer=args.checkpoint,
        dtype=args.dtype,
        max_model_len=2048,
        distributed_executor_backend="external_launcher",
        data_parallel_size=world_size,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        seed=42,
    )
    dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
    dp_size = llm.llm_engine.vllm_config.parallel_config.data_parallel_size

    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "GSM8K_5shot", "GSM8K_5shot_cot", "HumanEval"]
    baseline_accuracies = {
        "ARC-Easy": 0.25,
        "ARC-Challenge": 0.25,
        "MMLU": 0.25,
        "GSM8K": 0.0,
        "GSM8K_5shot": 0.0,
        "GSM8K_5shot_cot": 0.0,
        "HumanEval": 0.0,
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split("|")

    log_file = None
    if args.log_file:
        log_file = Path(args.log_file)
        if dp_size > 1:
            log_file = log_file.with_name(f"{log_file.stem}.rank{dp_rank}{log_file.suffix}")
        if log_file.exists():
            log_file.unlink()

    results = {}
    for task_name in task_names:
        acc = run_chat_eval(
            task_name,
            tokenizer,
            llm,
            chat_template,
            dp_rank,
            dp_size,
            reduce_device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            num_samples=args.num_samples,
            max_problems=args.max_problems,
            log_file=log_file,
        )
        results[task_name] = acc
        if dp_rank == 0:
            print(f"{task_name} accuracy: {100 * acc:.2f}%")

    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
        results = {**results, **chatcore_metric_dict}

    if dp_rank == 0:
        print(results)
        with open(output_csv_path, "w") as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}\n")
    if dist.is_initialized():
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
