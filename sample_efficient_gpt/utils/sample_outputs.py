from pathlib import Path
from argparse import ArgumentParser

import torch

from sample_efficient_gpt.tokenizer import Tokenizer
from sample_efficient_gpt.training.trainer import Trainer
from sample_efficient_gpt.utils.logger import logger


def parse_args():
    p = ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="/home/george/cs336_solutions/assignment1-basics/sample_efficient_gpt/checkpoints/betas=[0.9, 0.99]/8000.pt",
    )
    p.add_argument("--prompt", default="Once")
    p.add_argument(
        "--tokenizer",
        default="/home/george/cs336_solutions/assignment1-basics/tokenizer/tinystories",
    )
    p.add_argument("--top-p", default=0.4, type=float)
    p.add_argument("--temperature", default=0.7, type=float)
    p.add_argument("--max-steps", default=512, type=int)
    p.add_argument("--device", default="cuda", type=str)
    return p.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(load_from=args.checkpoint, load_components="infer", **{"trainer.device": args.device})
    tokenizer = Tokenizer.from_files(
        Path(args.tokenizer) / "vocab.pickle",
        Path(args.tokenizer) / "merges.pickle",
        special_tokens=["<|endoftext|>"],
    )
    eos_token_id = tokenizer.encode(tokenizer.special_tokens[0])[0]

    logger.info(f"EOS {eos_token_id}")

    prompt = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0).to(trainer.cfg.trainer.device)
    generated = trainer.generate(
        prompt,
        eos_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
        max_steps=args.max_steps,
    )
    logger.info(generated)

    logger.info(tokenizer.decode(generated[0].cpu().tolist()))


if __name__ == "__main__":
    main()
