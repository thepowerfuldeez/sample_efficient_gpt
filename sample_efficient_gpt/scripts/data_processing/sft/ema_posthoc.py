"""Script to run EMA on already saved checkpoints"""

import os
import glob
import torch
from pathlib import Path
from argparse import ArgumentParser


def parse_args():
    p = ArgumentParser()
    p.add_argument("--checkpoint-dir", help="checkpoint dir to do ema in-place at")
    p.add_argument("--checkpoint1", help="first ckpt to do average of weights")
    p.add_argument("--checkpoint2", help="second ckpt to do average of weights")
    p.add_argument("--out", help="out ckpt path for average")
    return p.parse_args()


# EMA decay: ema = decay * ema + (1 - decay) * param
EMA_DECAY = 0.8  # pretty low value, 0.9 corresponds to 50k steps window with 23 checkpoints
EMA_SUFFIX = "_ema"


def list_normal_checkpoints(dirs):
    """
    Return list of (iteration, path) for non-EMA checkpoints across all dirs,
    sorted by iteration.
    """
    ckpts = []
    for d in dirs:
        paths = glob.glob(os.path.join(d, "*.pt"))
        for path in paths:
            base, ext = os.path.splitext(path)
            if base.endswith(EMA_SUFFIX):
                # skip EMA files
                continue
            # iteration number is the name of the file
            try:
                it = int(Path(base).stem)
            except:
                continue

            ckpts.append((it, path))

    ckpts.sort(key=lambda x: x[0])
    return ckpts


def find_latest_ema(dirs):
    """
    Look for existing *_ema.pt checkpoints and return:
    (latest_iteration, ema_model_state_dict) or (None, None) if none exist.
    """
    latest_iter = None
    latest_path = None

    for d in dirs:
        paths = glob.glob(os.path.join(d, f"*{EMA_SUFFIX}.pt"))
        for path in paths:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Failed to load EMA {path}: {e}")
                continue

            it = ckpt.get("iteration", None)
            if it is None:
                print(f"Skipping EMA {path}: no 'iteration' field")
                continue

            if latest_iter is None or it > latest_iter:
                latest_iter = it
                latest_path = path

    if latest_iter is None:
        print("No existing EMA checkpoints found. Starting EMA from scratch.")
        return None, None

    print(f"Resuming EMA from {latest_path} (iteration {latest_iter})")
    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    ema_state = ckpt["model"]
    return latest_iter, ema_state


def update_ema(ema_state, model_state, decay):
    """In-place EMA update: ema = decay * ema + (1 - decay) * param."""
    if ema_state is None:
        # First time initializing EMA
        return {k: v.clone() for k, v in model_state.items()}

    with torch.no_grad():
        for k, v in model_state.items():
            if torch.is_tensor(v):
                ema_state[k] = ema_state[k] * decay + (1 - decay) * v
    return ema_state


def merge_two_ckpts(ckpt1_path, ckpt2_path, out_path):
    model_state = torch.load(ckpt1_path, map_location="cpu", weights_only=False)["model"]
    ckpt = torch.load(ckpt2_path, map_location="cpu", weights_only=False)
    new_state = ckpt["model"]
    with torch.no_grad():
        for k, v in new_state.items():
            if torch.is_tensor(v):
                model_state[k] = model_state[k] * 0.5 + 0.5 * v
    ckpt = {
        "config": ckpt.get("config", None),
        "model": model_state,  # on CPU
        "optimizer": None,  # drop optimizer
        "iteration": ckpt.get("iteration"),
        "run_id": ckpt.get("run_id", None),
        "rng_state": ckpt.get("rng_state", None),
    }
    torch.save(ckpt, out_path)


def ema_iter(checkpoint_dirs):
    # 1) Get all non-EMA checkpoints sorted by iteration
    ckpts = list_normal_checkpoints(checkpoint_dirs)
    if not ckpts:
        print("No non-EMA checkpoints found.")
        return

    print(f"Found {len(ckpts)} non-EMA checkpoints total.")

    # 2) Find latest EMA to resume from, if any
    latest_ema_iter, ema_state = find_latest_ema(checkpoint_dirs)

    # 3) Process checkpoints in order, skipping those already covered by EMA
    for iteration, path in ckpts:
        if latest_ema_iter is not None and iteration <= latest_ema_iter:
            print(f"Skipping iteration {iteration} (already covered by EMA)")
            continue

        print(f"Processing iteration {iteration} from {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_state = ckpt["model"]

        # Update EMA
        ema_state = update_ema(ema_state, model_state, EMA_DECAY)

        # Build EMA checkpoint
        ema_ckpt = {
            "config": ckpt.get("config", None),
            "model": ema_state,  # on CPU
            "optimizer": None,  # drop optimizer
            "iteration": iteration,
            "run_id": ckpt.get("run_id", None),
            "rng_state": ckpt.get("rng_state", None),
        }

        base, ext = os.path.splitext(path)
        ema_path = f"{base}{EMA_SUFFIX}{ext}"

        print(f"Saving EMA checkpoint: {ema_path}")
        torch.save(ema_ckpt, ema_path)

        del ckpt
        del model_state

    print("Done updating EMA checkpoints.")


def main():
    args = parse_args()
    # do ema
    if args.checkpoint_dir:
        checkpoint_dirs = [args.checkpoint_dir]
        ema_iter(checkpoint_dirs)
    else:
        ckpt1, ckpt2 = args.checkpoint1, args.checkpoint2
        merge_two_ckpts(ckpt1, ckpt2, args.out)
        print(f"saved average to {args.out}")
    


if __name__ == "__main__":
    main()
