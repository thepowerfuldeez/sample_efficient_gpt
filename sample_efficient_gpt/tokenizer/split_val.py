"""Script to take tokenized folder and get validation dataset"""

from pathlib import Path
import numpy as np
import shutil
from bisect import bisect_right
from argparse import ArgumentParser


def parse_args():
    p = ArgumentParser()
    p.add_argument("--tokenized-data-path", type=Path, help="Folder with input .npy files")
    p.add_argument("--save-dir", type=Path, help="Folder where to save val.npy")
    p.add_argument("--filename", default="val.npy")
    p.add_argument("--val-tokens", default=20_000_000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inp_p = args.tokenized_data_path
    out_p = args.save_dir
    VAL_TOK = args.val_tokens
    filename = args.filename

    out_p.mkdir(exist_ok=True, parents=True)

    # exluce offsets_ files
    files = [file for file in inp_p.glob("*.npy") if "offsets_" not in file.stem]
    input_file = files[0]
    print(f"Using {input_file} to get validation from")
    base = input_file.stem
    tokens = np.load(inp_p / f"{base}.npy", mmap_mode="r")
    offsets = np.load(inp_p / f"offsets_{base}.npy")
    cut_doc = bisect_right(offsets, VAL_TOK) - 1
    cut_tok = int(offsets[cut_doc])

    np.save(out_p / filename, tokens[:cut_tok])
    np.save(out_p / f"offsets_{filename}", offsets[: cut_doc + 1])
    np.save(inp_p / f"{base}_temp.npy", np.asarray(tokens[cut_tok:]))
    np.save(inp_p / f"offsets_{base}_temp.npy", (offsets[cut_doc:] - cut_tok))

    shutil.move(inp_p / f"{base}_temp.npy", inp_p / f"{base}.npy")
    shutil.move(inp_p / f"offsets_{base}_temp.npy", inp_p / f"offsets_{base}.npy")
    print("Done.")
