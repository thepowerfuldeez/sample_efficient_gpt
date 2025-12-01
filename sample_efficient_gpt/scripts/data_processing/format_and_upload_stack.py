from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm.auto import tqdm
import subprocess
import tarfile
import tempfile
from pathlib import Path
import ast
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import create_repo, upload_file


def is_mostly_hex(text, threshold=0.9):
    if not text:
        return False
    hex_chars = sum(c in "0123456789abcdefABCDEF" for c in text)
    mostly_hex = hex_chars / len(text) >= threshold
    if mostly_hex:
        print("this is mostly hex")
    return mostly_hex


def apply_formatting(directory, verbose=False, timeout=90):
    lint_msg = ""
    if any(
        x.suffix in {".js", ".ts", ".jsx", ".tsx", ".json", ".html", ".css", ".graphql"}
        for x in Path(directory).glob("**/*")
    ):
        biome_cmd = f"npx @biomejs/biome check --unsafe --write {str(directory)}"
        try:
            biome_output = subprocess.run(biome_cmd, shell=True, capture_output=True, timeout=timeout).stdout.decode()
        except subprocess.TimeoutExpired:
            biome_output = ""
            pass
        if verbose:
            print(biome_output)
        lint_msg = biome_output
    elif any(x.suffix in {".py", ".ipynb"} for x in Path(directory).glob("**/*")):
        ruff_cmd = f"uvx ruff format --target-version py311 --config /home/george/coding_parser/pyproject.toml {str(directory)}"
        lint_cmd = f"uvx ruff check --fix --unsafe-fixes --target-version py311 --config /home/george/coding_parser/pyproject.toml {str(directory)}"
        ruff_output = subprocess.run(ruff_cmd, shell=True, capture_output=True).stdout.decode()
        lint_output = subprocess.run(lint_cmd, shell=True, capture_output=True).stdout.decode()
        # lint_msg = lint_output.strip("\x1b[1m\x1b[91m")
        if verbose:
            print(ruff_output)
            print(lint_output)
    else:
        if verbose:
            print(f"No formatting needed for {directory}")
    return lint_msg


chunk_dir = Path("/home/george/coding_parser/out/chunk-00")
mirrors_dir = Path("/mnt/harddrive/datasets/bigcode_the_stack_v2_updated_smol/mirror/chunk-00")


def process_chunk(df_chunk: pd.DataFrame):
    result = {"repo_name": [], "text": []}
    for _, item in df_chunk.iterrows():
        repo_name = item["repo_name"]
        files = ast.literal_eval(item["files"])
        local_filenames = {f["path"] for f in files}
        # print(local_filenames)

        file_contents = []
        archive_name = f"{repo_name.replace('/', '__')}.tar.gz"
        path = mirrors_dir / archive_name
        if path.exists():
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(path, "r:gz") as tar:
                    tar.extractall(temp_dir, filter="tar")

                parent_dir = list(Path(temp_dir).glob("*"))[0]

                len(list(parent_dir.glob("**/*")))
                exts = set()
                PEEK_BYTES = 128_000

                # remove filtered files
                for f in parent_dir.glob("**/*"):
                    try:
                        if f.is_file():
                            local_path = f.relative_to(parent_dir)
                            with f.open("rb") as fh:
                                snippet = fh.read(PEEK_BYTES).decode("utf-8", errors="ignore")
                            if str(local_path) not in local_filenames or is_mostly_hex(snippet):
                                f.unlink()
                            else:
                                exts.add(local_path.suffix)
                    except Exception:
                        pass
                        # print(f"Error processing {f}: {e}")

                len(list(parent_dir.glob("**/*")))

                # apply formatting
                if exts & {".js", ".ts", ".tsx", ".py"}:
                    apply_formatting(parent_dir)

                # read remaining files
                for f in parent_dir.glob("**/*"):
                    try:
                        if f.is_file():
                            with open(f) as f:
                                file_contents.append(f.read() + "<|endoftext|>")
                    except Exception:
                        pass
                        # print(f"Error processing {f}: {e}")
            result["repo_name"].append(repo_name)
            result["text"].append("".join(file_contents))
    return result


class HFParquetShardWriter:
    def __init__(
        self,
        repo_id: str,  # e.g. "yourname/code-text-ds"
        branch: str = "main",
        shard_prefix: str = "train",
        remote_folder: str = "data",  # store shards under data/
        shard_size: int = 5000,
        shard_idx: int = 1,
        compression: str = "zstd",  # good compression ratio & speed
    ):
        self.repo_id = repo_id
        self.branch = branch
        self.shard_prefix = shard_prefix
        self.remote_folder = remote_folder
        self.shard_size = shard_size
        self.compression = compression

        # in-memory buffer
        self.buffer: dict[str, list[str]] = {"repo_name": [], "text": []}
        self.shard_idx = shard_idx

        # ensure repo exists
        create_repo(repo_id, repo_type="dataset", exist_ok=True)

        # track schema once (both columns are strings)
        self.schema = pa.schema([("repo_name", pa.string()), ("text", pa.string())])

        # local tmp dir for shards
        self.tmp_dir = Path("./_hf_shards_tmp")
        self.tmp_dir.mkdir(exist_ok=True, parents=True)

    def add_batch(self, batch: dict[str, list[str]]):
        # Expecting {"repo_name": list[str], "text": list[str]}
        n = len(batch["repo_name"])
        assert len(batch["text"]) == n, "Column lengths must match"

        self.buffer["repo_name"].extend(batch["repo_name"])
        self.buffer["text"].extend(batch["text"])

        if len(self.buffer["repo_name"]) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.buffer["repo_name"]:
            return

        # Build arrow table
        table = pa.table(self.buffer, schema=self.schema)

        # Write compressed Parquet shard locally
        shard_name = f"{self.shard_prefix}-{self.shard_idx:06d}.parquet"
        local_path = self.tmp_dir / shard_name
        pq.write_table(table, local_path, compression=self.compression)

        # Upload to Hub (LFS) under data/
        remote_path = f"{self.remote_folder}/{shard_name}"
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.branch,
            commit_message=f"Add shard {shard_name} ({len(self.buffer['repo_name'])} rows)",
        )

        # Clear buffer, increment index
        self.buffer = {"repo_name": [], "text": []}
        self.shard_idx += 1

        # Optional: remove local shard after upload
        try:
            local_path.unlink()
        except FileNotFoundError:
            pass

    def finalize(self):
        # Flush remaining rows
        self._flush()


def batched_iter(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]


if __name__ == "__main__":
    # df = pd.concat([pd.read_csv("dicts_1.csv", header=0), pd.read_csv("dicts_2.csv", header=0)])
    df = pd.read_csv("dicts_python.csv", header=0)
    writer = HFParquetShardWriter(
        repo_id="thepowerfuldeez/the-stack-v2-extra-python-content",
        shard_size=5000,  # flush after 5000 files
        compression="zstd",
        shard_idx=0,
    )
    max_workers = 24
    chunk_size = 100
    max_inflight = max_workers * 3
    start_from = 0
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="uploader")
    try:
        chunks = batched_iter(df.iloc[start_from:], chunk_size)
        inflight = set()

        # prime
        for _ in range(min(max_inflight, (len(df) - start_from + chunk_size - 1) // chunk_size)):
            try:
                df_chunk = next(chunks)
            except StopIteration:
                break
            inflight.add(executor.submit(process_chunk, df_chunk))

        pbar_total = (len(df) - start_from + chunk_size - 1) // chunk_size
        with tqdm(total=pbar_total) as pbar:
            while inflight:
                done, inflight = wait(inflight, timeout=10, return_when=FIRST_COMPLETED)
                # drain completed
                for fut in done:
                    try:
                        out = fut.result()  # raises if worker failed
                        writer.add_batch(out)  # {"repo_name": [...], "text": [...]}
                    except Exception as e:
                        print(f"[worker error] {e}")
                    pbar.update(1)

                # replenish to keep at most max_inflight running
                while len(inflight) < max_inflight:
                    try:
                        df_chunk = next(chunks)
                    except StopIteration:
                        break
                    inflight.add(executor.submit(process_chunk, df_chunk))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # finish cleanly (may write a final short shard)
    writer.finalize()
