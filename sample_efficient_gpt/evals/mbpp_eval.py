#!/usr/bin/env python3
"""
MBPP evaluator.

Usage examples:
  # Evaluate JSONL samples against the bundled lite set
  python simple_mbpp_eval.py --samples completions.jsonl --dataset lite

  # Evaluate JSONL samples against a full MBPPJSONL(.gz) file you provide
  python simple_mbpp_eval.py --samples completions.jsonl --dataset /path/to/MbppPlus.jsonl.gz

  # Evaluate a directory layout: samples/<task_id>/*.py (full prompt code)
  python simple_mbpp_eval.py --samples samples_dir --dataset lite

Notes:
- Input format (jsonl):
    {"task_id": "Mbpp/10001", "completion": "    return s[::-1]\n"}
  or
    {"task_id": "Mbpp/10001", "solution": "def reverse_string(s):\n    return s[::-1]\n"}
- Task IDs must exist in the chosen dataset (lite or your MBPPfile).
- This evaluator is deliberately simple: it sandboxes by running each submission
  in a separate process with a global timeout per submission (not per test case).
"""

import argparse
import gzip
import io
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable, Callable

import sample_efficient_gpt.evals.mbpp_lite as L

PASS, FAIL, TIMEOUT = "pass", "fail", "timeout"


class SimpleMBPPEvaluator:
    """
    Programmatic API to run MBPP (or the bundled 'lite') during training.
    Usage from your Trainer:
        self.mbbp_evaluator = SimpleMBPPEvaluator(dataset='lite', timeout=8.0)
        metrics = self.mbbp_evaluator.evaluate_with_generate(self.generate, num_samples=1)
    """

    def __init__(self, dataset: str = "lite", timeout: float = 15.0, base_only: bool = False):
        self.timeout = float(timeout)
        self.base_only = bool(base_only)
        self.problems: dict[str, Problem] = load_dataset(dataset)
        # Precompute expected outputs once
        self.expected_map: dict[str, dict[str, list[Any]]] = {}
        for tid, prob in self.problems.items():
            try:
                b, p = _compute_expected(prob)
            except BaseException:
                b, p = [], []
            self.expected_map[tid] = {"base": b, "plus": p}

    def evaluate_with_generate(
        self,
        generate_fn: Callable[..., str],
        *,
        num_samples: int = 1,
        task_ids: list[str] | None = None,
        mode: str = "completion",  # "completion" => append to prompt; "solution" => full file
        ks: list[int] | None = None,
        plus: bool | None = None,
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
        """
        generate_fn(prompt: str, **gen_kwargs) -> str
        Returns {'eval': {...}, 'pass_at_k': {'base': {...}, 'plus': {...?}}}
        """
        if ks is None:
            ks = [1, 10, 100]
        if plus is None:
            plus = not self.base_only

        # choose tasks
        all_ids = list(self.problems.keys())
        tids = task_ids if task_ids is not None else all_ids
        # keep only known tasks; allow "Mbpp_10001" to map to "Mbpp/10001"
        fixed_tids: list[str] = []
        for tid in tids:
            if tid in self.problems:
                fixed_tids.append(tid)
            else:
                alt = tid.replace("_", "/")
                if alt in self.problems:
                    fixed_tids.append(alt)
        tids = fixed_tids or all_ids

        results_by_task: dict[str, list[dict[str, Any]]] = {}
        totals_by_task: dict[str, int] = {}
        base_correct_by_task: dict[str, int] = {}
        both_correct_by_task: dict[str, int] = {}

        for tid in tids:
            prob = self.problems[tid]
            exp = self.expected_map[tid]
            for _ in range(int(num_samples)):
                print("sending prompt ", prob.prompt)
                completion = generate_fn(prob.prompt, **gen_kwargs)
                completion = completion if isinstance(completion, str) else str(completion)
                code = (prob.prompt + completion) if mode == "completion" else completion

                base_status, _ = _untrusted_check(
                    code=code,
                    entry_point=prob.entry_point,
                    inputs=prob.base_input,
                    expected=exp["base"],
                    atol=prob.atol,
                    timeout=self.timeout,
                    fast_fail=True,
                )
                plus_status, _ = None, []
                if plus:
                    plus_status, _ = _untrusted_check(
                        code=code,
                        entry_point=prob.entry_point,
                        inputs=prob.plus_input,
                        expected=exp["plus"],
                        atol=prob.atol,
                        timeout=self.timeout,
                        fast_fail=True,
                    )

                results_by_task.setdefault(tid, []).append(
                    {"task_id": tid, "base_status": base_status, "plus_status": plus_status}
                )
                totals_by_task[tid] = totals_by_task.get(tid, 0) + 1
                base_correct_by_task[tid] = base_correct_by_task.get(tid, 0) + (1 if base_status == PASS else 0)
                if plus and base_status == PASS and plus_status == PASS:
                    both_correct_by_task[tid] = both_correct_by_task.get(tid, 0) + 1

        # compute pass@k restricted by min samples per task
        if not totals_by_task:
            return {"eval": {}, "pass_at_k": {}}
        min_n = min(totals_by_task.values())
        ks = [k for k in ks if k <= min_n]
        num_samples_vec = [totals_by_task[t] for t in totals_by_task.keys()]
        base_correct_vec = [base_correct_by_task.get(t, 0) for t in totals_by_task.keys()]
        out: dict[str, Any] = {
            "eval": results_by_task,
            "pass_at_k": {"base": {f"pass@{k}": _estimate_pass_at_k(num_samples_vec, base_correct_vec, k) for k in ks}},
        }
        if plus:
            both_correct_vec = [both_correct_by_task.get(t, 0) for t in totals_by_task.keys()]
            out["pass_at_k"]["plus"] = {
                f"pass@{k}": _estimate_pass_at_k(num_samples_vec, both_correct_vec, k) for k in ks
            }
        return out


# ----------------------------- dataset loading ------------------------------


def _stream_jsonl(path: str) -> Iterable[dict[str, Any]]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if line.strip():
                    yield json.loads(line)


def _load_solutions(samples_path: str) -> Iterable[dict[str, Any]]:
    """Accepts either a .jsonl(.gz) file or a directory:
    - file: jsonl with fields {task_id, completion? , solution?}
    - dir:  samples_dir/<task_id>/*.py  (full prompt code file)
    """
    if os.path.isfile(samples_path):
        for i, rec in enumerate(_stream_jsonl(samples_path)):
            if not any(k in rec for k in ("completion", "solution")):
                raise ValueError(f"Line {i}: missing 'completion' or 'solution'")
            rid = f"{os.path.basename(samples_path)}:{i}"
            rec["_identifier"] = rid
            yield rec
    else:
        # directory mode
        for task_id in sorted(os.listdir(samples_path)):
            tdir = os.path.join(samples_path, task_id)
            if not os.path.isdir(tdir):
                continue
            for fname in sorted(os.listdir(tdir)):
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(tdir, fname)
                with open(fpath, encoding="utf-8") as f:
                    code = f.read()
                yield {
                    "_identifier": fpath,
                    "task_id": task_id,
                    "solution": code,
                }


# ------------------------------ MBPPhelpers ------------------------------


# A minimal, task-specific input “fixup” ported from evalplus to handle
# a few quirky tasks in MBPP+. We keep it tiny; for best reliability use
# the bundled 'lite' dataset or supply a clean MBPPfile.
def _mbpp_deserialize_inputs(task_id: str, inputs: list) -> list:
    # Feel free to extend this mapping if you use the full MBPPrelease.
    # For the bundled lite set, this is a no-op.
    return inputs


# ------------------------------ lite dataset -------------------------------


@dataclass
class Problem:
    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    base_input: list[list[Any]]
    plus_input: list[list[Any]]
    atol: float = 0.0


def load_dataset(dataset: str) -> dict[str, Problem]:
    """Load either the bundled 'lite' set or a MBPPjsonl(.gz)."""
    if dataset.strip().lower() == "lite":
        problems = {}
        for task_id, rec in L.DATA.items():
            problems[task_id] = Problem(
                task_id=task_id,
                prompt=rec["prompt"],
                canonical_solution=rec["canonical_solution"],
                entry_point=rec["entry_point"],
                base_input=rec["base_input"],
                plus_input=rec["plus_input"],
                atol=rec.get("atol", 0.0),
            )
        return problems

    # Otherwise, treat 'dataset' as a path to MBPPjsonl(.gz)
    problems: dict[str, Problem] = {}
    for rec in _stream_jsonl(dataset):
        # MBPPuses "task_id" like "Mbpp/123"
        tid = rec["task_id"]
        problems[tid] = Problem(
            task_id=tid,
            prompt=rec["prompt"],
            canonical_solution=rec["canonical_solution"],
            entry_point=rec["entry_point"],
            base_input=_mbpp_deserialize_inputs(tid, rec["base_input"]),
            plus_input=_mbpp_deserialize_inputs(tid, rec["plus_input"]),
            atol=float(rec.get("atol", 0.0)),
        )
    return problems


# ------------------------------ execution core -----------------------------


def _safe_compare(out: Any, exp: Any, atol: float) -> bool:
    """Equality with float tolerance; supports nested (list/tuple) structures."""
    # exact type match not strictly required for this simple evaluator
    if isinstance(exp, float) or isinstance(out, float):
        return math.isclose(float(out), float(exp), rel_tol=1e-7, abs_tol=atol)
    if isinstance(exp, (list, tuple)) and isinstance(out, (list, tuple)):
        if len(exp) != len(out):
            return False
        return all(_safe_compare(o, e, atol) for o, e in zip(out, exp))
    return out == exp


def _compute_expected(problem: Problem) -> tuple[list[Any], list[Any]]:
    """Execute the *canonical* solution to produce expected outputs."""
    code = problem.prompt + problem.canonical_solution
    g: dict[str, Any] = {}
    exec(code, g)
    fn = g[problem.entry_point]
    base_out = [fn(*args) for args in problem.base_input]
    plus_out = [fn(*args) for args in problem.plus_input]
    return base_out, plus_out


def _worker_run(
    code: str,
    entry_point: str,
    inputs: list[list[Any]],
    expected: list[Any],
    atol: float,
    fast_fail: bool,
    out_queue: mp.Queue,
):
    """Run one submission against a set of inputs. No network/filesystem guards here
    beyond process isolation; keep this simple and fast. You can add 'resource'
    limits on Unix if you want."""
    try:
        # isolate globals, silence stdout/stderr
        g: dict[str, Any] = {}
        sys_stdout, sys_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            exec(code, g)
            fn = g[entry_point]
            details: list[bool] = []
            for args, exp in zip(inputs, expected):
                try:
                    out = fn(*args)
                    ok = _safe_compare(out, exp, atol)
                except BaseException:
                    ok = False
                details.append(ok)
                if fast_fail and not ok:
                    break
        finally:
            sys.stdout = sys_stdout
            sys.stderr = sys_stderr

        status = PASS if details and all(details) and len(details) == len(expected) else FAIL
        out_queue.put((status, details))
    except BaseException:
        out_queue.put((FAIL, []))


def _untrusted_check(
    code: str,
    entry_point: str,
    inputs: list[list[Any]],
    expected: list[Any],
    atol: float,
    timeout: float,
    fast_fail: bool = True,
) -> tuple[str, list[bool]]:
    """Run one completion in a subprocess with a *global* timeout."""
    q: mp.Queue = mp.Queue()
    p = mp.Process(
        target=_worker_run,
        args=(code, entry_point, inputs, expected, atol, fast_fail, q),
    )
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join(0.1)
        return TIMEOUT, []
    try:
        status, details = q.get_nowait()
    except Exception:
        status, details = FAIL, []
    return status, details


# ------------------------------ pass@k math --------------------------------


def _estimate_pass_at_k(num_samples: list[int], num_correct: list[int], k: int) -> float:
    """Unbiased estimator 1 - comb(n-c, k)/comb(n, k) (OpenAI HumanEval)."""

    def est(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        # 1 - Π_{i=n-c+1..n} (1 - k/i)
        prod = 1.0
        for i in range(n - c + 1, n + 1):
            prod *= 1.0 - k / i
        return 1.0 - prod

    vals = [est(int(n), int(c), k) for n, c in zip(num_samples, num_correct)]
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------- main ------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Simple MBPP evaluator (dependency-free)")
    ap.add_argument("--samples", required=True, help="Path to samples .jsonl(.gz) or a directory layout")
    ap.add_argument("--dataset", default="lite", help="'lite' or path to MBPPjsonl(.gz)")
    ap.add_argument("--timeout", type=float, default=15.0, help="Seconds per submission (global)")
    ap.add_argument("--base_only", action="store_true", help="Evaluate base tests only")
    ap.add_argument(
        "--k", default="1,10,100", help="Comma-separated ks for pass@k (we'll ignore ks > min(n_samples_per_task))"
    )
    ap.add_argument("--output", default=None, help="Where to save results JSON (default: alongside samples)")
    args = ap.parse_args()

    problems = load_dataset(args.dataset)
    if not problems:
        print("No problems loaded; check --dataset", file=sys.stderr)
        sys.exit(1)

    # Pre-compute ground truth for expected outputs using canonical solutions
    expected_map: dict[str, dict[str, list[Any]]] = {}
    print(f"Computing expected outputs for {len(problems)} tasks...")
    t0 = time.time()
    for tid, prob in problems.items():
        try:
            base_exp, plus_exp = _compute_expected(prob)
        except BaseException as e:
            print(f"[warn] failed to compute expected for {tid}: {e}", file=sys.stderr)
            base_exp, plus_exp = [], []
        expected_map[tid] = {"base": base_exp, "plus": plus_exp}
    print(f"Expected outputs computed in {time.time() - t0:.2f}s")

    # Read samples and evaluate
    results_by_task: dict[str, list[dict[str, Any]]] = {}
    totals_by_task: dict[str, int] = {}
    base_correct_by_task: dict[str, int] = {}
    both_correct_by_task: dict[str, int] = {}

    print("Evaluating samples...")
    for sample in _load_solutions(args.samples):
        task_id = sample["task_id"]
        if task_id not in problems:
            # be lenient: allow "Mbpp_10001" directory to map to "Mbpp/10001"
            alt = task_id.replace("_", "/")
            if alt in problems:
                task_id = alt
            else:
                print(f"[warn] sample for unknown task_id: {sample['task_id']}", file=sys.stderr)
                continue

        prob = problems[task_id]
        sol_code = sample.get("solution")
        if not sol_code:
            # 'completion' needs to be appended after the prompt
            sol_code = prob.prompt + sample["completion"]

        exp = expected_map[task_id]
        base_status, base_details = _untrusted_check(
            code=sol_code,
            entry_point=prob.entry_point,
            inputs=prob.base_input,
            expected=exp["base"],
            atol=prob.atol,
            timeout=args.timeout,
            fast_fail=True,
        )

        plus_status, plus_details = None, []
        if not args.base_only:
            plus_status, plus_details = _untrusted_check(
                code=sol_code,
                entry_point=prob.entry_point,
                inputs=prob.plus_input,
                expected=exp["plus"],
                atol=prob.atol,
                timeout=args.timeout,
                fast_fail=True,
            )

        # aggregate
        results_by_task.setdefault(task_id, []).append(
            {
                "task_id": task_id,
                "base_status": base_status,
                "plus_status": plus_status,
            }
        )
        totals_by_task[task_id] = totals_by_task.get(task_id, 0) + 1
        base_correct_by_task[task_id] = base_correct_by_task.get(task_id, 0)(1 if base_status == PASS else 0)
        if not args.base_only and plus_status == PASS and base_status == PASS:
            both_correct_by_task[task_id] = both_correct_by_task.get(task_id, 0) + 1

    # pass@k
    ks = []
    for tok in args.k.split(","):
        tok = tok.strip()
        if tok:
            try:
                ks.append(int(tok))
            except ValueError:
                pass
    # Ensure we only compute ks that are <= min samples per task
    min_n = min(totals_by_task.values()) if totals_by_task else 0
    ks = [k for k in ks if k <= min_n]

    num_samples = [totals_by_task[t] for t in totals_by_task.keys()]
    base_correct = [base_correct_by_task.get(t, 0) for t in totals_by_task.keys()]
    base_pass_at_k = {f"pass@{k}": _estimate_pass_at_k(num_samples, base_correct, k) for k in ks}

    out = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "eval": results_by_task,
        "pass_at_k": {"base": base_pass_at_k},
    }

    if not args.base_only:
        both_correct = [both_correct_by_task.get(t, 0) for t in totals_by_task.keys()]
        if ks:
            out["pass_at_k"]["plus"] = {f"pass@{k}": _estimate_pass_at_k(num_samples, both_correct, k) for k in ks}

    # save
    if args.output:
        out_path = args.output
    else:
        if os.path.isfile(args.samples):
            base = args.samples
            if base.endswith(".jsonl"):
                out_path = base.replace(".jsonl", ".eval_results.json")
            elif base.endswith(".jsonl.gz"):
                out_path = base.replace(".jsonl.gz", ".eval_results.json")
            else:
                out_path = base + ".eval_results.json"
        else:
            out_path = os.path.join(args.samples, "eval_results.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # brief printout
    print("\nResults:")
    for k, v in base_pass_at_k.items():
        print(f"  base {k}: {v:.3f}")
    if not args.base_only and "plus" in out["pass_at_k"]:
        for k, v in out["pass_at_k"]["plus"].items():
            print(f"  plus {k}: {v:.3f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
