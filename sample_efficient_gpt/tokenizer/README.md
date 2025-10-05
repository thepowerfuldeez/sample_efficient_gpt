### Speed profiling
```bash
sudo py-spy record -f speedscope -o profile.json -- python bpe1.py
```

### Pre-compiling fastsplit
```
curl https://sh.rustup.rs -sSf | sh
```

```
cd fastsplit
maturin develop --release
```


### Tokenizer training optimization

Pre-tokenization (Rust + PyO3)

Pre-tokenization was rewritten in Rust (via PyO3). The Rust code splits text with a regex and counts pre-tokens directly, avoiding round-tripping large Python string lists (pickle/marshalling).

Single-process benchmark (1.5M lines):
- Baseline (Python): 67.2s total

    – split: 45.6s, count: 17.3s
- Iter 1 (Rust split, Python count): 

    ~26.0s (split) + 17.3s (count)
- Iter 2 (Rust split + count): 
    
    21.2s total

Multi-process benchmark (large file):
- 1 process: 77.2s
- 2 processes: 73.5s
- 8 processes: 69.2s

Diminishing returns are expected due to I/O limits; multi-processing helps most on large files.

Reproduce:
```bash
python sample_efficient_gpt/pretokenization.py
```

#### Regex choice & newline handling
* The default Rust regex crate (no lookarounds) required simplifying the pattern, which initially failed to split \n\n\n into \n\n + \n, causing test/vocab mismatches.
* Switching to fancy-regex library (lookarounds) worked but was ~4× slower on the same 8-process run (268s vs 69.2s).
* Final approach: keep regex and add a tiny post-processing pass to split the trailing whitespace. Overhead is small (~79.8s vs 69.2s on the 8-proc run) and still far faster than fancy-regex.



### Sorting
To speed up sorting between iterations, a lightweight heuristic is used:
* Cache the top 10% from the previous sorted list.
* Re-sort only pairs involving keys updated in this iteration (“updated pairs”).
* Merge the cached top-10% with the newly sorted updated pairs and take the top-K.

In practice, this cut sorting time by ~3× with no observed quality loss in our runs.

Note: This is an approximation—there’s no strict guarantee that the true global top item won’t fall outside “updated pairs + cached top-10%”.