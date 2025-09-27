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




####
Optimization

### Pre-tokenization
I re-wrote part with pre-tokenization which splits data by regexp and counts pre-tokens
Before it was 67.2s on 1.5M lines of text, in which 45.6s took by split, 17.3s took by get_counts

Iter 1: 26s for split, and same 17.3s for get_counts

Iter 2: 21.2s total time (split + get counts)

Iter 2 is much faster because count is now happens inside rust code, and no large list: str gets marshalled via pickle from pyo3

Using multiple processes works best on a large file, so
1 process 77.2s
2 processes 73.5s
8 processes 69.2s

(this can be tested via running `python sample_efficient_gpt/pretokenization.py`)

Initially, I made splitting using default regex library in Rust, but it doesn't support lookahead, so I simplified regexp, but later
discovered that it never splits \n\n\n into \n\n + \n, so I had mismatching vocab in tests.
I tried to switch to fancy-regex library, but it was 4x slower just because of this lookaround support! (268s vs 69.2s on 8 processes)
So I went with post-processing of the last whitespace character, which adds to the overall time just a little bit (79.8s vs 69.2s initially)

### Sorting
For sorting I went with some heuristic -- what if I would store top 10% of sorted values from previous step + all sorted values when using subset of pairs (only pairs of updated keys) -- turns out I could drive sorting time to make it 3x lower without sacrificing quality. This is more like an approximation, we cannot 
guarantee that correct value won't be in the updated pairs + top10% though.