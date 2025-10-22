use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*;

use ahash::{AHashMap, AHashSet};

type Pair = (u32, u32);

/// A Byte Pair Encoding tokenizer that matches the GPT-4 style implementation
#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<Pair, u32>,   // (left_id,right_id) -> new_id
    pub pattern: String,
    compiled_pattern: Regex,

    // --- new fields ---
    token_bytes: Vec<Vec<u8>>,           // id -> bytes; 0..255 initialized to single bytes
    next_id: u32,                         // next merge id
    merges_order_bytes: Vec<(Vec<u8>, Vec<u8>)>, // [(left_bytes, right_bytes)] in exact order
}

// ------------------------ internal helpers ------------------------

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Merge all non-overlapping occurrences of pair -> new_id.
    /// Returns a small Vec of local pair-count deltas for THIS word only:
    ///   -1 for removed pairs, +1 for newly created pairs.
    ///
    /// NOTE: this version deliberately avoids a HashMap in the hot loop.
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n { Some(self.ids[i + 2]) } else { None };

                // remove old pairs
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                // write merged token
                out.push(new_id);
                i += 2; // skip 'a' and 'b'
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    /// set of word indices where this pair may occur and needs processing
    pos: AHashSet<usize>,
    // NEW: tie-break on concatenated bytes (lexicographic MAX wins)
    tie_key: Vec<u8>,
}

impl PartialEq for MergeJob { fn eq(&self, o: &Self) -> bool { self.count == o.count && self.tie_key == o.tie_key } }
impl PartialOrd for MergeJob { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }


impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.count.cmp(&other.count) {
            Ordering::Equal => self.tie_key.cmp(&other.tie_key), // lexicographic
            o => o,
        }
    }
}

#[inline]
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local_pc.entry((a, b)).or_default() += counts[i];
                    local_wtu.entry((a, b)).or_default().insert(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            },
        )
}

// ------------------------ END helpers ------------------------

impl Tokenizer {
    fn make_tie_key(&self, p: Pair) -> Vec<u8> {
        let (l, r) = p;
        let mut v = Vec::with_capacity(
            self.token_bytes[l as usize].len() + self.token_bytes[r as usize].len()
        );
        v.extend_from_slice(&self.token_bytes[l as usize]);
        v.extend_from_slice(&self.token_bytes[r as usize]);
        v
    }


    /// Core incremental BPE training given unique words and their counts.
    /// `words`: one entry per unique chunk (Vec<u32> of token-ids/bytes).
    /// `counts`: same length as `words`, count per chunk.
    fn train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        assert!(vocab_size >= 256, "vocab_size must be at least 256");
        let num_merges = vocab_size - 256;
        log::info!("Starting BPE training: {} merges to compute", num_merges);
        self.merges.clear();

        // ---- Initial pair_counts and where_to_update (parallel) ----
        log::info!("Computing initial pair counts from {} unique sequences", words.len());
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                heap.push(MergeJob {
                    pair,
                    count: c as u64,
                    pos,
                    tie_key: self.make_tie_key(pair),
                });
            }
        }

        // ---- Merge loop ----
        log::info!("Starting merge loop");
        let mut merges_done = 0u32;
        let mut last_log_percent = 0u32;

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else { break; };

            // Lazy refresh
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current as u64 {
                top.count = current as u64;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }
            if top.count == 0 {
                break;
            }

            // record merge -> assign id and bytes
            let new_id = self.next_id;
            self.next_id += 1;
            self.merges.insert(top.pair, new_id);

            let (l, r) = top.pair;
            let mut merged_b = self.token_bytes[l as usize].clone();
            merged_b.extend_from_slice(&self.token_bytes[r as usize]);
            if self.token_bytes.len() <= new_id as usize {
                self.token_bytes.resize(new_id as usize + 1, Vec::new());
            }
            self.token_bytes[new_id as usize] = merged_b.clone();
            self.merges_order_bytes.push((
                self.token_bytes[l as usize].clone(),
                self.token_bytes[r as usize].clone(),
            ));

            // Merge this pair in all words where it occurs
            let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &top.pos {
                // Apply merge to this word and collect pair-count deltas
                let changes = words[word_idx].merge_pair(top.pair, new_id);
                // Update global pair counts based on this word's count
                for (pair, delta) in changes {
                    let delta_total = delta * counts[word_idx];
                    if delta_total != 0 {
                        *pair_counts.entry(pair).or_default() += delta_total;
                        if delta > 0 {
                            local_pos_updates.entry(pair).or_default().insert(word_idx);
                        }
                    }
                }
            }

            // Add the updated pair counts back to the heap
            for (pair, pos) in local_pos_updates {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: cnt as u64,
                        pos,
                        tie_key: self.make_tie_key(pair),
                    });
                }
            }

            merges_done += 1;

            // Log progress every 1%
            let current_percent = (merges_done * 100) / num_merges;
            if current_percent > last_log_percent {
                log::info!(
                    "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                    current_percent, merges_done, num_merges, top.pair, new_id, top.count
                );
                last_log_percent = current_percent;
            }
        }

        log::info!("Finished training: {} merges completed", merges_done);
    }
}

/// Public methods for the Tokenizer class that will be exposed to Python.
#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new() -> Self {
        let mut token_bytes = Vec::with_capacity(256);
        for b in 0u32..256u32 { token_bytes.push(vec![b as u8]); }
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").expect("Empty regex ok"),
            token_bytes,
            next_id: 256,
            merges_order_bytes: Vec::new(),
        }
    }

    #[pyo3(signature = (pre_tokens, target_vocab_no_specials))]
    pub fn train_from_pretokens(&mut self,
        py: pyo3::Python<'_>,
        pre_tokens: Vec<(pyo3::Py<pyo3::types::PyBytes>, i32)>,
        target_vocab_no_specials: u32
    ) -> PyResult<()> {
        // reset state for repeatability
        self.merges.clear();
        self.merges_order_bytes.clear();
        self.token_bytes.clear();
        for b in 0u32..256u32 { self.token_bytes.push(vec![b as u8]); }
        self.next_id = 256;

        let mut words = Vec::with_capacity(pre_tokens.len());
        let mut counts = Vec::with_capacity(pre_tokens.len());

        for (b, c) in pre_tokens {
            // let ids: Vec<u32> = pyo3::Python::with_gil(|py| {
            //     b.as_ref(py).as_bytes().iter().map(|&x| x as u32).collect()
            // });
            // let slice = b.as_bytes(pyo3::Python::with_gil(|py| py));
            let slice = b.bind(py).as_bytes();
            let ids: Vec<u32> = slice.iter().map(|&x| x as u32).collect();
            words.push(Word::new(ids));
            counts.push(c);
        }
        self.train_core_incremental(words, counts, target_vocab_no_specials);
        Ok(())
    }

    /// Export merges as a list of (left_bytes, right_bytes) in merge order.
    pub fn export_merges_bytes(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.merges_order_bytes.clone()
    }

    /// Optionally export id->bytes for 0..next_id-1 (no specials).
    pub fn export_vocab_bytes(&self) -> Vec<(u32, Vec<u8>)> {
        (0..self.next_id).map(|i| (i, self.token_bytes[i as usize].clone())).collect()
    }

    /// Return the regex pattern
    pub fn get_pattern(&self) -> String {
        self.pattern.clone()
    }

    /// Return the mergeable ranks (token bytes -> token id / rank)
    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        let mut mergeable_ranks = Vec::new();

        // Build vocabulary incrementally from low to high token IDs
        let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();

        for (i, bytes) in token_bytes.iter().enumerate() {
            mergeable_ranks.push((bytes.clone(), i as u32));
        }

        // Sort merges by token id (so we can reconstruct bytes progressively)
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

        for (&pair, &merged_id) in sorted_merges {
            let (left, right) = pair;
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend(&token_bytes[right as usize]);

            if token_bytes.len() <= merged_id as usize {
                token_bytes.resize(merged_id as usize + 1, Vec::new());
            }
            token_bytes[merged_id as usize] = merged_bytes.clone();

            mergeable_ranks.push((merged_bytes, merged_id));
        }

        mergeable_ranks
    }

    /// Encode a string into token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();

        // Split text using the regex pattern
        for m in self.compiled_pattern.find_iter(text) {
            let chunk = m.expect("regex match failed").as_str();

            // Convert chunk to bytes then to u32 IDs
            let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();

            // Apply merges iteratively
            while ids.len() >= 2 {
                // Find the best pair to merge
                let mut best_pair: Option<(usize, Pair, u32)> = None;

                for i in 0..ids.len() - 1 {
                    let pair: Pair = (ids[i], ids[i + 1]);
                    if let Some(&new_id) = self.merges.get(&pair) {
                        if best_pair.is_none() || new_id < best_pair.unwrap().2 {
                            best_pair = Some((i, pair, new_id));
                        }
                    }
                }

                // If we found a pair to merge, apply it
                if let Some((idx, _pair, new_id)) = best_pair {
                    ids[idx] = new_id;
                    ids.remove(idx + 1);
                } else {
                    // No more merges possible
                    break;
                }
            }

            all_ids.extend(ids);
        }

        all_ids
    }
}

#[pymodule]
fn rustbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forwards Rust `log` to Python's `logging`
    m.add_class::<Tokenizer>()?;
    Ok(())
}