use once_cell::sync::Lazy;
use pyo3::{prelude::*, types::PyModule, types::PyBytes, Bound};
use regex::{Regex, escape};
use fancy_regex::Regex as FRegex;
use ahash::AHashMap as Map;
use std::collections::HashMap;
use simdutf::validate_utf8;

static SLOW_RE: Lazy<FRegex> = Lazy::new(|| {
    FRegex::new(r#"\'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#).unwrap()
});

// TODO: add digits grouping regexp
// (?=(\d{3})+(?!\d))


/// Ignore invalid UTF-8 bytes (drop them rather than replacing with U+FFFD).
fn decode_utf8_ignore(bytes: &[u8]) -> String {
    if validate_utf8(&bytes) {
        // Safe because we just validated it's valid UTF-8.
        unsafe { String::from_utf8_unchecked(bytes.to_vec()) }
    } else {
        // “Ignore errors”: replace invalid sequences with U+FFFD (�).
        String::from_utf8_lossy(&bytes).to_string()
    }
}

#[pyclass]
pub struct Splitter {
    split_re: Regex,
}

#[pymethods]
impl Splitter {
    #[new]
    fn new(special_token: String) -> PyResult<Self> {
        let split_re = Regex::new(&escape(&special_token)).unwrap();
        Ok(Self { split_re })
    }

    /// Split a bytes or str object into token-like substrings.
    /// Pass `bytes` for best performance.
    #[pyo3(signature = (chunk))]
    fn split(&self, chunk: Bound<'_, PyAny>) -> PyResult<HashMap<String, usize>> {
        let text: String = if let Ok(b) = chunk.downcast::<PyBytes>() {
            decode_utf8_ignore(b.as_bytes())
        } else if let Ok(s) = chunk.extract::<&str>() {
            s.to_owned()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Expected bytes or str"));
        };

        self.split_counts_internal(&text)
    }

    /// Read a byte range [start, end) from a file and split.
    #[pyo3(signature = (filepath, start, end))]
    fn seek_and_split<'py>(&self, filepath: &str, start: u64, end: u64) -> PyResult<HashMap<String, usize>> {
        if end < start {
            return Err(pyo3::exceptions::PyValueError::new_err("end < start"));
        }
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        let mut file = File::open(filepath)?;
        file.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; (end - start) as usize];
        file.read_exact(&mut buf)?;
        let text = decode_utf8_ignore(&buf);

        self.split_counts_internal(&text)
    }
}

impl Splitter {
    #[inline]
    fn split_counts_internal(&self, text: &str) -> PyResult<HashMap<String, usize>> {
        let mut counts: Map<String, usize> = Map::default();
        // Reserve a bit to avoid rehashing — tune this to your data.
        counts.reserve(text.len() / 8);

        for seg in self.split_re.split(text) {
            collect_counts(seg, &mut counts);
        }
        Ok(counts.into_iter().collect())
    }
}

#[inline]
fn collect_counts(segment: &str, counts: &mut Map<String, usize>) {
    if segment.is_empty() {
        return;
    }

    for m in SLOW_RE.find_iter(segment) {
        bump(counts, m.unwrap().as_str());
    }
}

#[inline]
fn bump(map: &mut Map<String, usize>, tok: &str) {
    if let Some(v) = map.get_mut(tok) {
        *v += 1;
    } else {
        map.insert(tok.to_owned(), 1);
    }
}

#[pymodule]
fn fastsplit(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Splitter>()?;
    Ok(())
}