import pickle
import tempfile
from pathlib import Path
from argparse import ArgumentParser

from tokenizers import Tokenizer, Regex, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.processors import ByteLevel as ByteLevelProc
from transformers import PreTrainedTokenizerFast


from sample_efficient_gpt.tokenizer.tokenizer import Tokenizer as RefTokenizer


def parse_args():
    p = ArgumentParser()
    p.add_argument("--tokenizer-path", default="data_dclm_edu/tokenizer_superbpe/")
    p.add_argument("--superbpe-transition-idx", default=-1, type=int)
    p.add_argument(
        "--save-path",
    )
    return p.parse_args()


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def btok_to_str(b: bytes, b2u: dict[int, str]) -> str:
    # 1:1 bytes->unicode via latin-1 decode, then map each byte through GPT-2 table
    return "".join(b2u[ord(ch)] for ch in b.decode("latin-1"))


# BOUNDARY = (
#     r"(?<=\d)(?=(?:\d{3})+(?!\d))|(?<=\D)(?=\d)|(?<=\d)(?=\D)|"
#     r"(?<=-)(?=\|)|(?<=\|)(?=-)|"
#     r"(?<=\p{L})(?='(?:s|d|m|t|ll|ve|re))|(?<='(?:s|d|m|t|ll|ve|re))"
# )


def build_and_save_tokenizer(
    vocab_bytes: list[tuple[int, bytes]],
    merges_bytes: list[tuple[bytes, bytes]],
    specials: list[str],
    add_prefix_space: bool = False,
    superbpe: bool = False,
):
    # 1) map bytes -> GPT-2 unicode domain (preserve IDs exactly as given)
    b2u = bytes_to_unicode()
    vocab = {btok_to_str(b, b2u): i for i, b in vocab_bytes.items()}

    # 2) convert merges (preserve order exactly)
    merges = [(btok_to_str(l, b2u), btok_to_str(r, b2u)) for (l, r) in merges_bytes]

    # 3) build tokenizer
    tok = Tokenizer(BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))

    if superbpe:
        BOUNDARY = r"""
        (?<=\p{L})(?=['\u2019](?i:(?:s|d|m|t|ll|ve|re)))
        | (?<=['\u2019](?i:(?:s|d|m|t|ll|ve|re)))
        | (?<=-)(?=\|) | (?<=\|)(?=-)
        | (?<=\d)(?=(?:\d{3})+(?!\d))                 # thousands boundaries inside digit runs
        """
    else:
        BOUNDARY = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    tok.pre_tokenizer = Sequence(
        [
            Split(Regex(BOUNDARY), behavior="isolated", invert=False),
            ByteLevelPre(add_prefix_space=add_prefix_space, use_regex=False),
        ]
    )
    tok.decoder = ByteLevelDec()
    tok.post_processor = ByteLevelProc(trim_offsets=False)

    # 4) mark specials so pretokenizer never splits them
    assert specials
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])

    # Build a contiguous id mapping
    items_by_old_id = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])  # [(token, old_id), ...] by old_id
    new_vocab = {tok_str: new_id for new_id, (tok_str, _) in enumerate(items_by_old_id)}

    # Rebuild model with contiguous ids
    tok.model = BPE(vocab=new_vocab, merges=merges, fuse_unk=False, byte_fallback=False)
    return tok


if __name__ == "__main__":
    args = parse_args()

    if args.superbpe_transition_idx == -1:
        superbpe = False
        print("not using superbpe")

    tokenizer_path = Path(args.tokenizer_path)

    tok = RefTokenizer.from_files(
        str(tokenizer_path / "vocab.pickle"),
        str(tokenizer_path / "merges.pickle"),
        superbpe_transition_idx=args.superbpe_transition_idx if superbpe else None,
    )

    vocab_bytes = pickle.loads((tokenizer_path / "vocab.pickle").read_bytes())  # [(id:int, token:bytes), ...]
    merges_bytes = pickle.loads((tokenizer_path / "merges.pickle").read_bytes())  # [(left:bytes, right:bytes), ...]
    specials = ["<|endoftext|>"] + [f"<|reserved_{i}|>" for i in range(20)]
    converted_tokenizer = build_and_save_tokenizer(vocab_bytes, merges_bytes, specials, superbpe=superbpe)
    with tempfile.NamedTemporaryFile() as f:
        converted_tokenizer.save(str(f.name))
        print(f.name)
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f.name)

    texts = [
        "Big cat sat on a mat",
        "I earn 1234567 dollars",
        "class NaturalHistory(nn.Module)",
        "directive number twenty-four",
        "am i on a right track here? help plz",
        "this function does summation on a last axis and then returns a scalar.",
        "def function_a(arg1: int, arg2: str) -> str:",
    ]

    for text in texts:
        ref = [tok.decode([x]) for x in tok.encode(text)]
        conv = [hf_tokenizer.decode([x]) for x in hf_tokenizer(text).input_ids]
        if ref != conv:
            print(f"{ref=} {conv=} dont match!")

    save_path = Path(args.save_path)
    hf_tokenizer.save_pretrained(save_path)
