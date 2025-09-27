# A tiny, clean MBPP-like subset with straightforward equality checks.
# No special oracles, no tricky input conversions. Good for quick smoke tests.

DATA = {
    # --------------------------------------------------------------------- #
    # Task IDs use the "Mbpp/100xx" range to avoid clashing with real IDs. #
    # --------------------------------------------------------------------- #
    "Mbpp/10001": {
        "entry_point": "reverse_string",
        "prompt": (
            "def reverse_string(s: str) -> str:\n"
            '    """Return the reverse of s.\n'
            "    Examples:\n"
            "    >>> reverse_string('abc')\n"
            "    'cba'\n"
            '    """\n'
        ),
        "canonical_solution": ("    return s[::-1]\n"),
        "base_input": [["abc"], ["racecar"], ["a"], [""]],
        "plus_input": [["hello"], ["ab"], ["XYZ"]],
        "atol": 0.0,
    },
    "Mbpp/10002": {
        "entry_point": "is_palindrome",
        "prompt": (
            'def is_palindrome(s: str) -> bool:\n    """Return True iff s is a palindrome (case-sensitive).\n    """\n'
        ),
        "canonical_solution": ("    return s == s[::-1]\n"),
        "base_input": [["madam"], ["abc"], [""]],
        "plus_input": [["abba"], ["abca"], ["x"]],
        "atol": 0.0,
    },
    "Mbpp/10003": {
        "entry_point": "factorial",
        "prompt": ('def factorial(n: int) -> int:\n    """Compute n! for n >= 0.\n    """\n'),
        "canonical_solution": (
            "    if n < 2:\n"
            "        return 1\n"
            "    res = 1\n"
            "    for i in range(2, n+1):\n"
            "        res *= i\n"
            "    return res\n"
        ),
        "base_input": [[0], [1], [3], [5]],
        "plus_input": [[6], [7]],
        "atol": 0.0,
    },
    "Mbpp/10004": {
        "entry_point": "count_vowels",
        "prompt": (
            'def count_vowels(s: str) -> int:\n    """Return number of vowels in s (aeiou, lowercase only).\n    """\n'
        ),
        "canonical_solution": ("    return sum(ch in 'aeiou' for ch in s)\n"),
        "base_input": [["abc"], ["sky"], ["aeiou"]],
        "plus_input": [["banana"], ["x"], [""]],
        "atol": 0.0,
    },
    "Mbpp/10005": {
        "entry_point": "gcd",
        "prompt": (
            'def gcd(a: int, b: int) -> int:\n    """Greatest common divisor of a and b (non-negative ints)."""\n'
        ),
        "canonical_solution": ("    while b:\n        a, b = b, a % b\n    return abs(a)\n"),
        "base_input": [[12, 18], [7, 5], [0, 10]],
        "plus_input": [[100, 25], [27, 9]],
        "atol": 0.0,
    },
    "Mbpp/10006": {
        "entry_point": "is_prime",
        "prompt": ('def is_prime(n: int) -> bool:\n    """Return True iff n is a prime number (n >= 0)."""\n'),
        "canonical_solution": (
            "    if n < 2:\n"
            "        return False\n"
            "    if n % 2 == 0:\n"
            "        return n == 2\n"
            "    i = 3\n"
            "    while i * i <= n:\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "        i += 2\n"
            "    return True\n"
        ),
        "base_input": [[0], [1], [2], [3], [4], [17]],
        "plus_input": [[19], [21], [97]],
        "atol": 0.0,
    },
    "Mbpp/10007": {
        "entry_point": "second_largest",
        "prompt": (
            "def second_largest(arr: list) -> int:\n"
            '    """Return the second largest distinct value in arr.\n'
            "    Raise ValueError if fewer than 2 distinct values.\n"
            '    """\n'
        ),
        "canonical_solution": (
            "    s = sorted(set(arr))\n"
            "    if len(s) < 2:\n"
            "        raise ValueError('need 2 distinct values')\n"
            "    return s[-2]\n"
        ),
        "base_input": [[[1, 2, 3]], [[10, 10, 9]], [[5, 5, 5, 5, 6]]],
        "plus_input": [[[7, 7, 8, 9]], [[-1, -2, -3, -4]]],
        "atol": 0.0,
    },
    "Mbpp/10008": {
        "entry_point": "unique_preserve",
        "prompt": (
            "def unique_preserve(items: list) -> list:\n"
            '    """Return a list with duplicates removed, preserving first appearance order."""\n'
        ),
        "canonical_solution": (
            "    seen = set()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
        "base_input": [[[1, 1, 2, 2, 3]], [["a", "a", "b"]]],
        "plus_input": [[[0, 0, 0]], [[1, 2, 3]]],
        "atol": 0.0,
    },
    "Mbpp/10009": {
        "entry_point": "sum_of_squares",
        "prompt": ('def sum_of_squares(n: int) -> int:\n    """Return 1^2 + 2^2 + ... + n^2 for n >= 0."""\n'),
        "canonical_solution": ("    return sum(i*i for i in range(1, n+1))\n"),
        "base_input": [[0], [1], [3]],
        "plus_input": [[5], [10]],
        "atol": 0.0,
    },
    "Mbpp/10010": {
        "entry_point": "flatten_once",
        "prompt": (
            'def flatten_once(x: list) -> list:\n    """Flatten a single level of nesting: [[1,2],[3]] -> [1,2,3]."""\n'
        ),
        "canonical_solution": (
            "    out = []\n"
            "    for sub in x:\n"
            "        if isinstance(sub, list):\n"
            "            out.extend(sub)\n"
            "        else:\n"
            "            out.append(sub)\n"
            "    return out\n"
        ),
        "base_input": [[[[1, 2], [3]]], [[[]]], [[1, [2, 3]]]],
        "plus_input": [[[["a"], ["b", "c"]]], [[[1], 2, 3]]],
        "atol": 0.0,
    },
}
