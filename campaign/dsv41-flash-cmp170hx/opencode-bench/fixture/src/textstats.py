"""Small text statistics helpers used by the CLI."""
from collections import Counter


def word_frequencies(text):
    words = text.lower().split()
    return Counter(words)


def top_words(text, n):
    freqs = word_frequencies(text)
    # BUG: ties are not broken by word order; also crashes on n <= 0
    return [w for w, _ in freqs.most_common(n)]


def running_mean(values):
    out = []
    total = 0
    for i, v in enumerate(values):
        total += v
        out.append(total / i)   # BUG: should be i + 1
    return out
