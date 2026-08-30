#!/usr/bin/env python3
"""Time naive vs unique-vocab spell correction vs text length. Real SpellChecker."""
from __future__ import annotations
import time
from sklearn.feature_extraction.text import CountVectorizer
from spellchecker import SpellChecker
from stock_ds.spellcorrect import correct_text, correct_text_naive
from stock_ds.spellcorrect.reconstruct import TOKEN_RE

PARAGRAPH = (
    "Yoou cannot end a sentence with becausse becuse becaus is a conjunction. "
    "The quik brown fox jumpps over the laazy dog and then comes back home. "
    "People write letters about family, friends, school, music, and weather. "
    "Farmers harvest wheat, barley, oats, and corn during autumn. "
    "Musicians practice scales, songs, pieces, and dances every morning. "
    "Sailors travel rivers, lakes, islands, and coasts with great care. "
)

def n_tokens(text):
    return sum(1 for _ in TOKEN_RE.finditer(text))

def n_unique(text):
    cv = CountVectorizer()
    try:
        cv.fit([text])
    except ValueError:
        return 0
    return len(cv.vocabulary_)

def fmt(seconds):
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.3f} s"

def main():
    sc = SpellChecker()
    sc.correction("becausse")
    print(f"seed: {n_tokens(PARAGRAPH)} tokens, {n_unique(PARAGRAPH)} unique types\n")
    rows = []
    for n in [1, 15, 70, 150, 320]:
        text = PARAGRAPH * n
        t0 = time.perf_counter()
        correct_text(text, spellchecker=sc)
        t_unique = time.perf_counter() - t0
        t0 = time.perf_counter()
        correct_text_naive(text, spellchecker=sc)
        t_naive = time.perf_counter() - t0
        rows.append((len(text), n_tokens(text), n_unique(text), t_naive, t_unique, t_naive / t_unique if t_unique else float("inf")))
    print("| n_chars | n_tokens | n_unique | t_naive | t_unique | speedup |")
    print("| --- | --- | --- | --- | --- | --- |")
    for chars, tokens, unique, t_naive, t_unique, speedup in rows:
        print(f"| {chars} | {tokens} | {unique} | {fmt(t_naive)} | {fmt(t_unique)} | {speedup:.1f}x |")

if __name__ == "__main__":
    main()
