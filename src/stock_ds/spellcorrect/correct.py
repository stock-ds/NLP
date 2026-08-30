"""Unique-vocabulary spell corrector plus a naive (per-token) baseline."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Optional, Union

from sklearn.feature_extraction.text import CountVectorizer
from stock_ds.spellcorrect.reconstruct import TOKEN_PATTERN, iter_token_spans, reconstruct_text

CorrectWord = Callable[[str], Optional[str]]
VocabLike = Union[Mapping[str, int], Iterable[str]]
_DEFAULT_SC = None

def _get_default_spellchecker():
    global _DEFAULT_SC
    if _DEFAULT_SC is None:
        from spellchecker import SpellChecker
        _DEFAULT_SC = SpellChecker()
    return _DEFAULT_SC

def _truncate(word, max_word_len):
    if max_word_len is None or max_word_len <= 0:
        return word
    return word[:max_word_len]

def _spell_backend(spellchecker=None, max_word_len=30, correct_word=None):
    cap = max_word_len
    if correct_word is not None:
        def correct_one(word):
            result = correct_word(_truncate(word, cap))
            return word if result is None else result
        def unknown_of(words):
            return set(words)
        return correct_one, unknown_of
    sc = spellchecker if spellchecker is not None else _get_default_spellchecker()
    def correct_one(word):
        result = sc.correction(_truncate(word, cap))
        return word if result is None else result
    def unknown_of(words):
        word_list = list(words)
        unknown_lower = {item.lower() for item in sc.unknown(word_list)}
        return {w for w in word_list if w.lower() in unknown_lower}
    return correct_one, unknown_of

def _as_word_list(vocab):
    if isinstance(vocab, Mapping):
        return list(vocab.keys())
    return list(vocab)

def correct_vocabulary(vocab, spellchecker=None, max_word_len=30, verbose=False, correct_word=None):
    words = _as_word_list(vocab)
    correct_one, unknown_of = _spell_backend(spellchecker=spellchecker, max_word_len=max_word_len, correct_word=correct_word)
    unique_words = list(dict.fromkeys(words))
    unknown = unknown_of(unique_words)
    unique_corrections = {}
    n_unknown = 0
    for word in unique_words:
        if word in unknown:
            unique_corrections[word] = correct_one(word)
            if verbose:
                if n_unknown % 10 == 0:
                    print(f"Corrected {n_unknown} / {len(unknown)}")
                n_unknown += 1
        else:
            unique_corrections[word] = word
    id_remap, seen, corrections, next_id = {}, {}, {}, 0
    for word in words:
        form = unique_corrections[word]
        if form not in seen:
            seen[form] = next_id
            next_id += 1
        id_remap[word] = seen[form]
        if form != word:
            corrections[word] = form
    return id_remap, corrections

def _vocabulary_from_text(text):
    vectorizer = CountVectorizer()
    try:
        vectorizer.fit([text])
    except ValueError:
        return {}
    return vectorizer.vocabulary_

def correct_text(text, spellchecker=None, max_word_len=30, verbose=False, correct_word=None):
    if not text:
        return text
    vocab = _vocabulary_from_text(text)
    if not vocab:
        return text
    _, corrections = correct_vocabulary(vocab, spellchecker=spellchecker, max_word_len=max_word_len, verbose=verbose, correct_word=correct_word)
    return reconstruct_text(text, corrections)

def correct_text_naive(text, spellchecker=None, max_word_len=30, verbose=False, correct_word=None):
    if not text:
        return text
    correct_one, unknown_of = _spell_backend(spellchecker=spellchecker, max_word_len=max_word_len, correct_word=correct_word)
    parts, last, n_done = [], 0, 0
    for start, end, token in iter_token_spans(text):
        key = token.lower()
        if key in unknown_of([key]):
            replacement = correct_one(key)
            if verbose and n_done % 10 == 0:
                print(f"Corrected {n_done} occurrences")
            n_done += 1
            if replacement != key:
                parts.append(text[last:start])
                parts.append(replacement)
                last = end
    parts.append(text[last:])
    return "".join(parts)

def main(argv=None):
    parser = argparse.ArgumentParser(prog="stock-ds-spellcorrect", description="Correct spelling with unique-vocabulary SpellChecker.")
    parser.add_argument("file", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--max-word-len", type=int, default=30)
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    fn = correct_text_naive if args.naive else correct_text
    sys.stdout.write(fn(args.file.read(), max_word_len=args.max_word_len, verbose=args.verbose))
    return 0

DEFAULT_TOKEN_PATTERN = TOKEN_PATTERN
