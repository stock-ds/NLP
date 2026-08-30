"""Stitch corrected vocabulary items back into the original text.

CountVectorizer's default token pattern is ``(?u)\\b\\w\\w+\\b`` (Unicode
word characters, length >= 2). Reconstruction uses that same pattern so
the unique-vocab pass and the stitch-back agree on what a "word" is.

Strategy
--------
1. Find every token span in the original text with the CountVectorizer
   pattern (``re.finditer``, so start/end of string and punctuation
   around words are all valid match sites — unlike naive
   ``text.replace(" " + word + " ", ...)`` which misses a misspelling
   at the start of the string).
2. Look up each token **case-insensitively** in the corrections table.
   CountVectorizer lowercases its vocabulary by default, and
   pyspellchecker does too, so table keys are lowercase.
3. If a token changed, emit the correction (typically lowercase — we
   cannot restore original case once CountVectorizer has lowercased).
   Unchanged tokens, whitespace, and punctuation are copied verbatim,
   so ``"(becausse)!"`` becomes ``"(because)!"``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping

# sklearn.feature_extraction.text.CountVectorizer default token_pattern
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
TOKEN_RE = re.compile(TOKEN_PATTERN)


def iter_token_spans(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, token)`` for each CountVectorizer-style token."""
    for match in TOKEN_RE.finditer(text):
        yield match.start(), match.end(), match.group(0)


def reconstruct_text(text: str, corrections: Mapping[str, str]) -> str:
    """Whole-word substitution of corrected vocab items into *text*.

    Parameters
    ----------
    text:
        Original text (any mix of case / punctuation).
    corrections:
        Mapping of **original** token (as in the vocabulary, usually
        lowercase) to its correction. Only words that actually changed
        need to be present.

    Returns
    -------
    str
        *text* with misspelled tokens replaced. Surrounding punctuation
        and whitespace are preserved. Tokens that did not change keep
        their original spelling and case.
    """
    if not text or not corrections:
        return text

    table = {key.lower(): value for key, value in corrections.items()}
    parts: list[str] = []
    last = 0
    for start, end, token in iter_token_spans(text):
        replacement = table.get(token.lower())
        if replacement is not None and replacement != token:
            parts.append(text[last:start])
            parts.append(replacement)
            last = end
    parts.append(text[last:])
    return "".join(parts)
