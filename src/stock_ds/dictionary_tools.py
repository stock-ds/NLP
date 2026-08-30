"""Dictionary helpers. ``sc_correct_dictionary`` is a thin wrapper around
the unique-vocabulary corrector (no pandas, no deprecated DataFrame.append).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Optional, Union

from stock_ds.spellcorrect.correct import correct_vocabulary

CorrectWord = Callable[[str], Optional[str]]


def sc_correct_dictionary(
    dictionary: Union[Mapping[str, int], list[str], tuple[str, ...]],
    spellchecker=None,
    max_word_len: Optional[int] = 30,
    verbose: bool = False,
    correct_word: Optional[CorrectWord] = None,
) -> tuple[dict[str, int], dict[str, str]]:
    """Correct a vocabulary dict using unique-word SpellChecker.

    Remake of the 2019 ``sc_correct_dictionary`` without pandas. The
    original truncated words to 30 characters for SpellChecker; that
    safety cap is still the default (pass ``max_word_len=None`` to
    disable).

    Parameters
    ----------
    dictionary:
        Typically ``CountVectorizer.vocabulary_`` (word -> old id), or
        any iterable of words.
    spellchecker, max_word_len, verbose, correct_word:
        Forwarded to :func:`stock_ds.spellcorrect.correct.correct_vocabulary`.

    Returns
    -------
    dict_out, corrections_out:
        Same contract as the original: integer ids shared by words that
        correct to the same form, and a dict of only the words that
        changed.
    """
    return correct_vocabulary(
        dictionary,
        spellchecker=spellchecker,
        max_word_len=max_word_len,
        verbose=verbose,
        correct_word=correct_word,
    )
