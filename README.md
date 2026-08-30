# stock-ds

An appendable data-science helper toolkit, starting with NLP.

This remakes the 2019 [stock-ds/NLP](https://github.com/stock-ds/NLP) helpers: unique-vocabulary spell correction and small regex utilities. New tools should land under `src/stock_ds/`.

## The unique-vocabulary spell-correct idea

Naive spell correction over **every token** is slow: each `SpellChecker.correction` is a dictionary lookup plus edit-distance search, so cost is **O(tokens)**.

The 2019 idea in `dictionary_tools.sc_correct_dictionary`:

1. Fit sklearn `CountVectorizer` so you get **unique vocabulary names**.
2. Run SpellChecker only on those unique names (unknown words only).
3. Reconstruct the original text by substituting corrected vocab names.

Unique-word count grows much slower than total token count, so the expensive work is **O(unique types)** plus a cheap stitch-back.

This shines when the same typos repeat (long letters, OCR of a repeated vocabulary). It does **not** help much if almost every token is a unique garbage string.

### Benchmark note

A local bench that **repeats one 56-type paragraph** hit ~33x at ~9,300 tokens / ~62k characters (naive 347 ms vs unique 10.5 ms). Unique types stayed flat at 56. That is the best case, not a 50-page diverse OCR scan. On tiny strings the unique path can be slightly *slower* (CountVectorizer overhead).

### Known limitation of the corrector

pyspellchecker picks the highest-frequency candidate, which is sometimes wrong (the original README had `cannot` → `cannon`). That is the corrector, not the unique-vocab speedup. Unique and naive apply the same corrector.

### Reconstruction

The old README used `sentence.replace(" " + key + " ", ...)`, which misses a misspelling at the **start** of the string (`Yoou` stayed `yoou`). The new reconstruct uses the same CountVectorizer token pattern and whole-word spans, so start/end of text and glue like `(becausse)!` → `(because)!` work. Corrections are lowercase (CountVectorizer lowercases). Unchanged tokens keep original case.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.9+. Dependencies: scikit-learn, pyspellchecker, pandas (pandas only for `regex_tools`).

## Quick example

```python
from sklearn.feature_extraction.text import CountVectorizer
from stock_ds.dictionary_tools import sc_correct_dictionary
from stock_ds.spellcorrect import correct_text
from stock_ds.spellcorrect.reconstruct import reconstruct_text

sentence = "Yoou cannot end a sentence with becausse becuse becaus is a conjunction."
cv = CountVectorizer().fit([sentence])
remap, corrections = sc_correct_dictionary(cv.vocabulary_)
print(remap, corrections)
print(reconstruct_text(sentence, corrections))
print(correct_text(sentence))
```

`becausse` / `becuse` / `becaus` share one id (`because`). `yoou` at the start becomes `you`.

```python
from stock_ds.spellcorrect import correct_text, correct_text_naive, correct_vocabulary
from stock_ds.dictionary_tools import sc_correct_dictionary
```

CLI:

```bash
echo "Yoou cannot end a sentence with becausse." | stock-ds-spellcorrect
pytest -q
python benchmarks/bench_spellcorrect.py
python examples/correct_sentence.py
```

Root shims (`dictionary_tools.py`, `regex_tools.py`) keep old imports working if the repo root is on `PYTHONPATH`.

## regex_tools

```python
from stock_ds.regex_tools import regex_sequence, regex_around

regex_sequence(["The quick brown fox, jumps."], steps=["lowercase", ", "])
regex_around("The quick brown fox jumps over the lazy dog", "fox", 10)
```

## Adding another tool

1. Module under `src/stock_ds/`.
2. Export from `src/stock_ds/__init__.py`.
3. Tests under `tests/`.
4. Optional root shim for old import paths.
5. Document it here.
