"""Tests for unique-vocabulary spell correction. FakeCorrector so tests never download dictionaries."""
from __future__ import annotations
import time
from stock_ds.spellcorrect import correct_text, correct_text_naive, correct_vocabulary
from stock_ds.spellcorrect.reconstruct import reconstruct_text

README_SENTENCE = "Yoou cannot end a sentence with becausse becuse becaus is a conjunction."
FAKE_MAP = {"yoou": "you", "becausse": "because", "becuse": "because", "becaus": "because", "cannot": "cannot"}

def fake_correct(word):
    return FAKE_MAP.get(word, word)

class CountingCorrector:
    def __init__(self, mapping, delay=0.0):
        self.mapping = mapping
        self.delay = delay
        self.calls = 0
    def __call__(self, word):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self.mapping.get(word, word)

def test_unique_vocab_shares_ids():
    remap, corrections = correct_vocabulary(["becausse", "becuse", "becaus"], correct_word=fake_correct)
    assert remap["becausse"] == remap["becuse"] == remap["becaus"]
    assert set(remap.values()) == {0}
    assert corrections == {"becausse": "because", "becuse": "because", "becaus": "because"}

def test_id_remap_distinct_forms_get_distinct_ids():
    remap, corrections = correct_vocabulary(["becausse", "hello", "becuse"], correct_word=fake_correct)
    assert remap["becausse"] == remap["becuse"]
    assert remap["hello"] != remap["becausse"]
    assert "hello" not in corrections

def test_correct_text_readme_sentence_and_start_of_string():
    out = correct_text(README_SENTENCE, correct_word=fake_correct)
    assert not out.lower().startswith("yoou")
    assert out.startswith("you ")
    assert "because because because" in out
    assert "cannot" in out

def test_unique_and_naive_agree_on_short_string():
    assert correct_text(README_SENTENCE, correct_word=fake_correct) == correct_text_naive(README_SENTENCE, correct_word=fake_correct)

def test_reconstruct_preserves_surrounding_punctuation():
    assert correct_text("(becausse)!", correct_word=fake_correct) == "(because)!"
    assert reconstruct_text("(becausse)!", {"becausse": "because"}) == "(because)!"

def test_unchanged_tokens_keep_original_case():
    assert correct_text("Hello becausse WORLD", correct_word=fake_correct) == "Hello because WORLD"

def test_empty_and_punctuation_only():
    assert correct_text("", correct_word=fake_correct) == ""
    assert correct_text("!!!", correct_word=fake_correct) == "!!!"

def test_max_word_len_truncates_before_corrector():
    seen = []
    def capture(word):
        seen.append(word)
        return word
    correct_vocabulary(["x" * 40], correct_word=capture, max_word_len=30)
    assert seen == ["x" * 30]

def test_correct_word_none_keeps_original():
    remap, corrections = correct_vocabulary(["xyzzy"], correct_word=lambda w: None)
    assert remap == {"xyzzy": 0}
    assert corrections == {}

def test_unique_path_faster_than_naive_on_long_repeated_text():
    text = ("yoou cannot end a sentence with becausse becuse becaus is a conjunction. ") * 5000
    unique_fn = CountingCorrector(FAKE_MAP, delay=0.00015)
    naive_fn = CountingCorrector(FAKE_MAP, delay=0.00015)
    unique_out = correct_text(text, correct_word=unique_fn)
    naive_out = correct_text_naive(text, correct_word=naive_fn)
    assert unique_out == naive_out
    assert unique_fn.calls < naive_fn.calls
    assert unique_fn.calls < 50
    assert naive_fn.calls > 1000
