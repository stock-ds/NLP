"""Tests for sc_correct_dictionary."""
from stock_ds.dictionary_tools import sc_correct_dictionary
from stock_ds.spellcorrect.reconstruct import reconstruct_text
from sklearn.feature_extraction.text import CountVectorizer

FAKE_MAP = {"yoou": "you", "becausse": "because", "becuse": "because", "becaus": "because"}

def fake_correct(word):
    return FAKE_MAP.get(word, word)

def test_sc_correct_dictionary_shares_ids():
    vocab = {"becausse": 0, "becuse": 1, "becaus": 2, "hello": 3}
    remap, corrections = sc_correct_dictionary(vocab, correct_word=fake_correct)
    assert remap["becausse"] == remap["becuse"] == remap["becaus"]
    assert remap["hello"] != remap["becausse"]
    assert corrections == {"becausse": "because", "becuse": "because", "becaus": "because"}
    assert "hello" not in corrections

def test_sc_correct_dictionary_with_countvectorizer_roundtrip():
    sentence = "Yoou cannot end a sentence with becausse becuse becaus is a conjunction."
    cv = CountVectorizer().fit([sentence])
    remap, corrections = sc_correct_dictionary(cv.vocabulary_, correct_word=fake_correct)
    assert remap["becausse"] == remap["becuse"] == remap["becaus"]
    stitched = reconstruct_text(sentence, corrections)
    assert stitched.startswith("you ")
    assert "because because because" in stitched
