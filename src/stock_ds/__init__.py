"""stock-ds: appendable data-science helper toolkit (NLP first)."""

from stock_ds.dictionary_tools import sc_correct_dictionary
from stock_ds.spellcorrect import (
    correct_text,
    correct_text_naive,
    correct_vocabulary,
)

__version__ = "0.1.0"

__all__ = [
    "correct_text",
    "correct_text_naive",
    "correct_vocabulary",
    "sc_correct_dictionary",
    "__version__",
]
