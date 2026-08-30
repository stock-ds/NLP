"""Regex helpers (original pandas-based API).

Copied from the stock-ds/NLP ``regex_tools.py``. ``Series.str.replace``
is called with ``regex=True`` so the original regex steps keep working
on pandas 2.x, where the default flipped to ``regex=False``.
"""

import re

import pandas as pd


def regex_sequence(ls, steps=["default_cleanup"]):
    r"""
    Apply a specified list of regex transformations in a squential order.
    Used to easily remove or replace regex-found sequences from text

    Args:
        ls: input list or column of pandas dataframe,
            for example: ['Text string... one', 'Text. string two']
        steps: list of regex replace steps. Make sure it's a list or None. (default = "default_cleanup")
            Each list element can be one of these types:
            1) string to replace identified text with " "
            2) 2-length tuple to replace text with a custom string
            3) (special case) "lowercase" to lowercase whole text
            If steps = None, no transformations will be performed.
            If steps = "default_cleanup", the defauls steps will be performed:
            1) '''[!"'#$%&()*+,-./:;<=>\[\]?@\^_`{|}~\\\]''' - remove all symbols except '
            2) " +" - remove repeating spaces
            3) "lowercase" - this is a pre-defined command
            4) (r"[a-z]\1\1+", r"\1\1") - limit repeating letters to two ('aaaaaannn' -> 'aann')
    Returns:
        Output regexed list, example: ['text*string*one','text*string*two']

    Examples:
        in_texts = [
            "The quick brown fox, jumps over the lazy dog.",
            "AAaaa"]
        steps = [
            "lowercase",
            ", ",
            (r"([a-z])\1\1+",
            r"\1\1")]
        print(regex_sequence(in_texts, steps=steps))
        >>> 0    the quick brown fox jumps over the lazy dog.
        >>> 1                                               aa
    """
    if steps == ["default_cleanup"]:
        steps = [
            r"""[!"'#$%&()*+,-./:;<=>\[\]?@\^_`{|}~\\]""",
            r" +",
            r"lowercase",
            (r"([a-z])\1\1+", r"\1\1")]
    elif steps is None:
        return ls

    if ls.__class__ is not pd.core.series.Series:
        ls = pd.DataFrame({"tmp": ls})["tmp"]

    for step in steps:
        if str(step).lower() in ["lowercase"]:
            ls = ls.str.lower()
        elif step.__class__ is str:
            ls = ls.str.replace(step, " ", regex=True)
        elif step.__class__ is tuple:
            # change regex, preserving ls:
            ls = pd.DataFrame(
                {"tmp": [re.sub(step[0], step[1], i) for i in ls]})["tmp"]
        else:
            print("Failed to replace ", step, ". Continuing...")
    return ls


def regex_around(text, regex_str, letters):
    """
    Extract specified number of characters around a specified string

    Args:
        text: input text.
        regex_str: regex string identifier.
        letters: number of letters to be extracted around regex_str.

    Returns:
        list of extracted items

    Examples:
        regex_around("The quick brown fox jumps over the lazy dog", "fox", 10)
        >>> ['ick brown fox jumps ove']
    """
    re_span = [i.span() for i in re.finditer(regex_str, text)]
    out_span = [(i[0]-letters, i[1]+letters) if i[0] > letters
                else (0, i[1]+letters) for i in re_span]
    out_re = [text[i[0]:i[1]] for i in out_span]
    return out_re
