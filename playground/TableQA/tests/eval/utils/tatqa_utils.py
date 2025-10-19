import re
import string
from typing import List
import numpy as np


def scale_to_num(scale):
    scale = scale.lower()
    num = 1
    if "hundred" in scale:  # hundred
        num = 100
    elif "thousand" in scale:  # thousand
        num = 1000
    elif "million" in scale:  # million
        num = 1000000
    elif "billion" in scale:  # billion
        num = 1000000000
    elif "percent" in scale:  # percent
        num = 0.01
    return num


def extract_one_num_from_str(s):
    s = _clean_num(s)
    r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
    groups = re.findall(r_num, s)
    if len(groups) == 0:
        return None
    num = groups[0][0]
    if num == "":
        return None
    if "." in num:
        return float(num)
    return int(num)


EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"


def _clean_num(text: str):
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def is_number(text: str) -> bool:
    try:
        words = " ".join([_clean_num(w) for w in text.split()]).split()
        if len(words) == 0:
            """1023 or 1 million"""
            return False
        num = float(words[0])
        if np.isnan(num):
            return False
        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False
        return True
    except ValueError:
        return False
    # except AttributeError:
    #     return False


def negative_num_handle(x):
    """
    :param x:  transform (134) -> -134
    :return:
    """
    all = re.findall("(\([\d.\s]+\))", x.strip())
    if len(all) > 0:
        return -1
    return 1


def percent_num_handle(x):
    """
    :param x:  transform 12% -> 12/100
    :return:
    """
    all = re.findall("([\d.\s]+%)", x.strip())
    if len(all) > 0:
        return 0.01
    return 1


def word_scale_handle(x):
    """
    :param x: 1 million = 1,000,000
    :return:
    """
    iter = re.finditer("([\d.]+\s?[a-zA-Z]+)", x)
    for one in iter:
        text = one.group(0).lower()
        scale_val = scale_to_num(text)
        return scale_val
    return 1


def to_number(text: str) -> float:
    num = extract_one_num_from_str(text)
    scale_val = word_scale_handle(text)
    negative_flag = negative_num_handle(text)
    percent_flag = percent_num_handle(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None


def remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def remove_punc(text: str) -> str:
    if not is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def lower(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> List[str]:
    return re.split(" ", text)


def normalize_number(text: str) -> str:
    if is_number(text):
        return str(to_number(text))
    else:
        return text


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [
        white_space_fix(remove_articles(normalize_number(remove_punc(lower(token)))))
        for token in tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])


def ws_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip().lower()
    if not text:
        return []
    text = white_space_fix(text)
    tokens = text.split()
    tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
    return tokens
