"""Which language to answer in, decided from the message itself rather than by the model.

Measured on 25 Nov 2026: the small model often labelled Hinglish (Hindi written in English letters) as Hindi. The three replies we have are
fixed texts, so the choice only needs the script and a few common words.
"""

from __future__ import annotations

import re

# Words that appear in Hindi written with English letters and are not ordinary English words.
STRONG = frozenset("hai hain chahiye karna karne karein mere meri mera aur wala wale wali liye nahi nahin kyunki lekin".split())
# Common but short or ambiguous: two of them are needed.
WEAK = frozenset("ka ke ki ko kar karo kuch bhi jo ya sab apne apna uska unko dene banana banao nikalna padh batao baant sawal jawab"
                 " achha bura gaya jaye thoda likhne likhni bataye dukaan baaki bahut kaun".split())
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def detect_language(message: str) -> str:
    """"hi" for Devanagari, "hinglish" for Hindi in English letters, otherwise "en"."""
    if _DEVANAGARI.search(message):
        return "hi"
    words = set(re.findall(r"[a-z]+", message.lower()))
    if words & STRONG or len(words & WEAK) >= 2:
        return "hinglish"
    return "en"
