#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Spine-read -> book-title matching. Pure text, no ML stack.

Split out of ocr_titles.py for the same reason splits.py is split out of the
classifier: the interesting logic here is the precision policy, and it should be
unit-testable without importing cv2 and easyocr. ocr_titles.py imports from here.

The policy is deliberately conservative. OCR on a book spine is noisy at the best of
times -- vertical text, glare, stylised fonts -- and a lookup that answers confidently
and wrongly is worse than one that shrugs. One bad match discredits every good one.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# A spine read has to carry some actual signal before it is worth a network call and
# a chance at a false positive. Real examples from one shelf that should never have
# been queried at all: "1 1", "0 1 0", "Jonan". Letters, not characters -- OCR noise
# on a spine is mostly digits and punctuation.
MIN_QUERY_LETTERS = 8


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", (s or "").lower()).strip()


def match_score(ocr_text: str, title: str) -> float:
    """Fuzzy similarity in [0,1] between an OCR read and a candidate title.

    stdlib difflib (no extra dep), plus a containment credit -- but only in proportion
    to how much of the title the read actually accounts for.

    That coverage term is load-bearing. Crediting containment a flat 0.9 (the previous
    behaviour) means any short fragment that happens to fall inside a long title clears
    min_match: on a real shelf, a spine that OCR'd as "Jonan" was confidently matched to
    "Jonan & evolusi kereta api Indonesia", a book about Indonesian railways. Cover 15%
    of the title, get 15% of the credit.
    """
    a, b = _norm(ocr_text), _norm(title)
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    if b in a or a in b:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        ratio = max(ratio, 0.9 * (len(shorter) / len(longer)))
    return round(ratio, 3)


def is_queryable(query: str) -> bool:
    """True if a cleaned OCR read has enough alphabetic signal to be worth looking up."""
    return sum(c.isalpha() for c in query) >= MIN_QUERY_LETTERS
