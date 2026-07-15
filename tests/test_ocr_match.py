"""Spine-read -> book match scoring.

These fixtures are real: every OCR string here came off an actual bookshelf capture
(a deliberately bad one — auto-exposure drifted 92% — which is exactly why the reads
are noisy and why precision matters). The false positive in
test_short_fragment_does_not_match_long_title actually happened.

The module imports cv2/easyocr at top level, so these tests exercise the pure scoring
functions by loading the source and exec'ing only what they need — no ML stack.
"""
import pytest


@pytest.fixture
def ocr(load_module):
    """understanding/matching.py — no cv2, no easyocr, no torch. That is the point."""
    return load_module("understanding/matching.py", "matching")


def test_short_fragment_does_not_match_long_title(ocr):
    """The regression. A 5-letter read is a substring of this title, and a flat
    containment credit scored it 0.9 — confidently matching a book about Indonesian
    railways to a fantasy paperback."""
    score = ocr.match_score("Jonan", "Jonan & evolusi kereta api Indonesia")
    assert score < 0.45, f"scored {score}; min_match default is 0.45"


def test_real_read_still_matches_its_real_title(ocr):
    """Precision must not have cost us recall on the reads that are actually good."""
    score = ocr.match_score("HBVTCHERS DiNmHan MASQUERADE", "The Butcher's Masquerade")
    assert score > 0.45, f"scored {score}; this is a genuine read of this book"


def test_clean_containment_still_scores_high(ocr):
    """A read that IS the title, with noise around it, should still be credited."""
    assert ocr.match_score("this inevitable ruin", "This Inevitable Ruin") > 0.9


def test_coverage_scales_the_containment_credit(ocr):
    """Same contained string, longer title -> less credit. That is the whole fix."""
    tight = ocr.match_score("the gate of the feral gods", "The Gate of the Feral Gods")
    loose = ocr.match_score("the gate of the feral gods",
                            "The Gate of the Feral Gods and Other Stories, Collected, Annotated")
    assert tight > loose


def test_empty_and_garbage_score_zero(ocr):
    assert ocr.match_score("", "Anything") == 0.0
    assert ocr.match_score("Anything", "") == 0.0


@pytest.mark.parametrize("q", ["1 1", "0 1 0", "Jonan", "VIB Ei", ""])
def test_degenerate_reads_are_not_queryable(ocr, q):
    """Real noise off a real shelf. None of these deserves a network call."""
    assert not ocr.is_queryable(q)


@pytest.mark.parametrize("q", [
    "MASQUERADE Ea DINNIMAN",
    "THIS INEVIIABLE Dinmhan RUIN",
    "THE EYE OnThg BEDLAM BRIDE DIMAMAN",
    "MuM RIST6 Robert Jordan",
])
def test_real_noisy_reads_are_queryable(ocr, q):
    """Noisy but genuinely informative reads must survive the gate."""
    assert ocr.is_queryable(q)
