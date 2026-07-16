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


# --------------------------------------------------------------------------- #
# Title-only coverage: spine reads concatenate title+author+series, which used
# to dilute the score below min_match even when retrieval found the right book.
# All reads here are real, off the same bad capture.
# --------------------------------------------------------------------------- #
def test_concatenated_spine_read_scores_its_own_title(ocr):
    """Title + author + series + volume in one read. The old symmetric ratio
    scored this ~0.33 against its own title; title coverage must lift it."""
    read = "THE GAIE OF THE FERALC GODS DINNMMAN MAuL DUNGEON CRAWLER CARL Book FOUR"
    assert ocr.match_score(read, "The Gate of the Feral Gods") > 0.45


def test_two_clean_tokens_beat_the_wrong_series_entry(ocr):
    """'CARL DUNGEON' was scored 0.438-0.444 against the right books and lost to
    nothing. It must clear the bar for the right title and still lose the wrong
    one from the same series/author."""
    right = ocr.match_score("CARL DUNGEON", "Dungeon Crawler Carl")
    wrong = ocr.match_score("CARL DUNGEON", "Carl's Doomsday Scenario")
    assert right > 0.45 > wrong


def test_single_wrong_letter_still_scores_high(ocr):
    assert ocr.match_score("THIS INEVIIABLE RVIN", "This Inevitable Ruin") > 0.45
    assert ocr.match_score("THE EYE OF IHE BEDLAM BrIDE",
                           "The Eye of the Bedlam Bride") > 0.9


def test_old_false_positives_now_rejected(ocr):
    """Both shipped in a real titles.json at 0.47-0.49. A short title 'covered'
    by a longer read is little evidence; a fragment covering little of a long
    title is none."""
    assert ocr.match_score("Spanish DUNES", "Dune") < 0.45
    assert ocr.match_score("dic tion-ar", "Italian diction for singers") < 0.45


def test_scattered_single_chars_are_not_evidence(ocr):
    """OR-union retrieval surfaces unrelated popular titles; stray shared
    characters must not add up to a match."""
    assert ocr.match_score("THIS INEVIIABLE RVIN", "This Thing Called love") < 0.45
    assert ocr.match_score("THE EYE OF IHE BEDLAM BrIDE", "The Heiress Bride") < 0.45
    assert ocr.match_score("THE EYE OF IHE BEDLAM BrIDE", "The Bluest Eye") < 0.45


def test_one_shared_word_is_not_a_match(ocr):
    """A single common word ('spanish', 'ruin', 'war') fully matched inside a
    short title cleared 0.45 on the first fixed run. All real false positives."""
    assert ocr.match_score("Spanish DUNES", "The Spanish Groom") < 0.45
    assert ocr.match_score("THIS INEVIIABLE DMlln RUIN", "Ruins") < 0.45
    assert ocr.match_score("FYTHMt WAR", "The Poppy War") < 0.45


def test_one_word_plus_scraps_is_not_a_match(ocr):
    """The character-overlap view needs one contiguous 8-letter run: a shared
    word plus 2-3 char scraps ('spanish'+'un', 'the'+'ed'+'bride') let these
    through a real run at 0.46-0.59."""
    assert ocr.match_score("Spanish DUNES", "The Spanish Uncle") < 0.45
    assert ocr.match_score("Spanish DUMMIL", "A Stormy Spanish Summer") < 0.45
    assert ocr.match_score("ThL LYE OHIHL BEDLAM BRIDE", "The Forced Bride") < 0.45
    assert ocr.match_score("The EYL Or IHE BEDLAMA BRIDE", "The Forced Bride") < 0.45
    # ... while the true book must keep beating them on the same reads
    assert ocr.match_score("ThL LYE OHIHL BEDLAM BRIDE",
                           "The Eye of the Bedlam Bride") > 0.45
    assert ocr.match_score("The EYL Or IHE BEDLAMA BRIDE",
                           "The Eye of the Bedlam Bride") > 0.45


def test_low_confidence_token_matches_are_not_evidence(ocr):
    """'winter'~'ineae' at 0.55 and 'boomerang'~'man' at 0.50 pushed these over
    the evidence floor on a real run; sub-0.7 token matches may shade coverage
    but must not count as proof."""
    assert ocr.match_score("INEAE OHINL BEDLAM BRIDE", "The Winter Bride") < 0.45
    assert ocr.match_score(
        "THHE GYE OF THE Man BEDLAM BRIDE DINNMMAN CanL Hocnai NuEON Taiim",
        "Boomerang Bride") < 0.45


def test_author_confirmation_lifts_a_short_title(ocr):
    """'Ie Holt 0 Stcphchic Mctce' is The Host + Stephenie Meyer. Four letters
    of title alone must not match -- but title plus a substantially-present
    author name is real evidence. The same read must NOT match Holt Physics
    (title fits, authors absent), and a half-matched 4-letter surname fragment
    must not rescue Blue Ruin."""
    read = "Ie Holt 0 Stcphchic Mctce"
    assert ocr.match_score(read, "Host", ["Stephenie Meyer"]) > 0.45
    assert ocr.match_score(read, "Host") < 0.45
    assert ocr.match_score(read, "Holt Physics",
                           ["Raymond A. Serway", "Jerry S. Faughn"]) < 0.45
    assert ocr.match_score("WHIS NEVIABLE RUIN MMAMiLAN", "Blue Ruin",
                           ["Grace Livingston Hill"]) < 0.45


def test_fuller_title_outranks_contained_shorter_title(ocr):
    """'Inevitable' is fully present in 'THIS INEVITABLE RVIN', but the title
    that also explains the residue must win the candidate set."""
    read = "THIS INEVITABLE RVIN"
    full = ocr.match_score(read, "This Inevitable Ruin")
    part = ocr.match_score(read, "Inevitable")
    assert full > part


# --------------------------------------------------------------------------- #
# Suspect tokens: general shape signals only (digits, vowel-starved runs,
# case flapping, stutters) -- no dictionary, no shelf knowledge.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", ["DiNmHan", "X3u", "RIST6", "NHLKFN",
                               "Henanaeeeee", "8AGFPN", "0duimmi"])
def test_junk_tokens_are_suspect(ocr, t):
    assert ocr.token_suspicion(t) >= ocr.SUSPECT


@pytest.mark.parametrize("t", ["DUNGEON", "Masquerade", "BEDLAM", "Rowling",
                               "The", "JK", "INEVIIABLE", "FERALC"])
def test_plausible_tokens_are_not_suspect(ocr, t):
    """Includes garbled-but-wordlike tokens (INEVIIABLE, FERALC): shape signals
    cannot catch those, which is exactly why the ladder has an OR-union rung."""
    assert ocr.token_suspicion(t) < ocr.SUSPECT


# --------------------------------------------------------------------------- #
# Query ladder: bounded degradation of one read into search attempts.
# --------------------------------------------------------------------------- #
def test_ladder_starts_verbatim_is_bounded_and_deduped(ocr):
    lad = ocr.build_query_ladder("THE EYE OF IHE BEDLAM BrIDE")
    assert lad[0] == ("q", "THE EYE OF IHE BEDLAM BrIDE")  # cache-compatible
    assert len(lad) <= 6
    assert len({(k, q.lower()) for k, q in lad}) == len(lad)


def test_ladder_or_union_drops_stopwords_keeps_content(ocr):
    lad = ocr.build_query_ladder("THE EYE OF IHE BEDLAM BrIDE")
    assert ("q_or", "EYE OR IHE OR BEDLAM OR BrIDE") in lad


def test_ladder_pairs_prefer_clean_long_tokens(ocr):
    lad = ocr.build_query_ladder(
        "THE GAIE OF THE FERALC GODS DINNMMAN MAuL DUNGEON CRAWLER CARL Book FOUR")
    pairs = [q for k, q in lad if k == "q" and len(q.split()) == 2]
    assert pairs and pairs[0] == "DUNGEON CRAWLER"


def test_ladder_degenerate_input(ocr):
    assert ocr.build_query_ladder("") == []
    assert ocr.build_query_ladder("Jonan") == [("q", "Jonan")]  # nothing to degrade


# --------------------------------------------------------------------------- #
# Ladder walk: all network mocked. Response shapes mirror what the live Open
# Library API returned for these exact queries (verified 2026-07-16): the AND
# search dies on one garbled token, the OR-union ranks the right book on top.
# --------------------------------------------------------------------------- #
def test_walk_retrieves_garbled_read_via_or_union(ocr):
    read = "THE EYE OF IHE BEDLAM BrIDE"   # one letter wrong -> q= finds 0
    calls = []

    def search(kind, q):
        calls.append((kind, q))
        if kind == "q_or":
            return [{"title": "The Eye of the Bedlam Bride"},
                    {"title": "The Heiress Bride"},
                    {"title": "The Bluest Eye"}]
        return []  # AND semantics: any garbled token -> zero docs

    best, tried = ocr.walk_ladder(read, search, min_match=0.45)
    assert best is not None and best["title"] == "The Eye of the Bedlam Bride"
    assert best["score"] >= 0.9
    assert best["retrieved_by"]["kind"] == "q_or"
    assert tried == 2 and calls[0][0] == "q"  # stopped at the first hit


def test_walk_scores_against_full_read_not_subquery(ocr):
    """The real 'Spanish DUNES' misread. Even if every rung returns Dune, it
    must be scored against the whole read and rejected -- a narrower sub-query
    cannot inflate the match."""
    best, _ = ocr.walk_ladder("Spanish DUNES", lambda k, q: [{"title": "Dune"}],
                              min_match=0.45)
    assert best is None


def test_walk_is_bounded_and_returns_none_when_hopeless(ocr):
    read = "THIS INEVIIABLE RVIN"  # 2 of 3 tokens garbled; junk comes back
    calls = []

    def search(kind, q):
        calls.append((kind, q))
        return [{"title": "This Thing Called love"},
                {"title": "This Side of Paradise"}]

    best, tried = ocr.walk_ladder(read, search, min_match=0.45)
    assert best is None
    assert tried == len(calls) <= 6
