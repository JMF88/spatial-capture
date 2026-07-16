#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Spine-read -> book-title matching. Pure text, no ML stack.

Split out of ocr_titles.py for the same reason splits.py is split out of the
classifier: the interesting logic here is the precision policy, and it should be
unit-testable without importing cv2 and easyocr. ocr_titles.py imports from here.

Two halves, both pure:

  * SCORING (match_score): how well does a candidate title explain an OCR read?
    A spine read is usually TITLE + AUTHOR + SERIES concatenated, with per-token
    garbling. A symmetric string ratio dilutes the true title with that extra
    material, so we take the max of two views:
      - sequence overlap normalized by the LONGER string (strict; an exactly
        contained fragment earns exactly its coverage, which keeps the "Jonan"
        regression fixed), and
      - title-token coverage: how much of the title's content the read accounts
        for, token by token, fuzzy per token, scaled by how many matched letters
        actually back the claim (a 4-letter title "matched" inside a longer read
        is 4 letters of evidence, not a match).
    Stopwords carry almost no weight in coverage — every title shares them.

  * RETRIEVAL PLANNING (build_query_ladder / walk_ladder): Open Library's
    search is AND-keyword matching — verified live: one garbled token returns
    zero docs, which is how most legible reads died. The ladder degrades a read
    into a bounded sequence of attempts (full query, OR-union of content
    tokens, title-field query of clean tokens, clean token pairs) and stops at
    the first attempt whose best candidate clears min_match against the FULL
    original read — a narrower sub-query can never inflate the score.

The policy is deliberately conservative. OCR on a book spine is noisy at the
best of times -- vertical text, glare, stylised fonts -- and a lookup that
answers confidently and wrongly is worse than one that shrugs. One bad match
discredits every good one.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# A spine read has to carry some actual signal before it is worth a network call and
# a chance at a false positive. Real examples from one shelf that should never have
# been queried at all: "1 1", "0 1 0", "Jonan". Letters, not characters -- OCR noise
# on a spine is mostly digits and punctuation. The same constant caps the evidence
# scaling in match_score: fewer than this many matched letters is not a match.
MIN_QUERY_LETTERS = 8

# Small closed-class English words. Two uses: they carry almost no weight in
# match_score (every title shares them), and they are dropped from constructed
# retrieval queries (in an OR-union they only add noise).
STOPWORDS = frozenset(
    "a an and the of for in on at to or is it its by with from as s this that".split())

# token_suspicion() at or above this means "likely OCR junk, not a word".
SUSPECT = 2.0

_VOWELS = set("aeiouy")


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", (s or "").lower()).strip()


def _tokens(s: str) -> list[str]:
    """Alphanumeric tokens of a raw (case-preserved) string."""
    return re.findall(r"[0-9A-Za-z]+", s or "")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _greedy_tokens(query_toks: list[str], read_toks: list[str], used: list[bool],
                   content_only_reads: bool) -> tuple[float, float, float, float]:
    """Greedy fuzzy assignment of query tokens onto unused read tokens.

    Each query token (heaviest first) claims its best-matching unused read
    token; per-token similarity below 0.5 is no match at all. Weights:
    stopwords count 1, content tokens their length -- "of" missing from a read
    means nothing, "harvest" missing means a lot. When content_only_reads is
    set, content query tokens may not claim stopword read tokens ("and" in a
    read is not evidence of "Candle" -- a real false positive).

    Mutates `used`. Returns (matched weight, total weight, matched content
    letters, confident content letters). The last counts only tokens matched at
    0.7+ -- a token matched at 0.55 ("winter"~"ineae", a real false positive)
    may shade the coverage but is not EVIDENCE that the book is this one.
    Assignment is by descending match quality, not query-token order: otherwise
    "harbor" claims the read token "hare" at 0.55 before the title token
    "hare" can claim it exactly (a real miss class on series-banner spine reads).
    """
    if not query_toks:
        return 0.0, 0.0, 0.0, 0.0
    weights = [1.0 if t in STOPWORDS else float(len(t)) for t in query_toks]
    pairs = []
    for i, qt in enumerate(query_toks):
        is_content = qt not in STOPWORDS
        for j, rt in enumerate(read_toks):
            if used[j] or (content_only_reads and is_content and rt in STOPWORDS):
                continue
            r = SequenceMatcher(None, qt, rt).ratio()
            if r >= 0.5:
                pairs.append((r, weights[i], i, j))
    got = letters = confident = 0.0
    taken: set[int] = set()
    for r, w, i, j in sorted(pairs, key=lambda p: (-p[0], -p[1])):
        if i in taken or used[j]:
            continue
        taken.add(i)
        used[j] = True
        got += w * r
        if query_toks[i] not in STOPWORDS:
            letters += len(query_toks[i]) * r
            if r >= 0.7:
                confident += len(query_toks[i]) * r
    return got, sum(weights), letters, confident


def match_score(ocr_text: str, title: str, authors=()) -> float:
    """Fuzzy similarity in [0,1] between an OCR read and a candidate book.

    stdlib difflib (no extra dep). Max of two views:

    * Sequence overlap normalized by the LONGER string, counting only matching
      blocks of 2+ characters (scattered single characters are coincidence, not
      evidence), and credited only when the LONGEST block carries
      MIN_QUERY_LETTERS of alphabetic material -- this view exists for reads
      whose tokenization failed (glued or split words), so its evidence must be
      one substantial contiguous run. One shared word plus scraps ("orange "
      + "un" against 'The Orange Uncle', "the " + "ed" + " tower" against
      'The Cursed Tower' -- the real false-positive shape, titles anonymized)
      is what unrelated titles share; a single 8-letter run is not. Stricter than plain ratio(): a
      fragment exactly contained in a long title earns exactly its coverage
      share (the "Jonan" fix -- cover 15% of the title, get 15% of the credit),
      and a tiny title inside a long read earns equally little.

    * Title-token coverage. Spine reads concatenate author and series around
      the title, which dilutes any symmetric measure; asking "is the title
      present in the read?" restores those matches. Three guards keep the
      lenient direction honest, all measured off real false positives:
        - evidence floor: below MIN_QUERY_LETTERS of CONFIDENTLY matched content
          letters (per-token 0.7+; weaker matches shade coverage but prove
          nothing) the view scores 0 outright ("Ash" inside "WASHING TIPS" is
          3 letters; "winter"~"ineae" at 0.55 is none);
        - author letters count toward that evidence when the name is
          substantially present (4+ matched letters) -- a spine usually carries
          the author, so a garbled-but-recognizable author name is real
          signal, while a half-matched 4-letter surname fragment is noise;
        - unexplained-residue discount: the more of the read that neither title
          nor author accounts for, the less the coverage is worth (a title that
          explains all of "THE FINAL HARVEZT" outranks the shorter
          "Final", which leaves "HARVEZT" dangling).
    """
    a, b = _norm(ocr_text), _norm(title)
    if not a or not b:
        return 0.0
    blocks = [bl for bl in SequenceMatcher(None, a, b).get_matching_blocks()
              if bl.size >= 2]
    overlap = 0.0
    if blocks and max(sum(c.isalpha() for c in a[bl.a:bl.a + bl.size])
                      for bl in blocks) >= MIN_QUERY_LETTERS:
        overlap = sum(bl.size for bl in blocks) / max(len(a), len(b))

    read_toks = a.split()
    used = [False] * len(read_toks)
    got, total, letters, confident = _greedy_tokens(b.split(), read_toks, used, True)
    cov = got / total if total else 0.0
    # Best single author, matched over the read tokens the title did not claim.
    author_letters = 0.0
    for name in authors or ():
        n = _norm(name).split()
        _, _, al, _ = _greedy_tokens([t for t in n if t not in STOPWORDS],
                                     read_toks, list(used), True)
        author_letters = max(author_letters, al)
    if author_letters < 4.0:
        author_letters = 0.0

    coverage_view = 0.0
    if confident + author_letters >= MIN_QUERY_LETTERS:
        read_letters = sum(len(t) for t in read_toks if t not in STOPWORDS) or 1
        explained = min(1.0, (letters + author_letters) / read_letters)
        coverage_view = cov * (0.6 + 0.4 * explained)
    return round(max(overlap, coverage_view), 3)


def is_queryable(query: str) -> bool:
    """True if a cleaned OCR read has enough alphabetic signal to be worth looking up."""
    return sum(c.isalpha() for c in query) >= MIN_QUERY_LETTERS


# --------------------------------------------------------------------------- #
# Retrieval planning
# --------------------------------------------------------------------------- #
def token_suspicion(tok: str) -> float:
    """How likely a single OCR token is junk rather than a word.

    GENERAL shape signals only -- no dictionary, no knowledge of any particular
    shelf: digit/letter mixes, vowel-starved runs, mid-word case flapping,
    character stutters. >= SUSPECT means "drop me from constructed queries".
    """
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return 3.0  # bare digits/punct -- useless as a search term
    score = 0.0
    if len(letters) < 3:
        score += 1.0
    if any(c.isdigit() for c in tok):
        score += 2.0
    low = "".join(letters).lower()
    if len(low) >= 3 and sum(c in _VOWELS for c in low) / len(low) < 0.2:
        score += 2.0
    run = best_run = 0
    for c in low:
        run = 0 if c in _VOWELS else run + 1
        best_run = max(best_run, run)
    if best_run >= 4:
        score += 1.5
    # Mid-word case flapping: "DiNmHan". A normal word changes case at most once
    # after its first letter ("The"); count the extras.
    flips = sum(1 for x, y in zip(letters, letters[1:]) if x.islower() != y.islower())
    if flips > 1:
        score += 0.5 * (flips - 1)
    if re.search(r"(.)\1\1", low):  # "Henanaeeeee"
        score += 2.0
    return score


def build_query_ladder(query: str, max_attempts: int = 6) -> list[tuple[str, str]]:
    """Bounded retrieval plan for one cleaned spine read.

    Returns [(kind, query), ...], kinds understood by the fetcher:
      "q"      keyword AND over all fields (Open Library default)
      "q_or"   keyword OR-union, relevance-ranked
      "title"  AND restricted to the title field

    Rationale, each verified against the live Open Library API:
      * q= is AND -- a single garbled token returns zero docs, and most spine
        reads carry at least one. It stays first because it is precise when it
        does hit and is already cached from earlier runs.
      * q="a OR b OR c" is honored Solr syntax; rare correct tokens dominate the
        relevance ranking, so ONE call tolerates any number of junk tokens.
      * title= with clean tokens is forgiving of author-token garbage.
      * two long clean tokens are usually enough to pin a book (spines carry
        title+author+series, so pairs survive when full queries cannot).
    """
    attempts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, q: str) -> None:
        q = " ".join(q.split())
        key = (kind, q.lower())
        if q and key not in seen and len(attempts) < max_attempts:
            seen.add(key)
            attempts.append((kind, q))

    add("q", query)

    content: list[str] = []
    seen_tok: set[str] = set()
    for t in _tokens(query):
        if (len(t) >= 3 and t.lower() not in STOPWORDS
                and any(c.isalpha() for c in t) and t.lower() not in seen_tok):
            seen_tok.add(t.lower())
            content.append(t)

    if len(content) >= 2:
        add("q_or", " OR ".join(content))

    clean = [t for t in content if token_suspicion(t) < SUSPECT]
    if len(clean) >= 2 and sum(len(t) for t in clean) >= MIN_QUERY_LETTERS:
        add("title", " ".join(clean))

    # Pairs of the most promising tokens: cleanest first, then longest. Ties keep
    # reading order, which favors title words (they lead on a spine).
    ranked = sorted(clean, key=lambda t: (token_suspicion(t), -len(t)))[:4]
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            add("q", f"{ranked[i]} {ranked[j]}")
    return attempts


def walk_ladder(read_query: str, search, min_match: float,
                max_attempts: int = 6) -> tuple[dict | None, int]:
    """Run the retrieval ladder until a candidate matches the FULL read.

    `search(kind, q)` -> iterable of candidate dicts, each with a "title" key.
    Every candidate is scored against the original read -- never the sub-query
    that retrieved it -- so a narrower rung cannot inflate the match. Stops at
    the first attempt whose best candidate clears min_match.

    Returns (accepted candidate + score/retrieved_by, attempts made), or
    (None, attempts made) if the whole ladder comes up dry.
    """
    tried = 0
    for kind, q in build_query_ladder(read_query, max_attempts):
        tried += 1
        best, best_sc = None, -1.0
        for cand in search(kind, q) or []:
            sc = match_score(read_query, cand.get("title") or "",
                             cand.get("authors") or ())
            if sc > best_sc:
                best, best_sc = cand, sc
        if best is not None and best_sc >= min_match:
            out = dict(best)
            out["score"] = best_sc
            out["retrieved_by"] = {"kind": kind, "query": q, "attempt": tried}
            return out, tried
    return None, tried
