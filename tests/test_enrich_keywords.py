"""Tests for understanding/enrich_keywords.py -- category-synonym expansion of
scene object keywords, so a lay query ("toy", "photo") finds the right object.
Pure logic, no torch/cv2 -- light CI."""
import pytest


@pytest.fixture
def enr(load_module):
    return load_module("understanding/enrich_keywords.py", "enrich_keywords")


def _scene(objects):
    return {"version": 1, "objects": objects}


def test_adds_category_synonyms(enr):
    s = _scene([{"id": "a", "category": "figurine", "keywords": ["figurine"]}])
    n = enr.enrich(s)
    assert n == 1
    kw = s["objects"][0]["keywords"]
    # a person typing "toy" should now hit a figurine
    assert "toy" in kw and "figure" in kw and "figurine" in kw


def test_dedups_and_lowercases_without_dropping_existing(enr):
    # existing keywords (incl. a real book title token) survive; synonyms append; no dups
    s = _scene([{"id": "b", "category": "book", "keywords": ["Book", "dune", "herbert"]}])
    enr.enrich(s)
    kw = s["objects"][0]["keywords"]
    assert kw.count("book") == 1              # "Book" lowercased, not duplicated by synonym
    assert "dune" in kw and "herbert" in kw   # title tokens preserved
    assert "novel" in kw                      # synonym added


def test_unknown_category_is_untouched(enr):
    s = _scene([{"id": "c", "category": "spaceship", "keywords": ["spaceship"]}])
    n = enr.enrich(s)
    assert n == 0
    assert s["objects"][0]["keywords"] == ["spaceship"]


def test_empty_keywords_ok(enr):
    s = _scene([{"id": "d", "category": "owl", "keywords": []}])
    enr.enrich(s)
    assert "owl" in s["objects"][0]["keywords"] and "bird" in s["objects"][0]["keywords"]


def test_every_synonym_list_covers_its_category_tokens(enr):
    # each word of a category name must be searchable via its synonyms, so a
    # multi-word category ("lego model") is still found by typing "lego".
    # (guards a typo'd map entry without over-constraining phrase categories.)
    for cat, syns in enr.SYNONYMS.items():
        for tok in cat.split():
            assert tok in syns, f"{cat!r}: token {tok!r} missing from its synonym list"
