"""Unit tests for the terminal scene-query scoring (pure, stdlib only)."""


def _objs():
    return [
        {"id": "o0", "label": "The Pragmatic Programmer", "category": "book",
         "keywords": ["book", "pragmatic"], "position": [0, 0, 0]},
        {"id": "o1", "label": "potted plant", "category": "plant",
         "keywords": ["plant", "potted"], "position": [1, 0, 0]},
        {"id": "o2", "label": "desk lamp", "category": "lamp",
         "keywords": ["lamp", "desk"], "position": [0, 1, 0]},
    ]


def test_query_exact_category(load_module):
    q = load_module("understanding/query.py", "query")
    hits = q.query(_objs(), "book")
    assert [h["id"] for h in hits] == ["o0"]
    assert hits[0]["score"] >= 3


def test_query_case_insensitive_and_keyword(load_module):
    q = load_module("understanding/query.py", "query")
    assert [h["id"] for h in q.query(_objs(), "PLANT")] == ["o1"]
    assert [h["id"] for h in q.query(_objs(), "desk")] == ["o2"]   # keyword hit


def test_query_no_match_is_empty(load_module):
    q = load_module("understanding/query.py", "query")
    assert q.query(_objs(), "helicopter") == []


def test_query_top_k(load_module):
    q = load_module("understanding/query.py", "query")
    objs = _objs() + [{"id": "o3", "label": "book two", "category": "book",
                       "keywords": ["book"], "position": [2, 0, 0]}]
    hits = q.query(objs, "book", top=1)
    assert len(hits) == 1 and hits[0]["category"] == "book"
