import importlib.util

import pytest

from scipubrag.stores import get_store_class, matches

BACKENDS = ["numpy"]
if importlib.util.find_spec("chromadb") is not None:
    BACKENDS.append("chroma")


def _data():
    ids = ["a", "b", "c", "d"]
    vecs = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]]
    metas = [{"doc_id": "x", "part": "body", "page_start": 1},
             {"doc_id": "y", "part": "body", "page_start": 4},
             {"doc_id": "x", "part": "back", "page_start": 9},
             {"doc_id": "y", "part": "body", "page_start": 2}]
    return ids, vecs, metas


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_order_and_scores(tmp_path, backend):
    store = get_store_class(backend).create(tmp_path / "s")
    ids, vecs, metas = _data()
    store.add(ids, vecs, metas)
    hits = store.search([1, 0, 0], k=2)
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert len(store) == 4


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("where,expected", [
    ({"doc_id": "y"}, {"b", "d"}),
    ({"part": {"$ne": "back"}, "doc_id": "x"}, {"a"}),
    ({"doc_id": {"$in": ["x"]}}, {"a", "c"}),
    ({"page_start": {"$gte": 4}}, {"b", "c"}),
])
def test_filters(tmp_path, backend, where, expected):
    store = get_store_class(backend).create(tmp_path / "s")
    store.add(*_data())
    assert {h.id for h in store.search([1, 1, 1], k=10, where=where)} == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_roundtrip(tmp_path, backend):
    cls = get_store_class(backend)
    store = cls.create(tmp_path / "s")
    store.add(*_data())
    store.save()
    del store
    loaded = cls.load(tmp_path / "s")
    assert len(loaded) == 4
    assert loaded.search([0, 0, 1], k=1)[0].id == "d"


def test_numpy_rejects_duplicates_and_bad_dims(tmp_path):
    store = get_store_class("numpy").create(tmp_path / "s")
    store.add(*_data())
    with pytest.raises(ValueError):
        store.add(["a"], [[1, 0, 0]], [{}])
    with pytest.raises(ValueError):
        store.add(["z"], [[1, 0]], [{}])


def test_matches_unknown_operator():
    with pytest.raises(ValueError):
        matches({"a": 1}, {"a": {"$regex": "x"}})
