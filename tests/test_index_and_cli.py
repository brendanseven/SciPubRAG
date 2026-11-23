import json

from scipubrag.chunking import SectionChunker
from scipubrag.cli import main
from scipubrag.config import load_config
from scipubrag.embedding import FakeEmbedder
from scipubrag.index import Index


def test_index_build_load_search(tmp_path, docs):
    chunks = SectionChunker().chunk_all(docs)
    Index.build(chunks, FakeEmbedder(), tmp_path / "idx", verbose=False)
    index = Index.load(tmp_path / "idx")  # embedder recreated from the manifest
    assert len(index) == len(chunks)

    dense = index.dense_search("satellite imagery convolutional network", k=3)
    assert dense[0].chunk.doc_id == "vision2023"
    bm25 = index.keyword_search("rainfall emission scenarios", k=3)
    assert bm25[0].chunk.doc_id == "climate2023"
    filtered = index.dense_search("soil nitrogen", k=5, where={"doc_id": "vision2023"})
    assert {h.chunk.doc_id for h in filtered} == {"vision2023"}


def test_is_current(tmp_path, docs):
    chunks = SectionChunker().chunk_all(docs)
    emb = FakeEmbedder()
    Index.build(chunks, emb, tmp_path / "idx", verbose=False)
    assert Index.is_current(tmp_path / "idx", chunks, emb, "numpy")
    assert not Index.is_current(tmp_path / "idx", chunks[:-1], emb, "numpy")
    assert not Index.is_current(tmp_path / "idx", chunks, FakeEmbedder(dim=64), "numpy")


def test_config_overrides(tmp_path):
    cfg = load_config(None, ["chunking.target=250", "index.name=c250", "store.backend=chroma",
                             "parsing.strip_citations=true"])
    assert cfg.chunking.target == 250 and cfg.index.name == "c250"
    assert cfg.store.backend == "chroma" and cfg.parsing.strip_citations is True


def test_cli_end_to_end_markdown(tmp_path, capsys):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "bees.md").write_text("# Bees\n\n## Diet\n\n" + "Bees collect nectar and pollen. " * 40)
    (src / "rocks.md").write_text("# Rocks\n\n## Types\n\n" + "Basalt is a volcanic rock. " * 40)
    sets = [f"paths.input_dir='{src}'", f"paths.parsed_dir='{tmp_path / 'parsed'}'",
            f"paths.chunks_file='{tmp_path / 'chunks.jsonl'}'",
            f"paths.index_dir='{tmp_path / 'index'}'", "loader.type='markdown'",
            "embedding.provider='fake'"]
    args = [x for s in sets for x in ("--set", s)]

    assert main(args + ["ingest"]) == 0
    assert main(args + ["index"]) == 0
    assert "Up to date" in capsys.readouterr().out
    assert main(args + ["search", "volcanic basalt", "-k", "1", "--mode", "bm25"]) == 0
    assert "rocks__types__000" in capsys.readouterr().out
    assert main(args + ["info"]) == 0
    assert json.loads(capsys.readouterr().out)["n_docs"] == 2
