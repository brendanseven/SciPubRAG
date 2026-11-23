"""The ingestion pipeline as plain functions. Each stage writes its output to disk
so you can inspect it and re-run later stages without repeating earlier ones.

    extract  PDFs -> TEI           (GROBID only; cached)
    parse    sources -> Documents  data/parsed/*.json
    chunk    Documents -> Chunks   data/chunks/chunks.jsonl
    index    Chunks -> Index       index/<name>/
"""

from __future__ import annotations

import json
from pathlib import Path

from scipubrag.chunking import TOKENIZER, get_chunker
from scipubrag.config import Config
from scipubrag.embedding import get_embedder
from scipubrag.index import Index, read_chunks, write_chunks
from scipubrag.loaders import GrobidLoader, get_loader
from scipubrag.types import Chunk, Document


def extract(cfg: Config, force: bool = False) -> dict[str, str]:
    loader = get_loader(cfg)
    if not isinstance(loader, GrobidLoader):
        return {}
    return loader.client.process_dir(Path(cfg.paths.input_dir), Path(cfg.paths.tei_dir),
                                     cfg.grobid.workers, force=force)


def parse(cfg: Config) -> list[Document]:
    loader = get_loader(cfg)
    out_dir = Path(cfg.paths.parsed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = list(loader.load_dir(Path(cfg.paths.input_dir)))
    for doc in docs:
        (out_dir / f"{doc.doc_id}.json").write_text(
            json.dumps(doc.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return docs


def load_documents(cfg: Config) -> list[Document]:
    files = sorted(Path(cfg.paths.parsed_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No parsed documents in {cfg.paths.parsed_dir}. Run `parse`.")
    return [Document.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in files]


def chunk(cfg: Config, docs: list[Document] | None = None) -> list[Chunk]:
    docs = docs if docs is not None else load_documents(cfg)
    chunks = get_chunker(cfg).chunk_all(docs)
    write_chunks(chunks, Path(cfg.paths.chunks_file))
    return chunks


def build_index(cfg: Config, chunks: list[Chunk] | None = None, force: bool = False,
                verbose: bool = True) -> tuple[Index, bool]:
    """Returns (index, rebuilt). Skips embedding when nothing changed."""
    chunks = chunks if chunks is not None else read_chunks(Path(cfg.paths.chunks_file))
    embedder = get_embedder(cfg)
    path = cfg.index_path
    if not force and Index.is_current(path, chunks, embedder, cfg.store.backend):
        return Index.load(path, embedder), False
    extra = {"chunking": vars(cfg.chunking).copy(), "tokenizer": TOKENIZER,
             "loader": cfg.loader.type}
    index = Index.build(chunks, embedder, path, cfg.store.backend,
                        batch_size=max(cfg.embedding.batch_size, 1), verbose=verbose,
                        extra_manifest=extra)
    return index, True


def ingest(cfg: Config, force: bool = False, verbose: bool = True) -> tuple[Index, bool]:
    docs = parse(cfg)
    chunks = chunk(cfg, docs)
    return build_index(cfg, chunks, force=force, verbose=verbose)
