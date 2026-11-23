"""Command-line interface.

    scipubrag ingest                      parse + chunk + index in one go
    scipubrag extract | parse | chunk | index
    scipubrag search "What dataset did they use?" [-k 5] [--mode both] [--doc smith2021]
    scipubrag info                        show the current index manifest
    scipubrag config                      print the effective configuration

Global options (before the command):
    --config config/c250.toml
    --set chunking.target=250 --set index.name=c250
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scipubrag import pipeline
from scipubrag.config import Config, load_config
from scipubrag.index import Index
from scipubrag.types import Chunk, Document

DEFAULT_CONFIG = Path("config/default.toml")


def _doc_summary(doc: Document) -> str:
    n_blocks = sum(len(s.blocks) for s in doc.sections)
    n_words = sum(len(b.text.split()) for s in doc.sections for b in s.blocks)
    warn = [w for w, bad in [("no title", not doc.title), ("no abstract", not doc.abstract),
                             ("little body text", n_words < 1000)] if bad]
    flag = f"  <-- check: {', '.join(warn)}" if warn else ""
    return (f"  {doc.doc_id:<32} sections={len(doc.sections):>3} paragraphs={n_blocks:>4} "
            f"words={n_words:>6} figs/tables={len(doc.figures):>3}{flag}")


def _chunk_summary(chunks: list[Chunk]) -> list[str]:
    by_doc: dict[str, list[int]] = {}
    kinds = Counter(c.kind for c in chunks)
    for c in chunks:
        if c.kind == "body":
            by_doc.setdefault(c.doc_id, []).append(c.n_tokens)
    lines = [f"  {d:<32} body chunks={len(s):>4} tokens min/avg/max="
             f"{min(s)}/{sum(s) // len(s)}/{max(s)}" for d, s in sorted(by_doc.items())]
    lines.append(f"  total: {len(chunks)} chunks ({dict(kinds)})")
    return lines


def cmd_extract(cfg: Config, args) -> int:
    status = pipeline.extract(cfg, force=args.force)
    if not status:
        print("Nothing to extract (loader isn't grobid).")
    for name, s in status.items():
        print(f"  {name:<40} {s}")
    return 1 if any(s.startswith("error") for s in status.values()) else 0


def cmd_parse(cfg: Config, args) -> int:
    docs = pipeline.parse(cfg)
    for d in docs:
        print(_doc_summary(d))
    print(f"Wrote {len(docs)} documents to {cfg.paths.parsed_dir}/")
    return 0


def cmd_chunk(cfg: Config, args) -> int:
    chunks = pipeline.chunk(cfg)
    print("\n".join(_chunk_summary(chunks)))
    print(f"Wrote {cfg.paths.chunks_file}")
    return 0


def cmd_index(cfg: Config, args) -> int:
    index, rebuilt = pipeline.build_index(cfg, force=args.force)
    m = index.manifest
    verb = "Built" if rebuilt else "Up to date:"
    print(f"{verb} index '{cfg.index.name}' at {index.path}: {m['n_chunks']} chunks, "
          f"{m['n_docs']} docs, {m['embedder']['model']} (dim {m['dim']}), "
          f"{m['store_backend']} store")
    if not rebuilt:
        print("  (nothing changed; use --force to rebuild)")
    return 0


def cmd_ingest(cfg: Config, args) -> int:
    print("Parsing...")
    docs = pipeline.parse(cfg)
    for d in docs:
        print(_doc_summary(d))
    print("Chunking...")
    chunks = pipeline.chunk(cfg, docs)
    print("\n".join(_chunk_summary(chunks)))
    print("Indexing...")
    index, rebuilt = pipeline.build_index(cfg, chunks, force=args.force)
    print(f"{'Built' if rebuilt else 'Up to date:'} index '{cfg.index.name}' "
          f"({len(index)} chunks) at {index.path}")
    return 0


def cmd_search(cfg: Config, args) -> int:
    index = Index.load(cfg.index_path)
    where = {"doc_id": args.doc} if args.doc else None
    modes = ["dense", "bm25"] if args.mode == "both" else [args.mode]
    for mode in modes:
        hits = (index.dense_search if mode == "dense" else index.keyword_search)(
            args.question, args.k, where)
        print(f"\n--- {mode} (top {args.k}) ---")
        if not hits:
            print("  (no results)")
        for h in hits:
            preview = h.chunk.text.replace("\n", " ")[:args.preview]
            print(f"{h.rank:>2}. {h.score:7.3f}  {h.chunk.chunk_id}\n"
                  f"    [{h.chunk.citation()}]\n    {preview}...")
    return 0


def cmd_info(cfg: Config, args) -> int:
    path = cfg.index_path / "manifest.json"
    if not path.exists():
        print(f"No index at {cfg.index_path}")
        return 1
    print(path.read_text())
    return 0


def cmd_config(cfg: Config, args) -> int:
    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="scipubrag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help=f"TOML config (default: {DEFAULT_CONFIG} if present)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="SECTION.KEY=VALUE", help="override a config value (repeatable)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="PDFs -> TEI with GROBID")
    p.add_argument("--force", action="store_true", help="re-process cached PDFs")
    p.set_defaults(func=cmd_extract)

    sub.add_parser("parse", help="sources -> parsed JSON").set_defaults(func=cmd_parse)
    sub.add_parser("chunk", help="parsed JSON -> chunks.jsonl").set_defaults(func=cmd_chunk)

    p = sub.add_parser("index", help="chunks -> embeddings + vector store")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ingest", help="parse + chunk + index")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="smoke-test retrieval")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["dense", "bm25", "both"], default="both")
    p.add_argument("--doc", help="only search this doc_id")
    p.add_argument("--preview", type=int, default=160, help="characters of text to show")
    p.set_defaults(func=cmd_search)

    sub.add_parser("info", help="show index manifest").set_defaults(func=cmd_info)
    sub.add_parser("config", help="print effective config").set_defaults(func=cmd_config)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    try:
        cfg = load_config(config_path, args.overrides)
        return args.func(cfg, args)
    except (FileNotFoundError, ConnectionError, ValueError, RuntimeError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
