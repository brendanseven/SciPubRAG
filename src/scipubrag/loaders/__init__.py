from scipubrag.loaders.base import Loader, make_doc_id
from scipubrag.loaders.grobid import GrobidClient, GrobidLoader
from scipubrag.loaders.markdown import MarkdownLoader
from scipubrag.loaders.tei import parse_tei

__all__ = ["GrobidClient", "GrobidLoader", "Loader", "MarkdownLoader", "get_loader",
           "make_doc_id", "parse_tei"]


def get_loader(cfg) -> Loader:
    """Build the loader named in cfg.loader.type."""
    kind = cfg.loader.type
    if kind == "grobid":
        client = GrobidClient(cfg.grobid.url, cfg.grobid.timeout,
                              cfg.grobid.consolidate_header, cfg.grobid.coordinates)
        return GrobidLoader(cfg.paths.tei_dir, client, cfg.parsing.strip_citations,
                            cfg.parsing.include_acknowledgements, cfg.grobid.workers)
    if kind == "markdown":
        return MarkdownLoader()
    raise ValueError(f"Unknown loader type {kind!r} (expected 'grobid' or 'markdown')")
