# SciPubRAG

As someone who previously worked in academic research, I have my hands on a lot of science papers.

Thus, I had an idea: A modular, evaluation-driven RAG (retrieval-augmented generation) pipeline for scientific
publications, running fully locally with [GROBID](https://github.com/kermitt2/grobid) for PDF
parsing and [Ollama](https://ollama.com) for embeddings and generation.

The pipeline can be reused for other corpora by swapping the loader.

**Status (WIP):** Simple RAG ingestion complete (loading, chunking, embedding, indexing). Currently working on the evaluation harness next.

## Architecture

```
            ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐
 PDFs ────▶ │  Loader  │──▶│ Chunker  │──▶│ Embedder │──▶│  Vector  │──▶│   Index     │
 (.md)      │ GROBID / │   │ section /│   │  Ollama /│   │  Store   │   │ + BM25      │
            │ Markdown │   │  fixed   │   │  fake    │   │ NumPy /  │   │ + manifest  │
            └──────────┘   └──────────┘   └──────────┘   │ Chroma   │   └─────────────┘
                 │              │                         └──────────┘
            data/parsed/   data/chunks/                                  index/<name>/
```

| Stage | Interface | Implementations |
|---|---|---|
| Loading | `Loader` | `GrobidLoader` (PDF via GROBID, TEI cached), `MarkdownLoader` |
| Chunking | `Chunker` | `SectionChunker` (structure-aware), `FixedSizeChunker` (baseline) |
| Embedding | `Embedder` | `OllamaEmbedder` (task prefixes handled), `FakeEmbedder` (tests) |
| Vector store | `VectorStore` | `NumpyStore` (exact search, default), `ChromaStore` (optional) |
| Keyword search | | `BM25Index` |

Each stage writes its output to disk, so it can be inspected and later stages re-run on
their own.

## Quickstart

```bash
# 1. Services
docker compose up -d grobid          # GROBID on :8070
ollama pull nomic-embed-text         # Ollama installed natively (recommended on macOS)

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # add ,chroma to use the Chroma backend (Python <= 3.14)

# 3. Ingest: put PDFs in data/pdfs/, then
scipubrag ingest

# 4. Smoke-test retrieval
scipubrag search "What dataset did they use?"
```

## Configuration

Settings live in [`config/default.toml`](config/default.toml). Experiments are just other
config files or command-line overrides, and each gets its own index folder:

```bash
scipubrag --set chunking.target=250 --set chunking.max=350 --set index.name=c250 ingest
scipubrag --set chunking.strategy=fixed --set index.name=fixed512 ingest
scipubrag --set store.backend=chroma --set index.name=chroma ingest
```

Each index's `manifest.json` records the embedder, prefixes, chunking settings and a
fingerprint of the chunks, so an unchanged index is never re-embedded and results are traceable.

## Commands

| Command | Does |
|---|---|
| `scipubrag extract` | PDFs to TEI XML through GROBID (cached in `data/tei/`) |
| `scipubrag parse` | Sources to `data/parsed/<doc_id>.json` |
| `scipubrag chunk` | Parsed docs to `data/chunks/chunks.jsonl` |
| `scipubrag index` | Embed and store in `index/<name>/` |
| `scipubrag ingest` | All of the above |
| `scipubrag search "..."` | Dense and BM25 results side by side |
| `scipubrag info` / `config` | Show index manifest / effective config |

## Using it as a library

```python
from scipubrag.config import load_config
from scipubrag.index import Index

index = Index.load(load_config("config/default.toml").index_path)
for hit in index.dense_search("How was nitrogen measured?", k=5, where={"part": "body"}):
    print(hit.score, hit.chunk.citation())
```

To add a new kind of source, subclass `scipubrag.loaders.Loader` and implement `load(path)`.

## Design decisions

- **Structure-aware chunking:** Chunks follow paragraph and section boundaries from GROBID's TEI
  and carry a `Title > Section > Subsection` header in the embedded text. A fixed-size chunker is
  included as the baseline to compare against.
- **Exact search by default:** At this corpus size (hundreds to thousands of chunks) brute-force
  cosine search in NumPy is about as fast as an ANN index, loses no recall, and has no heavy
  dependencies. Chroma is available behind the same interface.
- **One filter language:** `where` filters (`$eq $ne $in $nin $gt $gte $lt $lte`) work the
  same across backends and BM25.
- **Few dependencies:** The core needs only `numpy` and `rank_bm25`; HTTP clients for GROBID and
  Ollama use the standard library.

## Development

```bash
pytest          # runs offline: uses the fake embedder and a fixture TEI file
ruff check .
```

## Known limitations

- GROBID extracts table contents poorly
- Scanned PDFs need OCR before GROBID can read them.

## License

MIT
