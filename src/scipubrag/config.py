"""Configuration: dataclasses with defaults, loaded from TOML (stdlib tomllib).

    cfg = load_config("config/default.toml", overrides=["chunking.target=250"])
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    input_dir: str = "data/pdfs"
    tei_dir: str = "data/tei"
    parsed_dir: str = "data/parsed"
    chunks_file: str = "data/chunks/chunks.jsonl"
    index_dir: str = "index"


@dataclass
class LoaderConfig:
    type: str = "grobid"


@dataclass
class GrobidConfig:
    url: str = "http://localhost:8070"
    workers: int = 4
    timeout: int = 180
    consolidate_header: bool = False
    coordinates: bool = True


@dataclass
class ParsingConfig:
    strip_citations: bool = False
    include_acknowledgements: bool = False


@dataclass
class ChunkingConfig:
    strategy: str = "section"
    target: int = 400
    max: int = 512
    min: int = 80
    overlap: int = 1
    include_captions: bool = True
    include_back: bool = True


@dataclass
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    host: str = "http://localhost:11434"
    num_ctx: int = 2048
    batch_size: int = 32


@dataclass
class StoreConfig:
    backend: str = "numpy"


@dataclass
class IndexConfig:
    name: str = "default"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    grobid: GrobidConfig = field(default_factory=GrobidConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    @property
    def index_path(self) -> Path:
        return Path(self.paths.index_dir) / self.index.name

    def to_dict(self) -> dict:
        return asdict(self)


def _apply(cfg: Config, data: dict[str, Any]) -> None:
    for section_name, values in data.items():
        if not hasattr(cfg, section_name):
            raise ValueError(f"Unknown config section [{section_name}]")
        section = getattr(cfg, section_name)
        valid = {f.name for f in fields(section)}
        for key, value in values.items():
            if key not in valid:
                raise ValueError(f"Unknown config key {section_name}.{key}")
            setattr(section, key, value)


def _parse_override(item: str) -> dict[str, Any]:
    """'chunking.target=250' -> {'chunking': {'target': 250}}"""
    if "=" not in item or "." not in item.split("=", 1)[0]:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")
    dotted, raw = item.split("=", 1)
    section, key = dotted.strip().split(".", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]  # numbers, booleans, quoted strings
    except tomllib.TOMLDecodeError:
        value = raw  # bare string
    return {section: {key: value}}


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("rb") as fh:
            _apply(cfg, tomllib.load(fh))
    for item in overrides or []:
        _apply(cfg, _parse_override(item))
    return cfg
