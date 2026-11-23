"""PDF loader backed by a GROBID server.

PDFs are sent to GROBID once; the TEI XML is cached in tei_dir so later runs
(and re-parsing with different options) don't hit the server again.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipubrag.loaders.base import Loader, make_doc_id
from scipubrag.loaders.tei import parse_tei
from scipubrag.types import Document


def _multipart(fields: list[tuple[str, str]], file_field: str,
               file_path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    lines: list[bytes] = []
    for name, value in fields:
        lines += [f"--{boundary}".encode(),
                  f'Content-Disposition: form-data; name="{name}"'.encode(), b"", value.encode()]
    lines += [f"--{boundary}".encode(),
              f'Content-Disposition: form-data; name="{file_field}"; '
              f'filename="{file_path.name}"'.encode(),
              b"Content-Type: application/pdf", b"", file_path.read_bytes(),
              f"--{boundary}--".encode(), b""]
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


class GrobidClient:
    def __init__(self, url: str = "http://localhost:8070", timeout: int = 180,
                 consolidate_header: bool = False, coordinates: bool = True):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.consolidate_header = consolidate_header
        self.coordinates = coordinates

    def is_alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/api/isalive", timeout=5) as resp:
                return resp.read().decode().strip().lower() == "true"
        except (urllib.error.URLError, OSError):
            return False

    def process_pdf(self, pdf: Path, retries: int = 5) -> str:
        """Return TEI XML for one PDF. Retries when GROBID answers 503 (busy)."""
        fields = [("consolidateHeader", "1" if self.consolidate_header else "0"),
                  ("consolidateCitations", "0")]
        if self.coordinates:
            fields += [("teiCoordinates", t) for t in ("p", "head", "figure", "formula")]
        body, content_type = _multipart(fields, "input", Path(pdf))

        for attempt in range(retries):
            req = urllib.request.Request(
                f"{self.url}/api/processFulltextDocument", data=body,
                headers={"Content-Type": content_type, "Accept": "application/xml"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                if e.code == 503 and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                detail = e.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"GROBID failed on {Path(pdf).name}: {e.code} {detail}") from e
        raise RuntimeError(f"GROBID stayed busy for {Path(pdf).name}")

    def process_dir(self, pdf_dir: Path, tei_dir: Path, workers: int = 4,
                    force: bool = False) -> dict[str, str]:
        """Convert every PDF in pdf_dir, writing <doc_id>.tei.xml. Returns {file: status}."""
        pdf_dir, tei_dir = Path(pdf_dir), Path(tei_dir)
        tei_dir.mkdir(parents=True, exist_ok=True)
        if not self.is_alive():
            raise ConnectionError(
                f"GROBID isn't reachable at {self.url}. "
                "Start it with `docker compose up -d grobid`."
            )
        pdfs = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

        status: dict[str, str] = {}
        todo = []
        for pdf in pdfs:
            out = tei_dir / f"{make_doc_id(pdf)}.tei.xml"
            if out.exists() and not force:
                status[pdf.name] = "cached"
            else:
                todo.append((pdf, out))

        def work(pdf: Path, out: Path) -> str:
            out.write_text(self.process_pdf(pdf), encoding="utf-8")
            return "ok"

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, pdf, out): pdf for pdf, out in todo}
            for fut in as_completed(futures):
                pdf = futures[fut]
                try:
                    status[pdf.name] = fut.result()
                except Exception as e:  # keep going; report at the end
                    status[pdf.name] = f"error: {e}"
        return dict(sorted(status.items()))


class GrobidLoader(Loader):
    """Loads PDFs through GROBID, using cached TEI files when present."""

    extensions = (".pdf",)

    def __init__(self, tei_dir: str | Path = "data/tei", client: GrobidClient | None = None,
                 strip_citations: bool = False, include_acknowledgements: bool = False,
                 workers: int = 4):
        self.tei_dir = Path(tei_dir)
        self.client = client or GrobidClient()
        self.strip_citations = strip_citations
        self.include_acknowledgements = include_acknowledgements
        self.workers = workers

    def tei_path(self, pdf: Path) -> Path:
        return self.tei_dir / f"{make_doc_id(pdf)}.tei.xml"

    def load(self, path: Path) -> Document:
        path = Path(path)
        tei = self.tei_path(path)
        if not tei.exists():
            self.tei_dir.mkdir(parents=True, exist_ok=True)
            tei.write_text(self.client.process_pdf(path), encoding="utf-8")
        doc = parse_tei(tei, self.strip_citations, self.include_acknowledgements,
                        doc_id=make_doc_id(path))
        doc.source = str(path)
        return doc

    def load_dir(self, directory: Path) -> Iterator[Document]:
        pdfs = self.files(directory)
        if any(not self.tei_path(p).exists() for p in pdfs):
            status = self.client.process_dir(Path(directory), self.tei_dir, self.workers)
            errors = {k: v for k, v in status.items() if v.startswith("error")}
            if errors:
                print("GROBID errors:\n" + json.dumps(errors, indent=2))
        for pdf in pdfs:
            if self.tei_path(pdf).exists():
                yield self.load(pdf)
