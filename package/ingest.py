"""Turn a student's notes into chunks that remember where they came from.

The citation feature depends entirely on this module: every chunk keeps the
line range it occupied in the original document, so a verdict can point at an
exact line instead of vaguely "referring to the notes".
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(
    r"""^\s*(?:
          \#{1,6}\s+\S                      # markdown heading
        | \d+(?:\.\d+)*[.)]?\s+[A-Z]        # 1. Title  /  2.3 Title
        | [A-Z][\w '&/(),.-]{2,60}:$       # Title:
        | (?:chapter|unit|part|topic|section|lecture)\s+\S+
    )\s*$""",
    re.VERBOSE | re.IGNORECASE,
)

_PDF_SUFFIXES = {".pdf"}
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ".rst"}


@dataclass
class Chunk:
    """A slice of the document plus its coordinates in the original text."""

    id: int
    text: str
    line_start: int  # 1-based, inclusive
    line_end: int  # 1-based, inclusive
    heading: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def cite(self) -> str:
        return f"lines {self.line_start}-{self.line_end}"


def clean_text(text: str) -> str:
    """Normalise pasted or extracted text without destroying its line structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader  # imported lazily so .txt users need no PDF stack

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        body = (page.extract_text() or "").strip()
        if body:
            pages.append(f"## Page {number}\n{body}")
    if not pages:
        raise ValueError("No selectable text found in that PDF - it is probably a scan of images.")
    return "\n\n".join(pages)


def load_document(*, text: str = "", filename: str = "", data: bytes | None = None) -> str:
    """Return clean plain text from pasted text, an uploaded .txt/.md, or a PDF."""
    if data:
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix in _PDF_SUFFIXES:
            return clean_text(_pdf_to_text(data))
        if suffix in _TEXT_SUFFIXES or not suffix:
            return clean_text(data.decode("utf-8", errors="replace"))
        raise ValueError(f"Unsupported file type '{suffix}'. Use a PDF, TXT or MD file.")

    if not text.strip():
        raise ValueError("Nothing to read - paste some notes or upload a file.")
    return clean_text(text)


def _blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    """Group lines into blank-line-separated blocks with their line numbers."""
    blocks: list[tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip():
            start = index
            while index < len(lines) and lines[index].strip():
                index += 1
            blocks.append((start, index - 1, "\n".join(lines[start:index])))
        else:
            index += 1
    return blocks


def _is_heading(block: str) -> bool:
    first = block.splitlines()[0] if block.splitlines() else ""
    return len(block) < 80 and bool(_HEADING_RE.match(first))


def chunk_document(
    text: str,
    *,
    target_chars: int = 1100,
    min_chars: int = 220,
) -> list[Chunk]:
    """Split notes into topical chunks, never cutting a heading off its body."""
    lines = text.split("\n")
    chunks: list[Chunk] = []
    buffer: list[str] = []
    start = end = 0
    heading = ""

    def flush() -> None:
        nonlocal buffer, start, end
        if buffer:
            chunks.append(
                Chunk(id=len(chunks), text="\n\n".join(buffer), line_start=start + 1, line_end=end + 1, heading=heading)
            )
            buffer = []

    for block_start, block_end, block in _blocks(lines):
        if _is_heading(block):
            heading = block.splitlines()[0].strip()
            if buffer and len("\n\n".join(buffer)) > target_chars * 0.5:
                flush()

        candidate = len("\n\n".join(buffer + [block]))
        if buffer and candidate > target_chars:
            flush()

        if not buffer:
            start, end = block_start, block_end
        buffer.append(block)
        end = block_end

    flush()

    # Fold runt chunks backwards so we never retrieve a two-line fragment.
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and len(chunk.text) < min_chars:
            previous = merged[-1]
            previous.text = f"{previous.text}\n\n{chunk.text}"
            previous.line_end = chunk.line_end
            if not previous.heading:
                previous.heading = chunk.heading
        else:
            merged.append(chunk)

    for index, chunk in enumerate(merged):
        chunk.id = index
    return merged


def document_stats(text: str, chunks: list[Chunk]) -> dict[str, int]:
    return {
        "characters": len(text),
        "words": len(text.split()),
        "chunks": len(chunks),
    }
