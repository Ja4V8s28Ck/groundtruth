"""Retrieval over the student's own notes, plus quote verification.

Two retrievers live here. The primary one embeds chunks with a local Ollama
embedding model. If that model is not installed we fall back to TF-IDF keyword
scoring so a friend who pulled exactly one model can still run the app.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .ingest import Chunk
from .ollama import OllamaClient, OllamaError

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "were", "what", "when",
    "which", "who", "will", "with", "you", "your",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


@dataclass
class Hit:
    """A retrieved chunk with its similarity score."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text


class NoteStore:
    """Holds the chunks and answers 'what do my notes actually say about this?'."""

    def __init__(self, chunks: list[Chunk], client: OllamaClient) -> None:
        self.chunks = chunks
        self.client = client
        self.mode = "pending"
        self._vectors: np.ndarray | None = None
        self._matrix: dict[int, dict[str, float]] = {}
        self._idf: dict[str, float] = {}

    # ----------------------------------------------------------------- index

    def build(self, progress=None) -> str:
        """Index the chunks. Returns the retrieval mode actually used."""
        total = len(self.chunks) or 1
        for position, chunk in enumerate(self.chunks, start=1):
            if progress:
                progress(position / total, f"Reading section {position} of {total}")
            self._build_lexical(chunk)

        # One batched embed call for the whole document.
        #
        # This used to walk the chunks one at a time and bail out as soon as
        # `_vectors` existed, so only chunk 0 ever received a real vector and
        # every other row stayed zero. `_search_semantic` filters zero scores
        # out, so retrieval silently returned nothing but chunk 0 while still
        # reporting mode "semantic" - grading was citing the first section of
        # the notes and nothing else.
        try:
            wanted = [i for i, chunk in enumerate(self.chunks) if chunk.text.strip()]
            vectors = self.client.embed([self.chunks[i].text for i in wanted])
            if wanted and len(vectors) == len(wanted):
                matrix = np.zeros((len(self.chunks), len(vectors[0])), dtype=np.float32)
                for row, index in enumerate(wanted):
                    matrix[index] = np.asarray(vectors[row], dtype=np.float32)
                self._vectors = matrix
                self.mode = "semantic"
            else:
                self.mode = "keyword"
        except OllamaError:
            # Stay on keyword scoring. The app keeps working with one model.
            self._vectors = None
            self.mode = "keyword"
        return self.mode

    def _build_lexical(self, chunk: Chunk) -> None:
        counts: dict[str, float] = {}
        for token in _tokenize(chunk.text):
            counts[token] = counts.get(token, 0.0) + 1.0
        self._matrix[chunk.id] = counts

    def _finish_lexical(self) -> None:
        total = len(self.chunks) or 1
        tokens = {token for counts in self._matrix.values() for token in counts}
        for token in tokens:
            seen = sum(1 for counts in self._matrix.values() if token in counts)
            self._idf[token] = math.log(1 + total / (1 + seen))

    # ---------------------------------------------------------------- search

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []

        if self._vectors is not None:
            return self._search_semantic(query, k)
        return self._search_lexical(query, k)

    def _search_semantic(self, query: str, k: int) -> list[Hit]:
        vectors = self.client.embed([query])
        if not vectors:
            return []
        matrix = np.asarray(self._vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        unit = matrix / norms[:, None]

        probe = np.asarray(vectors[0], dtype=np.float32)
        probe_norm = float(np.linalg.norm(probe)) or 1.0
        scores = (unit @ probe) / probe_norm

        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def _search_lexical(self, query: str, k: int) -> list[Hit]:
        if not self._idf:
            self._finish_lexical()
        query_counts: dict[str, float] = {}
        for token in _tokenize(query):
            query_counts[token] = query_counts.get(token, 0.0) + 1.0
        if not query_counts:
            return []

        query_vector = {
            token: count * self._idf.get(token, 0.0) for token, count in query_counts.items()
        }
        query_norm = math.sqrt(sum(v * v for v in query_vector.values())) or 1.0

        hits: list[Hit] = []
        for chunk in self.chunks:
            chunk_vector = {
                token: count * self._idf.get(token, 0.0)
                for token, count in self._matrix[chunk.id].items()
            }
            norm = math.sqrt(sum(v * v for v in chunk_vector.values()))
            if norm == 0:
                continue
            dot = sum(value * chunk_vector.get(token, 0.0) for token, value in query_vector.items())
            hits.append(Hit(chunk, dot / (query_norm * norm)))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in hits[:k] if hit.score > 0]


def normalize(text: str) -> str:
    """Collapse whitespace and case so quotes can be matched against real text."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase + collapse whitespace, keeping a map back to original indices."""
    out: list[str] = []
    index_map: list[int] = []
    previous_was_space = True
    for position, character in enumerate(text):
        if character.isspace():
            if not previous_was_space:
                out.append(" ")
                index_map.append(position)
                previous_was_space = True
        else:
            out.append(character.lower())
            index_map.append(position)
            previous_was_space = False
    return "".join(out), index_map


def term_coverage(query: str, texts: Iterable[str]) -> float:
    """How much of the query's distinctive vocabulary appears in the given texts.

    Used to keep the refusal honest. "Your notes do not cover this" is a testable
    claim: if the question's own key terms are sitting in the document, a model
    that says otherwise is wrong, and we say so in code rather than trusting it.
    """
    haystack = " ".join(texts).lower()
    terms = set(_tokenize(query))
    if not terms:
        return 0.0
    return sum(1 for term in terms if term in haystack) / len(terms)


def spoils_facts(text: str, facts: Iterable[str], *, threshold: float = 0.6) -> bool:
    """True when `text` repeats the very facts it is meant to be pointing past.

    A nudge that gives the answer away is worse than no nudge at all: the student
    walks away believing they recalled something. We check that lexically rather
    than trusting a 4B model to keep its mouth shut.
    """
    said = set(_tokenize(text))
    if not said:
        return False
    for fact in facts:
        terms = set(_tokenize(fact))
        if terms and len(terms & said) / len(terms) >= threshold:
            return True
    return False


def verify_quote(chunk: Chunk, quote: str, *, min_coverage: float = 0.6) -> str | None:
    """Return the exact text from `chunk` that `quote` refers to, or None.

    Models paraphrase and sometimes pad a quote with a whole paragraph. This finds
    the longest contiguous run of the quote's words that genuinely appears in the
    student's notes, and only accepts it if enough of the quote is really there.

    That strictness is the whole trust model. A loose matcher would "verify" any
    quote against any chunk, the evidence check would always pass, and the app
    would happily display a confident verdict that nothing supports.
    """
    if not quote.strip():
        return None

    haystack, index_map = _normalise_with_map(chunk.text)
    needle_words = normalize(quote).split()
    if not needle_words:
        return None

    def _slice(start: int, length: int) -> str:
        if start >= len(index_map):
            return ""
        end = min(start + length, len(index_map)) - 1
        return chunk.text[index_map[start] : index_map[end] + 1].strip()

    joined = " ".join(needle_words)
    position = haystack.find(joined)
    if position != -1:
        return _slice(position, len(joined))

    # No exact run: take the longest window of the quote that does appear.
    best_size = 0
    best_position = -1
    for size in range(len(needle_words), 2, -1):
        for start in range(0, len(needle_words) - size + 1):
            window = " ".join(needle_words[start : start + size])
            position = haystack.find(window)
            if position != -1:
                best_size, best_position = size, position
                break
        if best_position != -1:
            break

    if best_position == -1 or best_size / len(needle_words) < min_coverage:
        return None

    return _slice(best_position, len(" ".join(needle_words[0:best_size])))
