"""The AI core: question generation, grading, and the honest refusal.

This is where the model does work that no amount of string matching can do -
deciding what is worth asking about a page of notes, and judging whether a
student's free-text answer actually matches what their own notes say.

Two invariants hold everywhere in here:

1. The notes are the only permitted source of truth.
2. Every citation is verified against the real text before it reaches the UI.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .ingest import Chunk
from .ollama import OllamaError
from .store import Hit, NoteStore, spoils_facts, term_coverage, verify_quote

# A refusal is only believable if the question's key terms are absent from the
# notes. Above this fraction, "not in your notes" is provably a misjudgement.
COVERAGE_TO_COVER = 0.6

CORRECT = "correct"
INCOMPLETE = "incomplete"
NOT_IN_NOTES = "not_in_notes"

VERDICT_LABELS = {
    CORRECT: "Correct",
    INCOMPLETE: "Not quite",
    NOT_IN_NOTES: "Not in your notes",
}

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "topic": {"type": "string"},
                    "kind": {"type": "string", "enum": ["recall", "explain", "apply"]},
                },
                "required": ["question", "topic", "kind"],
            },
        }
    },
    "required": ["questions"],
}

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [CORRECT, INCOMPLETE, NOT_IN_NOTES]},
        "score": {"type": "integer"},
        "topic": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["chunk_id", "quote"],
            },
        },
        "missing": {"type": "array", "items": {"type": "string"}},
        "nudge": {"type": "string"},
    },
    "required": ["verdict", "score", "topic", "evidence", "nudge"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "weak_topics": {"type": "array", "items": {"type": "string"}},
        "next_step": {"type": "string"},
        "encouragement": {"type": "string"},
    },
    "required": ["weak_topics", "next_step", "encouragement"],
}

_QUESTION_PROMPT = """You are building a revision quiz from ONE student's own notes.

Rules:
- Every question must be answerable using ONLY the passage below. Nothing else exists.
- Never ask about a concept the passage does not mention.
- Do not reveal the answer inside the question.
- Vary the kinds: "recall" (state a specific fact), "explain" (describe a mechanism in your own words), "apply" (a short scenario the student has to reason through using the passage).
- Each question under 30 words. Plain sentences, no markdown, no numbering, no options.
- "topic" is a 2 to 4 word label for the part of the notes this question covers.

Passage (lines {start}-{end}{heading}):
\"\"\"
{body}
\"\"\"
"""

_GRADE_PROMPT = """You are grading a student's answer against the student's OWN notes. The passages below are the ONLY permitted source of truth.

Work in this order:

Step 1 - Read the passages. Decide whether THEY contain the answer to this exact question.
Step 2 - If the passages do NOT contain the answer: verdict "not_in_notes", score 0, evidence [], missing [].
         This is the ONLY reason to ever use "not_in_notes".
Step 3 - If the passages DO contain the answer, judge the student's answer against them:
         - it conveys the answer                      -> "correct",    score 2
         - it is partly right, or too vague to be right by accident -> "incomplete", score 1
         - it contradicts them, or is simply wrong    -> "incomplete", score 0

CRITICAL: a wrong, lazy or vague student answer is NEVER "not_in_notes". Only the passages being
unable to answer the question makes it "not_in_notes". Grading a bad answer against notes that do
contain the answer is the normal case, not an exception.

evidence: 1 or 2 quotes of AT MOST 20 WORDS, copied verbatim from the passages, that settle the answer. Short quotes only - do not paste a whole paragraph. If you cannot copy a short verbatim quote, that is a strong sign the verdict should not be "correct".

missing: up to 3 short phrases naming what a complete answer needed.

nudge: for "incomplete" or "not_in_notes", ONE short sentence naming the section to re-read. If anything appears in your missing list, the nudge must NOT contain it - point at the topic, never hand over the fact. For "correct", use an empty string.

Worked example 1
Passages: [chunk 2] "The sodium-potassium pump spends one ATP to export 3 sodium ions and import 2 potassium ions."
Question: How does the pump move ions, and in what ratio?
Answer: "It uses ATP to push sodium out and bring potassium in, three out for two in."
Output: verdict "correct", score 2, evidence quoting "spends one ATP to export 3 sodium ions and import 2 potassium ions", missing [], nudge "".

Worked example 2
Same passages.
Question: How does the pump move ions, and in what ratio?
Answer: "It swaps sodium and potassium using glucose, two sodium in for three potassium out."
Output: verdict "incomplete", score 0, evidence quoting "spends one ATP to export 3 sodium ions and import 2 potassium ions", missing ["ATP, not glucose", "sodium out and potassium in", "3 out and 2 in"], nudge "Your notes are explicit about the pump - find that line and check what it uses, and which way each ion goes."
Note: the passages DID answer this, so the verdict is "incomplete", not "not_in_notes".

Worked example 3
Same passages.
Question: How does the pump move ions, and in what ratio?
Answer: "Something to do with ATP and proteins, I think?"
Output: verdict "incomplete", score 1, evidence quoting "spends one ATP to export 3 sodium ions and import 2 potassium ions", missing ["the exact ratio"], nudge "You are in the right area - your notes state the ratio outright, go and read it."
Note: a vague answer is still graded against passages that answer the question. It is never "not_in_notes".

Worked example 4
Passages: [chunk 7] "Glycolysis happens in the cytosol, not the mitochondrion. One glucose becomes two pyruvate, netting 2 ATP and 2 NADH."
Question: What are the four phases of mitosis?
Answer: "Prophase condenses the chromosomes, metaphase aligns them, anaphase separates them and telophase reforms the nuclei."
Output: verdict "not_in_notes", score 0, evidence [], missing [], nudge "Your notes do not cover mitosis - that topic is missing from this material."
Note: the passages are about respiration and say nothing about mitosis, so this is the one case that is "not_in_notes".

Question: {question}
Student's answer: {answer}

Passages:
{passages}
"""


@dataclass
class Question:
    """One quiz question. Deliberately carries no answer."""

    id: int
    text: str
    topic: str
    kind: str
    chunk_ids: list[int] = field(default_factory=list)

    @property
    def kind_label(self) -> str:
        return {"recall": "Recall", "explain": "Explain", "apply": "Apply"}.get(self.kind, "Question")


@dataclass
class Verdict:
    """A graded answer, with every citation checked against the real notes."""

    question_id: int
    verdict: str
    score: int
    topic: str
    nudge: str
    evidence: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return VERDICT_LABELS.get(self.verdict, self.verdict)


def _passages(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        chunk = hit.chunk
        heading = f" | {chunk.heading}" if chunk.heading else ""
        blocks.append(f"[chunk {chunk.id} | lines {chunk.line_start}-{chunk.line_end}{heading}]\n{chunk.text}")
    return "\n\n".join(blocks)


def _truncate(text: str, limit: int = 220, words: int | None = None) -> str:
    text = " ".join(str(text).split())
    if words is not None and len(text.split()) > words:
        text = " ".join(text.split()[:words])
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class Tutor:
    """Generates questions from a NoteStore and grades answers against it."""

    def __init__(self, store: NoteStore, client) -> None:
        self.store = store
        self.client = client

    # ------------------------------------------------------------- generation

    def _questions_for(self, chunk: Chunk) -> list[dict]:
        """Ask the model what is worth asking about one passage.

        Runs in a worker thread. It touches nothing but its own argument and the
        client, so several of these can be in flight at once.
        """
        raw = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": "You write revision questions that can only be answered from a given passage. You reply with JSON only.",
                },
                {
                    "role": "user",
                    "content": _QUESTION_PROMPT.format(
                        start=chunk.line_start,
                        end=chunk.line_end,
                        heading=f" | {chunk.heading}" if chunk.heading else "",
                        body=chunk.text,
                    ),
                },
            ],
            schema=QUESTION_SCHEMA,
            options={"temperature": 0.7},
        )
        return list((raw or {}).get("questions", []))

    def build_session(self, count: int = 8, per_chunk: int = 2, progress=None) -> list[Question]:
        """Spread `count` questions across the whole document.

        Chunks are sampled evenly rather than front-loaded, so the session covers
        the entire set of notes instead of only the first page.
        """
        chunks = self.store.chunks
        if not chunks:
            return []

        needed = max(1, -(-count // per_chunk))  # ceil
        if len(chunks) <= needed:
            picks = list(range(len(chunks)))
        else:
            step = len(chunks) / needed
            picks = sorted({min(len(chunks) - 1, int(i * step)) for i in range(needed)})

        batches: list[list[dict]] = [[] for _ in picks]
        with ThreadPoolExecutor(max_workers=min(len(picks), 4)) as pool:
            futures = {
                pool.submit(self._questions_for, chunks[index]): slot
                for slot, index in enumerate(picks)
            }
            # Progress is reported from here rather than from the workers, so the
            # st.status update always happens on the main thread.
            for done, future in enumerate(as_completed(futures), start=1):
                slot = futures[future]
                try:
                    batches[slot] = future.result()
                except OllamaError:
                    # One unreachable passage should not cost the whole session.
                    batches[slot] = []
                if progress:
                    progress(
                        done / len(picks),
                        f"Wrote questions from {done} of {len(picks)} sections",
                    )

        questions: list[Question] = []
        seen: set[str] = set()

        for index, batch in zip(picks, batches):
            chunk = chunks[index]
            for item in batch[:per_chunk]:
                text = _truncate(item.get("question", ""), 260)
                if len(text) < 12:
                    continue
                fingerprint = text.lower()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                questions.append(
                    Question(
                        id=len(questions),
                        text=text,
                        topic=_truncate(item.get("topic", "") or (chunk.heading or "General"), 48),
                        kind=item.get("kind", "recall"),
                        chunk_ids=[chunk.id],
                    )
                )

        return questions[:count]

    # ---------------------------------------------------------------- grading

    def grade_answer(self, question: Question, answer: str, k: int = 5) -> Verdict:
        answer = answer.strip()
        if not answer:
            raise ValueError("Write an answer first - in your own words is fine.")

        hits = self.store.search(f"{question.text}\n{answer}", k=k)
        if not hits:
            return Verdict(
                question_id=question.id,
                verdict=NOT_IN_NOTES,
                score=0,
                topic=question.topic,
                nudge="I could not find anything close to this in your notes. Check the source passage yourself.",
            )

        raw = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": "You grade a student's answer strictly against their own notes. You reply with JSON only.",
                },
                {
                    "role": "user",
                    "content": _GRADE_PROMPT.format(
                        question=question.text,
                        answer=answer,
                        passages=_passages(hits),
                    ),
                },
            ],
            schema=GRADE_SCHEMA,
            options={"temperature": 0.1},
        )
        return self._clean_verdict(question, raw or {}, hits)

    def _clean_verdict(self, question: Question, raw: dict, hits: list[Hit]) -> Verdict:
        """Verify every citation and refuse to show an unsupported 'correct'.

        If the model claims the student was right but cannot point at a real line
        in the notes, the verdict is downgraded rather than shown. The promise of
        this tool is that a citation is always something you can read yourself.
        """
        verdict = raw.get("verdict", INCOMPLETE)
        if verdict not in VERDICT_LABELS:
            verdict = INCOMPLETE

        score = raw.get("score", 0)
        try:
            score = max(0, min(2, int(score)))
        except (TypeError, ValueError):
            score = 0

        available = {hit.chunk.id: hit.chunk for hit in hits}
        evidence: list[dict] = []
        for item in raw.get("evidence", []) or []:
            try:
                chunk_id = int(item.get("chunk_id"))
            except (TypeError, ValueError):
                continue
            chunk = available.get(chunk_id)
            if chunk is None:
                continue
            # Cap what we look for: a 20-word quote is plenty, and an unbounded
            # one is almost always the model pasting a paragraph.
            quote = _truncate(str(item.get("quote", "")), 160, words=24)
            verified = verify_quote(chunk, quote)
            if verified:
                evidence.append(
                    {
                        "chunk_id": chunk_id,
                        "quote": verified,
                        "heading": chunk.heading,
                        "lines": (chunk.line_start, chunk.line_end),
                    }
                )

        nudge = _truncate(raw.get("nudge", ""), 220)
        missing = [_truncate(m, 80) for m in (raw.get("missing") or [])][:3]

        if verdict == NOT_IN_NOTES:
            coverage = term_coverage(question.text, [c.text for c in self.store.chunks])
            if coverage >= COVERAGE_TO_COVER:
                # The notes demonstrably contain this question's own terms, so a
                # refusal is provably wrong. We do not trust the model on this.
                verdict, score = INCOMPLETE, 0
                missing = []
                nudge = (
                    "Your notes do cover this one, but your answer did not match them. "
                    "Find the section and try again."
                )
            else:
                evidence, missing, score = [], [], 0
                nudge = nudge or "Your notes do not cover this. Worth flagging to whoever set the reading."
        else:
            if verdict == CORRECT and not evidence:
                verdict = INCOMPLETE
                score = min(score, 1)
                nudge = nudge or "Close, but I could not find a line in your notes that confirms it - go and check the passage."
            if score == 2 and verdict != CORRECT:
                verdict = INCOMPLETE
            if verdict == CORRECT:
                score = 2
                nudge = ""  # nothing to nudge about
            elif not nudge:
                nudge = "Go back to that part of your notes and read it once more."

        if nudge and spoils_facts(nudge, missing):
            # The model handed over the answer instead of pointing at it.
            nudge = (
                f"Your notes do cover this - re-read the {question.topic} section "
                "and check it against what you wrote."
            )

        return Verdict(
            question_id=question.id,
            verdict=verdict,
            score=score,
            topic=_truncate(raw.get("topic", "") or question.topic, 48),
            nudge=nudge,
            evidence=evidence[:2],
            missing=missing,
        )

    # --------------------------------------------------------------- wrap-up

    def summarise(self, verdicts: list[Verdict]) -> dict:
        """Turn the session's verdicts into a next step."""
        if not verdicts:
            return {"weak_topics": [], "next_step": "Answer a few questions first.", "encouragement": ""}

        lines = []
        for verdict in verdicts:
            detail = "; ".join(verdict.missing) if verdict.missing else verdict.label
            lines.append(f"- {verdict.topic} ({verdict.label}, score {verdict.score}/2): {detail}")

        raw = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": "You give short, specific study advice based on a student's own graded answers. You reply with JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        "Here is how a student did on a revision quiz built from their own notes:\n\n"
                        + "\n".join(lines)
                        + "\n\nGive weak_topics (the 1-3 topics worth re-reading, as short labels), "
                        "next_step (ONE concrete action, under 20 words) and encouragement (ONE sentence, "
                        "never generic praise). Reply with JSON only."
                    ),
                },
            ],
            schema=SUMMARY_SCHEMA,
            options={"temperature": 0.4},
        ) or {}

        return {
            "weak_topics": [_truncate(t, 40) for t in (raw.get("weak_topics") or [])][:3],
            "next_step": _truncate(raw.get("next_step", ""), 140),
            "encouragement": _truncate(raw.get("encouragement", ""), 160),
        }
