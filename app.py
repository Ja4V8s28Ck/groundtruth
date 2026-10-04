"""Groundtruth - a study partner that only answers from your own notes.

Run it with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from package.ingest import chunk_document, document_stats, load_document  # noqa: E402
from package.ollama import (  # noqa: E402
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    OllamaClient,
    OllamaError,
    clear_cache,
)
from package.store import NoteStore  # noqa: E402
from package.tutor import CORRECT, INCOMPLETE, NOT_IN_NOTES, Tutor, Verdict  # noqa: E402

SAMPLE = Path(__file__).parent / "samples" / "cell_bio_notes.md"

BADGE = {
    CORRECT: ":green-badge[Correct]",
    INCOMPLETE: ":orange-badge[Not quite]",
    NOT_IN_NOTES: ":red-badge[Not in your notes]",
}
BADGE_COLOR = {CORRECT: "green", INCOMPLETE: "orange", NOT_IN_NOTES: "red"}
KIND_ICON = {"recall": ":material/short_text:", "explain": ":material/record_voice_over:", "apply": ":material/science:"}

# Everything the app remembers between clicks.
DEFAULTS = {
    "chat_model": DEFAULT_CHAT_MODEL,
    "embed_model": DEFAULT_EMBED_MODEL,
    "doc_text": None,
    "chunks": [],
    "store": None,
    "questions": [],
    "verdicts": {},
    "summary": None,
    "summary_attempted": False,
    "bet": None,
    "ready": False,
    "retrieval_mode": "pending",
    # Which action a click queued, and which one is currently running. See
    # dispatch() for why the work happens a run later rather than inline.
    "pending": None,
    "busy": None,
}


# --------------------------------------------------------------------- helpers


def client() -> OllamaClient:
    return OllamaClient(
        chat_model=st.session_state.chat_model,
        embed_model=st.session_state.embed_model,
    )


def highlight(text: str, quote: str) -> str:
    """Bold the cited quote inside the passage it came from. Plain markdown, no HTML."""
    position = text.lower().find(quote.lower())
    if position == -1:
        return text
    return f"{text[:position]}**{text[position:position + len(quote)]}**{text[position + len(quote):]}"


def reset() -> None:
    for key in DEFAULTS:
        if key not in ("chat_model", "embed_model"):
            st.session_state.pop(key, None)


def busy() -> bool:
    """True while a slow action is running, so every button can grey itself out."""
    return st.session_state.busy is not None


def queue(action: str, **payload) -> None:
    """Queue a slow action and rerun.

    Streamlit reruns the entire script on every interaction, and elements are
    drawn as the script runs. So a button that called Ollama inline would already
    have been painted enabled by the time the slow call blocked, and the user
    could click it again - queueing a second identical model call.

    Queuing instead means this pass finishes immediately and the next one starts
    with `busy` set, so the button is rendered greyed out for the whole duration
    of the work. A disabled button also stops the double-click.
    """
    st.session_state.pending = (action, payload)
    st.rerun()


def load(text: str = "", filename: str = "", data: bytes | None = None) -> None:
    """Ingest a document and index it for retrieval."""
    with st.spinner("Reading and indexing your notes…"):
        text = load_document(text=text, filename=filename, data=data)
        chunks = chunk_document(text)
        store = NoteStore(chunks, client())
        bar = st.progress(0.0, "Indexing your notes…")
        mode = store.build(progress=lambda fraction, message: bar.progress(min(fraction, 1.0), message))
        bar.empty()

    st.session_state.doc_text = text
    st.session_state.chunks = chunks
    st.session_state.store = store
    st.session_state.retrieval_mode = mode
    st.session_state.questions = []
    st.session_state.verdicts = {}
    st.session_state.summary = None
    st.session_state.summary_attempted = False
    st.session_state.bet = None


def dispatch() -> None:
    """Run whatever slow action was queued, with every button disabled.

    Called once per script run, before any page is drawn. `busy` is set for the
    whole duration so the rendering that follows shows the disabled state, and
    it is cleared in a `finally` so a failed model call cannot leave the app
    permanently greyed out.
    """
    pending = st.session_state.pop("pending", None)
    if not pending:
        return

    action, payload = pending
    st.session_state.busy = action
    try:
        _run(action, payload)
    finally:
        st.session_state.busy = None


def _run(action: str, payload: dict) -> None:
    """The slow work itself. One branch per queueable action."""
    if action == "load_sample":
        load(SAMPLE.read_text(encoding="utf-8"), SAMPLE.name)
        st.rerun()

    elif action == "load_upload":
        try:
            load(filename=payload.get("filename", ""), data=payload.get("data"))
        except Exception as exc:  # noqa: BLE001
            st.session_state.ready = True  # do not retry the same file in a loop
            st.error(str(exc), icon=":material/error:")
            return
        st.session_state.ready = True
        st.rerun()

    elif action == "load_paste":
        text = payload.get("text", "")
        if not text.strip():
            return
        try:
            load(text)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc), icon=":material/error:")
            return
        st.rerun()

    elif action == "build":
        tutor = Tutor(st.session_state.store, client())
        with st.status("Reading your notes and writing questions…", expanded=True) as status:
            try:
                questions = tutor.build_session(
                    count=payload.get("count", 8),
                    progress=lambda fraction, message: status.update(label=message, state="running"),
                )
            except OllamaError as exc:
                status.update(label="Something went wrong", state="error")
                st.error(str(exc), icon=":material/error:")
                return
            status.update(label=f"{len(questions)} questions ready", state="complete")

        if not questions:
            st.warning("The model did not produce any questions from that material. Try more notes.")
            return
        st.session_state.questions = questions
        st.rerun()

    elif action == "grade":
        question = next(q for q in st.session_state.questions if q.id == payload.get("qid"))
        answer = st.session_state.get(f"answer_{question.id}", "")
        try:
            with st.spinner("Checking against your notes…"):
                verdict = Tutor(st.session_state.store, client()).grade_answer(question, answer)
        except (OllamaError, ValueError) as exc:
            st.error(str(exc), icon=":material/error:")
            return
        st.session_state.verdicts[question.id] = verdict
        st.rerun()

    elif action == "summarise":
        st.session_state.summary_attempted = True
        try:
            with st.spinner("Working out what to re-read…"):
                st.session_state.summary = Tutor(st.session_state.store, client()).summarise(
                    list(st.session_state.verdicts.values())
                )
        except OllamaError as exc:
            st.error(str(exc), icon=":material/error:")


# ----------------------------------------------------------------------- setup

st.set_page_config(
    page_title="Groundtruth",
    page_icon=":material/bookmark:",
    layout="centered",
)

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

probe = OllamaClient()

with st.sidebar:
    st.markdown("### :material/bookmark: Groundtruth")
    st.caption("Local model. Your notes never leave this machine.")

    health = probe.health()

    if not health["online"]:
        st.error("Ollama is not reachable.", icon=":material/error:")
        st.markdown("Start it, then reload:\n\n```\nollama serve\n```")
        st.stop()

    st.success(f"Ollama {health['version']} running", icon=":material/check:")

    chat_options = health["chat_models"]
    embed_options = health["embed_models"]

    if not chat_options:
        st.warning("No chat model installed.", icon=":material/warning:")
        st.code(f"ollama pull {DEFAULT_CHAT_MODEL}", language="bash")
        st.stop()

    if not embed_options:
        st.caption(
            f"No embedding model found - search will use keywords. "
            f"For better results: `ollama pull {DEFAULT_EMBED_MODEL}`"
        )
        embed_options = [DEFAULT_EMBED_MODEL]

    st.selectbox(
        "Chat model",
        chat_options,
        index=0 if st.session_state.chat_model not in chat_options else chat_options.index(st.session_state.chat_model),
        key="chat_model",
        help="Writes the questions and grades you. Bigger models grade better but run slower.",
    )
    st.selectbox(
        "Embedding model",
        embed_options,
        index=0 if st.session_state.embed_model not in embed_options else embed_options.index(st.session_state.embed_model),
        key="embed_model",
        help="Finds the right passage in your notes.",
    )

    st.button(
        "Refresh model list",
        icon=":material/refresh:",
        on_click=clear_cache,
        width="stretch",
    )

    if st.session_state.doc_text:
        st.button(
            "Start over",
            icon=":material/restart_alt:",
            on_click=reset,
            width="stretch",
        )


def render_landing() -> None:
    st.title("Groundtruth", icon=":material/bookmark:")
    st.markdown(
        "A revision partner that will only answer from **your own notes** "
        "and shows you the exact line it used."
    )

    steps = st.container(horizontal=True)
    with steps:
        for icon, heading, body in (
            (":material/upload_file:", "Drop in your notes", "PDF, Markdown or pasted text."),
            (":material/quiz:", "Get quizzed on them", "Questions from your material, not the internet."),
            (":material/format_quote:", "See the evidence", "Every verdict cites the line that settles it."),
        ):
            with st.container(border=True):
                st.markdown(f"#### {icon} {heading}")
                st.caption(body)

    with st.container(border=True):
        st.markdown("#### Start with the sample")
        st.caption("A student's cell biology lecture notes, so you can try it with zero setup.")
        if st.button(
            "Use the sample notes",
            type="primary",
            icon=":material/science:",
            disabled=busy(),
        ):
            queue("load_sample")

    with st.container(border=True):
        st.markdown("#### Use your own")
        uploaded = st.file_uploader(
            "Upload notes",
            type=["pdf", "txt", "md"],
            disabled=busy(),
            help="Text-based PDFs only - scanned pages are not OCR'd.",
        )
        if uploaded is not None and not st.session_state.ready and not st.session_state.pending:
            queue("load_upload", filename=uploaded.name, data=uploaded.getvalue())

        with st.form("paste_notes", border=False):
            pasted = st.text_area(
                "Or paste your notes",
                height=140,
                disabled=busy(),
                placeholder="Paste your lecture notes here…",
            )
            submitted = st.form_submit_button(
                "Use pasted notes",
                icon=":material/content_paste:",
                disabled=busy(),
            )

        if submitted:
            queue("load_paste", text=pasted)

    st.caption("No API keys, no cost, works offline. Every answer cites a line in your own file.")


def render_loaded() -> None:
    chunks = st.session_state.chunks
    stats = document_stats(st.session_state.doc_text, chunks)

    st.header("Your notes are loaded", icon=":material/check_circle:")
    metrics = st.container(horizontal=True)
    with metrics:
        for label, value in (
            ("Words", f"{stats['words']:,}"),
            ("Sections", str(stats["chunks"])),
            ("Search", "semantic" if st.session_state.retrieval_mode == "semantic" else "keyword"),
            ("Model", st.session_state.chat_model),
        ):
            with st.container(border=True):
                st.metric(label, value)

    st.space("medium")

    st.markdown("#### Build your session")
    count = st.slider("How many questions?", 4, 12, 8, disabled=busy())
    st.caption(f"Questions are spread across all {stats['chunks']} sections, not just the first page.")

    # `on_change="rerun"` opts into state tracking, which is what makes the
    # `.open` guard below meaningful. Without it `.open` is always False and the
    # expander stays empty no matter what the user clicks.
    first = st.expander(
        "What the model is looking at",
        icon=":material/visibility:",
        on_change="rerun",
    )
    if first.open:
        with first:
            st.code(chunks[0].text[:700] + "…", language=None)

    if st.button(
        "Build my session",
        type="primary",
        icon=":material/auto_awesome:",
        disabled=busy(),
    ):
        queue("build", count=count)


def render_quiz() -> None:
    questions = st.session_state.questions
    verdicts: dict[int, Verdict] = st.session_state.verdicts
    answered = len(verdicts)

    st.header(f"Your session · {answered} of {len(questions)} graded", icon=":material/quiz:")

    if st.session_state.bet is None:
        st.info(
            f"Before you start: how many of these {len(questions)} will you get right? "
            "Groundtruth checks you at the end.",
            icon=":material/lightbulb:",
        )
        st.session_state.bet = st.number_input(
            "My prediction",
            min_value=0,
            max_value=len(questions),
            value=len(questions) // 2,
            step=1,
        )

    st.progress(answered / len(questions))
    st.space("small")

    for question in questions:
        verdict = verdicts.get(question.id)

        with st.container(border=True):
            row = st.container(horizontal=True)
            with row:
                st.badge(question.topic, color="blue")
                st.caption(f"{KIND_ICON.get(question.kind, ':material/help:')} {question.kind_label}")

            st.markdown(f"**{question.text}**")

            if verdict is None:
                st.text_area(
                    "Your answer",
                    key=f"answer_{question.id}",
                    height=90,
                    disabled=busy(),
                    placeholder="Answer in your own words…",
                    label_visibility="collapsed",
                )
                if st.button(
                    "Check my answer",
                    key=f"check_{question.id}",
                    type="primary",
                    icon=":material/check:",
                    disabled=busy(),
                ):
                    queue("grade", qid=question.id)
                continue

            st.markdown(
                f"{BADGE[verdict.verdict]} :grey[{verdict.score}/2]"
            )

            for piece in verdict.missing:
                st.markdown(f"- missed: {piece}")

            if verdict.evidence:
                st.markdown("**From your notes**")
                for item in verdict.evidence:
                    chunk = next((c for c in st.session_state.chunks if c.id == item["chunk_id"]), None)
                    if chunk is None:
                        continue
                    start, end = item["lines"]
                    with st.container(border=True):
                        st.markdown(highlight(chunk.text, item["quote"]))
                        st.caption(f"section {item['chunk_id']} · lines {start}–{end}")
            else:
                st.caption("No supporting line found in your notes.")

            if verdict.nudge:
                st.info(verdict.nudge, icon=":material/lightbulb:")

            # Retrieval costs a model call, so only do it once the panel is open.
            # The key is required, not cosmetic: this sits in a loop, and Streamlit
            # derives element IDs from type plus params, so ten identical
            # expanders raise StreamlitDuplicateElementId.
            passage = st.expander(
                "Show the whole passage",
                key=f"passage_{question.id}",
                icon=":material/article:",
                on_change="rerun",
            )
            if passage.open:
                hits = st.session_state.store.search(question.text, k=1)
                if hits:
                    st.markdown(hits[0].text)
                    st.caption(f"section {hits[0].chunk.id} · {hits[0].chunk.cite()}")


def render_results() -> None:
    questions = st.session_state.questions
    verdicts = st.session_state.verdicts

    score = sum(v.score for v in verdicts.values())
    correct_count = sum(1 for v in verdicts.values() if v.verdict == CORRECT)
    missing_count = sum(1 for v in verdicts.values() if v.verdict == NOT_IN_NOTES)
    bet = st.session_state.bet or 0

    st.header("Your result", icon=":material/emoji_events:")

    metrics = st.container(horizontal=True)
    with metrics:
        for label, value, caption in (
            ("Score", f"{score}/{len(questions) * 2}", None),
            ("Fully correct", f"{correct_count}/{len(questions)}", None),
            ("Not in your notes", str(missing_count), None),
            (
                "Your bet",
                str(bet),
                "nailed it" if correct_count >= bet else f"{bet - correct_count} short",
            ),
        ):
            with st.container(border=True):
                st.metric(label, value)
                if caption:
                    st.caption(caption)

    st.space("small")

    rows = []
    for question in questions:
        verdict = verdicts.get(question.id)
        rows.append(
            {
                "topic": question.topic,
                "score": verdict.score if verdict else 0,
                "result": (verdict.label if verdict else "Not answered"),
            }
        )
    st.bar_chart(
        pd.DataFrame(rows),
        x="score",
        y="topic",
        color="result",
        horizontal=True,
        alt="Score per topic, coloured by verdict. Green topics were fully correct.",
    )

    if st.session_state.summary is None and not st.session_state.summary_attempted:
        # Queued rather than called inline, so the page does not appear frozen
        # while the model writes the wrap-up. `summary_attempted` stops a failed
        # call from queueing itself forever.
        queue("summarise")

    summary = st.session_state.summary or {}
    if summary.get("weak_topics"):
        topics = " ".join(f":orange-badge[{topic}]" for topic in summary["weak_topics"])
        st.markdown(f"**Re-read these:** {topics}")
    if summary.get("next_step"):
        st.info(summary["next_step"], icon=":material/arrow_forward:")
    if summary.get("encouragement"):
        st.caption(summary["encouragement"])

    st.caption(
        "Every judgement above is tied to a line in your own notes. Ask about something your "
        "notes never covered and Groundtruth will tell you instead of guessing."
    )


# ----------------------------------------------------------------------- route

dispatch()

if not st.session_state.doc_text:
    render_landing()
    st.stop()

render_loaded()

if not st.session_state.questions:
    st.stop()

if len(st.session_state.verdicts) < len(st.session_state.questions):
    render_quiz()
    st.stop()

st.space("medium")
render_results()
