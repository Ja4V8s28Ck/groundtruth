# Groundtruth

A revision partner that only answers from a student's own notes and shows the exact line it used for every verdict.

## Why I built it

My cousin asked me to be his revision helper. He was working through college coursework and wanted someone to quiz him on it. I could say yes to the first session and not to the six after it, so I built the part that does not need me in the room.

He had a specific problem before this. He re-read his notes until they felt familiar, then walked into an exam believing he knew the material. When he asked a chatbot to quiz him instead, it explained things that were never in his syllabus, invented citations that looked real and told him he was right when he was not. That is not a bug in that tool. It has every fact on the internet and no idea which lines of his notes are the ones that matter.

Groundtruth reads his notes and does three things:

1. writes questions that can be answered only from that material
2. grades his free-text answers against it
3. shows the exact line that settles every verdict

If his notes do not cover something, it says so rather than guessing. He was happy to use it, which is the part I did not expect going in.

## Why it runs on your machine

Three reasons, in the order they mattered to us.

**The notes are not mine to upload.** Course material, a lecturer's slides and the record of what a student is weak at are not the builder's to hand to a third-party server. It runs over localhost, so the notes never leave the machine. There is no server to leak them to because there is no server.

**It has to work with no network.** Revision happens at 7am in a quiet room the night before an exam. Depending on a cloud API means depending on someone else's uptime and someone else's billing.

**The behaviour is the product, so we need to be able to change it.** The refusal, the grading rubric and the tone of the hint all live in a prompt we can read. The model can be swapped from the sidebar without touching code.

There is a fourth argument about cost and it is the weakest. A small model on your own hardware is free and renting an equivalent one costs a small amount per session. Free is not the interesting part.

The part that convinced me to build it was the opposite one. Handing a model a single passage and telling it that the passage is the entire world makes its job easier, not harder. Grading inside a closed boundary is a more demanding task than answering from everything in the training data. The grounding is not a limitation we tolerate. It is the reason the output is worth checking in the first place.

## Quickstart

You need [Ollama](https://ollama.com) installed and running. Then:

```bash
git clone https://github.com/Ja4V8s28Ck/groundtruth.git
cd groundtruth

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3:4b           # writes and grades, 2.5 GB
ollama pull nomic-embed-text   # finds passages, 274 MB

streamlit run app.py
```

Open the URL Streamlit prints and click **Use the sample notes**. You have a working session in about 20 seconds.

`nomic-embed-text` is optional. Without it, Groundtruth falls back to keyword search on its own. It just finds passages less well.

The app reads each model's capabilities from Ollama instead of guessing from its name, so an embedding model cannot end up in the chat box or the other way round. If you pull something new while the app is open, press **Refresh model list** in the sidebar.

## How the AI is used

Take the model out and there is no product left. There is no keyword path to writing a good question.

| Step | What the model does | Why rules cannot do it |
| --- | --- | --- |
| **Writes the questions** | Reads each section and decides what is testable: which claims matter, which need a mechanism spelled out, which could be turned into a scenario. | It has to read forty pages and judge importance. Nothing matches "worth asking about". |
| **Grades the answer** | Compares the answer to the retrieved passage semantically. "Only small uncharged things get through" counts as correct against notes that say selectively permeable and that nonpolar molecules cross freely. | Keyword matching fails the moment the student uses their own words, which is the entire point. |
| **Refuses** | When the retrieved passages do not address the question, it returns `not_in_notes` instead of falling back on training data. | Only a model can tell covered from not covered. |
| **Chooses what to re-read** | Reads the session's verdicts and names the topics worth going back to. | It has to weigh which of the misses matter together. |

### Citations you can check

Most AI study tools tell you whether you were right. This one shows why it thinks so, in the words of your own notes.

Every chunk keeps the line range it came from. When the model grades an answer it has to return a verbatim quote and the chunk id. Back in Python, that quote is checked against the real text with a sliding-window match, which tolerates the model paraphrasing slightly and then returns the exact span from the file.

If the model returns `correct` and cannot produce a quote that exists in the notes, the verdict is downgraded rather than shown. That is the whole trust model. You can open the file and check every claim the app made.

### Checks on the model's judgement

A 4B model is small enough to run on a laptop and loose enough to grade generously. Three checks in `store.py` catch the cases where it would otherwise wave something through:

- **Quote verification.** A `correct` verdict with no matching quote in the notes is never displayed.
- **Term coverage.** If the retrieved passage covers almost none of the terms in the answer, the verdict is downgraded. This is what makes the refusal falsifiable: the app cannot be wrong by inventing and it cannot be wrong by being vague.
- **No spoilers.** The hint shown for a wrong answer is checked so it does not hand over the answer itself.

### Two things worth trying

Ask it about mitosis. The sample notes do not cover mitosis, so it will say so, because they do not. Watching a model decline to be useful on purpose is the demo moment and it is the opposite of what a chatbot does when it is unsure.

Before the session it asks how many questions you think you will get right, then tells you at the end. It turns revision from vague guilt into a scoreboard you argued with.

## How it is built

```
groundtruth/
  app.py                    Streamlit UI
  .streamlit/config.toml    native theme tokens, no custom CSS anywhere
  package/
    ollama.py               client for /api/chat, /api/embed, /api/show, /api/tags
    ingest.py               PDF, markdown and text into chunks that remember line numbers
    store.py                retrieval, plus quote verification and the guards above
    tutor.py                prompts and schemas for questions, grading and the summary
  samples/
    cell_bio_notes.md       a student's notes, so you can try it with no setup
```

- **Structured output, not prompt-wishing.** Grading hands a JSON Schema to Ollama, so the verdict shape is guaranteed rather than hoped for. There is still a fenced-JSON extractor and one retry behind it, because a failed grading call mid-session is the one thing that must not happen.
- **Capability-aware model picking.** `/api/show` reports which models can chat and which can only embed, so the two dropdowns cannot be confused.
- **Questions spread across the document.** They are sampled from every section, so a session covers the whole set of notes rather than the first page.
- **No AI SDK, no vector database, no cloud SDK, no CSS.** The dependencies are `requests` and `numpy`.

### Swapping the model

Pick another one from the sidebar, no code change. A small model on your laptop grades loosely and a larger open-weight one grades better.

| Model | Size | Notes |
| --- | --- | --- |
| `qwen3:4b` | 2.5 GB | Default. Runs on a laptop. |
| `llama3.2:3b` | 2 GB | Lighter, slightly weaker grading. |
| `qwen3:8b` | 5 GB | Noticeably better grading, still CPU-friendly. |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Could not reach Ollama` | Start it with `ollama serve`. |
| `Model 'x' is not installed` | `ollama pull x`. The exact name is in the message. |
| `'x' is an embedding model and cannot hold a conversation` | You picked an embedding model as the chat model. Choose from the **Chat model** list. |
| `No chat model installed` | `ollama pull qwen3:4b`. |
| A newly pulled model is not listed | Press **Refresh model list** in the sidebar. |
| "Search: keyword" in the header | No embedding model installed. `ollama pull nomic-embed-text` upgrades it. |
| Grading is slow | Expected on CPU with a large model. Switch to a smaller one in the sidebar. |
| PDF came out empty | It is probably a scan of images. Groundtruth reads text, it does not OCR. |

## Known limitations

- No OCR, so scanned PDFs will not work.
- Chunking is layout-aware but not table-aware.
- One document per session, by design.
- Grading is only as good as the model. It shows you the evidence so you can disagree with it, but a weak model still makes weak calls.
- Building a session makes one model call per section of the notes, so a long document takes a while.
