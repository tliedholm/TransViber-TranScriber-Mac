"""Local LLM notes generation (mlx-lm, Apple-silicon GPU)."""

from __future__ import annotations

import datetime
import gc

from PySide6.QtCore import QThread, Signal

import diagnostics

# UI tier names -> mlx-community 4-bit conversions, mirroring the PC app's
# tiers (which use CUDA/bnb-4bit repos) at equivalent sizes.
NOTES_MODEL_GROUPS = [
    ("Max Detail (Large Tier)", ["qwen3-14b", "gemma-3-12b", "mistral-nemo-12b"]),
    ("Standard Speed (Medium Tier)", ["qwen3-8b", "llama-3.1-8b", "ministral-8b"]),
    ("Quick Review (Small Tier)", ["qwen3-4b", "llama-3.2-3b", "phi-4-mini",
                                   "granite-3.3-2b"]),
]

LLM_REPOS = {
    "qwen3-14b": "mlx-community/Qwen3-14B-4bit",
    # Use the text-only conversion: the plain gemma-3 repo is a vision model
    # for mlx-vlm and cannot be loaded by this module's mlx-lm runtime.
    "gemma-3-12b": "mlx-community/gemma-3-text-12b-it-4bit",
    "mistral-nemo-12b": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
    "llama-3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "ministral-8b": "mlx-community/Ministral-8B-Instruct-2410-4bit",
    "qwen3-4b": "mlx-community/Qwen3-4B-4bit",
    "llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "phi-4-mini": "mlx-community/Phi-4-mini-instruct-4bit",
    "granite-3.3-2b": "mlx-community/granite-3.3-2b-instruct-4bit",
}

DEFAULT_NOTES_MODEL = "qwen3-4b"

# The body wording is user-approved — change only on request. The tail
# sections ('Where we left off', 'Unresolved threads') exist so the next
# session can start from where this one stopped; keep them last and in order.
TAIL_SECTIONS = (
    "Then a 'Where we left off' section: where things stood at the very end "
    "of the session — the situation in progress, anything interrupted or "
    "unfinished, and the immediate next steps to pick up from. "
    "Finish with an 'Unresolved threads' section listing every thread still "
    "open, one bullet each, and for each one give: what it is, who it "
    "involves, its last known status, and the suggested next action."
)

DEFAULT_PROMPT = (
    "Write concise, well-organized Markdown notes on this conversation "
    "transcript: a short summary, key points and decisions, memorable "
    "moments or quotes with speaker attribution, and open questions or "
    "follow-ups. Use the speakers' names. Then a 'Per-person summary' "
    "section: for each speaker (and any character they play), a full-session "
    "summary of what they did, said, decided, and what happened to them. "
    + TAIL_SECTIONS
)

# Per-part pass of a chunked transcript: no tail sections (a middle part's
# ending is not the session's), but flag what it leaves open so the combine
# pass has the material to write them.
PART_PROMPT = (
    "Write concise, well-organized Markdown notes on this part of a longer "
    "conversation transcript: a short summary, key points and decisions, "
    "memorable moments or quotes with speaker attribution, and open "
    "questions or follow-ups. Use the speakers' names. Then a 'Per-person "
    "summary' section: for each speaker (and any character they play), a "
    "summary of what they did, said, decided, and what happened to them. "
    "Also note how this part ends and anything left unfinished or "
    "unresolved in it, including who is involved and its status."
)

COMBINE_PROMPT = (
    "The following are Markdown notes taken over consecutive parts of one "
    "long conversation transcript, in order. Merge them into one set of "
    "concise, well-organized Markdown notes with this structure: summary, "
    "key points and decisions, memorable moments, open questions, then a "
    "'Per-person summary' section covering the whole session per speaker. "
    + TAIL_SECTIONS +
    " Base those two final sections on the end of the whole session — the "
    "last part's ending — and drop any thread the later parts resolved. "
    "Remove duplication; keep speaker names."
)

# Intermediate level of a hierarchical combine: still a middle of the session,
# so no tail sections — only the final pass uses COMBINE_PROMPT.
MERGE_PROMPT = (
    "The following are Markdown notes taken over consecutive parts of one "
    "long conversation transcript, in order. Merge them into one set of "
    "notes covering this stretch of the session, keeping the same structure: "
    "summary, key points and decisions, memorable moments, open questions, "
    "then a 'Per-person summary' section per speaker. Also keep how this "
    "stretch ends and anything left unfinished in it. Remove duplication; "
    "keep speaker names."
)

MAX_NOTES_TOKENS = 6_000  # generation cap per pass, also drives the percent

# Context window assumed when the model does not report one. Deliberately
# below the smallest window in NOTES_MODEL_GROUPS rather than an optimistic
# guess: overshooting silently truncates the transcript inside the runtime.
DEFAULT_CONTEXT_TOKENS = 32_768
CONTEXT_LIMITS = (2_048, 262_144)  # sanity clamp on whatever a model claims

# Slack for the chat template, the transcript's own headers, and tokenizer
# disagreement between counting and generating.
PROMPT_MARGIN_TOKENS = 512

# One LLM kept loaded between runs (reruns with tweaked instructions are
# common); freed when a transcription starts — see free_llm().
_llm_cache: dict = {"key": None, "model": None, "tokenizer": None}


def free_llm() -> None:
    """Drop the cached notes LLM so ASR models can have the memory."""
    _llm_cache["key"] = None
    _llm_cache["model"] = None
    _llm_cache["tokenizer"] = None
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass


def _get_llm(model_name: str, status_cb):
    repo = LLM_REPOS.get(model_name.lower().strip(), model_name)
    if _llm_cache["key"] != repo:
        free_llm()
        status_cb(f"Loading notes model '{model_name}' — first use downloads it…")
        from mlx_lm import load

        model, tokenizer = load(repo)
        _llm_cache.update(key=repo, model=model, tokenizer=tokenizer)
    return _llm_cache["model"], _llm_cache["tokenizer"]


def _build_prompt(tokenizer, model_name: str, system_text: str, user_text: str) -> str:
    """Chat-template a system+user pair, handling per-family quirks."""
    name = model_name.lower()
    kwargs = {}
    if "qwen3" in name:
        kwargs["enable_thinking"] = False  # notes, not chain-of-thought
    if "gemma" in name:
        # Gemma has no system role: fold the system text into the user turn.
        messages = [{"role": "user", "content": system_text + "\n\n" + user_text}]
    else:
        messages = [{"role": "system", "content": system_text},
                    {"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs)


def context_window(model, tokenizer) -> int:
    """The model's context length, clamped to something believable."""
    for source, attribute in ((getattr(model, "args", None), "max_position_embeddings"),
                              (getattr(model, "config", None), "max_position_embeddings"),
                              (tokenizer, "model_max_length")):
        value = getattr(source, attribute, None)
        if isinstance(value, int) and CONTEXT_LIMITS[0] <= value <= CONTEXT_LIMITS[1]:
            return value
    return DEFAULT_CONTEXT_TOKENS


def input_budget(tokenizer, context: int, system_text: str) -> int:
    """How many tokens of transcript one pass can take.

    The old constants budgeted the input alone. A pass also has to fit its
    system prompt and everything it generates — up to MAX_NOTES_TOKENS — in
    the same window, so both are subtracted here.
    """
    overhead = len(tokenizer.encode(system_text)) + MAX_NOTES_TOKENS + PROMPT_MARGIN_TOKENS
    return max(1_000, context - overhead)


def _line_pieces(line: str, tokenizer, limit: int) -> list[str]:
    """A line, subdivided at token boundaries if it is too big on its own.

    coalesce() joins consecutive same-speaker segments into one line, so a
    monologue or a single-speaker recording can produce one line longer than
    the whole context window. Splitting only at line boundaries left such a
    line whole and blew the budget.
    """
    ids = tokenizer.encode(line)
    if len(ids) <= limit:
        return [line]
    return [tokenizer.decode(ids[i:i + limit]) for i in range(0, len(ids), limit)]


def split_text(text: str, tokenizer, limit: int) -> list[str]:
    """Split text into parts of at most `limit` tokens, preferring line breaks."""
    if len(tokenizer.encode(text)) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in text.splitlines(keepends=True):
        for piece in _line_pieces(line, tokenizer, limit):
            n = len(tokenizer.encode(piece))
            if current and current_tokens + n > limit:
                parts.append("".join(current))
                current, current_tokens = [], 0
            current.append(piece)
            current_tokens += n
    if current:
        parts.append("".join(current))
    return parts


def batch_by_tokens(chunks: list[str], tokenizer, limit: int, joiner: str) -> list[list[str]]:
    """Group already-written notes into batches that each fit one pass.

    The combine pass used to concatenate every part summary into a single
    prompt with no ceiling: enough parts at MAX_NOTES_TOKENS each overflowed
    the window on exactly the long sessions that needed chunking.
    """
    join_cost = len(tokenizer.encode(joiner))
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for chunk in chunks:
        n = len(tokenizer.encode(chunk))
        if current and current_tokens + join_cost + n > limit:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(chunk)
        current_tokens += n + (join_cost if current_tokens else 0)
    if current:
        batches.append(current)
    return batches


class NotesWorker(QThread):
    status = Signal(str)
    chunk = Signal(str)  # streamed text for the notes editor
    progress = Signal(int, int, float)  # (percent 0-100, cumulative tokens, tok/s)
    download_progress = Signal(int, float)  # shared model-download readout
    completed = Signal()
    failed = Signal(str)

    def __init__(self, transcript: str, instructions: str, model_name: str,
                 parent=None):
        super().__init__(parent)
        self.transcript = transcript
        self.instructions = instructions.strip()
        self.model_name = model_name.strip() or DEFAULT_NOTES_MODEL
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    # ----- generation -----

    def _system_text(self, base: str = DEFAULT_PROMPT) -> str:
        text = base
        if self.instructions:
            text += "\n\nAdditional instructions: " + self.instructions
        return text

    def _generate(self, model, tokenizer, prompt: str, stream: bool) -> str:
        """One generation pass; streams chunks to the editor if `stream`."""
        from mlx_lm import stream_generate

        pieces = []
        for response in stream_generate(model, tokenizer, prompt,
                                        max_tokens=MAX_NOTES_TOKENS):
            if self._stop:
                break
            pieces.append(response.text)
            if stream:
                self.chunk.emit(response.text)
            self._tokens += 1
            # mlx-lm reports decode speed itself; excludes prompt processing.
            self._last_tps = response.generation_tps
            frac = min(self._tokens - self._pass_base, MAX_NOTES_TOKENS) / MAX_NOTES_TOKENS
            percent = int((self._passes_done + min(frac, 0.99)) / self._passes_total * 100)
            self.progress.emit(min(percent, 99), self._tokens, self._last_tps)
        self._passes_done += 1
        self._pass_base = self._tokens
        return "".join(pieces)

    JOINER = "\n\n---\n\n"

    def _combine(self, model, tokenizer, notes: list[str], context: int) -> None:
        """Reduce part summaries to one final set of notes, in bounded passes.

        Batches that do not all fit in one window are merged a level at a
        time (MERGE_PROMPT) until a single batch remains; only that last pass
        uses COMBINE_PROMPT and streams to the editor, so the tail sections
        are written once, from the whole session.
        """
        final_budget = input_budget(tokenizer, context, self._system_text(COMBINE_PROMPT))
        merge_budget = input_budget(tokenizer, context, MERGE_PROMPT)
        # Batching cannot help if one summary is bigger than a whole pass, so
        # split those first: every batch then provably fits. Generation is
        # capped at MAX_NOTES_TOKENS and the budgets already subtract it, so
        # this is a backstop rather than the normal path.
        piece_limit = min(final_budget, merge_budget)
        level = 0
        while True:
            if self._stop:
                return
            notes = [piece for note in notes
                     for piece in split_text(note, tokenizer, piece_limit)]
            batches = batch_by_tokens(notes, tokenizer, final_budget, self.JOINER)
            if len(batches) == 1:
                self.status.emit("Combining notes…")
                self._generate(model, tokenizer,
                               _build_prompt(tokenizer, self.model_name,
                                             self._system_text(COMBINE_PROMPT),
                                             self.JOINER.join(batches[0])),
                               stream=True)
                return
            level += 1
            batches = batch_by_tokens(notes, tokenizer, merge_budget, self.JOINER)
            # Each extra level is passes the initial estimate did not know about.
            self._passes_total += len(batches)
            merged = []
            for n, batch in enumerate(batches, 1):
                if self._stop:
                    return
                self.status.emit(
                    f"Merging notes (round {level}, {n}/{len(batches)})…")
                merged.append(self._generate(
                    model, tokenizer,
                    _build_prompt(tokenizer, self.model_name, MERGE_PROMPT,
                                  self.JOINER.join(batch)),
                    stream=False))
            if len(merged) >= len(notes):
                # No progress (one oversized note per batch): stop reducing and
                # let the final pass run on what there is rather than loop.
                notes = merged
                self.status.emit("Combining notes…")
                self._generate(model, tokenizer,
                               _build_prompt(tokenizer, self.model_name,
                                             self._system_text(COMBINE_PROMPT),
                                             self.JOINER.join(notes)),
                               stream=True)
                return
            notes = merged

    def run(self) -> None:
        from transcriber import begin_download_reporting, end_download_reporting

        begin_download_reporting(self.download_progress.emit)
        try:
            # The ASR model and the notes LLM must never coexist in memory.
            from transcriber import free_asr_models

            free_asr_models()
            model, tokenizer = _get_llm(self.model_name, self.status.emit)

            header = "## AI Notes — " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            self.chunk.emit(header + "\n\n")

            context = context_window(model, tokenizer)
            if len(tokenizer.encode(self.transcript)) <= input_budget(
                    tokenizer, context, self._system_text()):
                parts = [self.transcript]  # one pass, full prompt, tail sections and all
            else:
                parts = split_text(
                    self.transcript, tokenizer,
                    input_budget(tokenizer, context, self._system_text(PART_PROMPT)))
            self._tokens = 0
            self._pass_base = 0
            self._passes_done = 0
            self._last_tps = 0.0
            self._passes_total = 1 if len(parts) == 1 else len(parts) + 1

            if len(parts) == 1:
                self.status.emit("Writing notes…")
                self._generate(model, tokenizer,
                               _build_prompt(tokenizer, self.model_name,
                                             self._system_text(), parts[0]),
                               stream=True)
            else:
                part_text = self._system_text(PART_PROMPT)
                part_notes = []
                for n, part in enumerate(parts, 1):
                    if self._stop:
                        break
                    self.status.emit(f"Writing notes for part {n}/{len(parts)}…")
                    part_notes.append(self._generate(
                        model, tokenizer,
                        _build_prompt(tokenizer, self.model_name, part_text, part),
                        stream=False))
                if not self._stop:
                    self._combine(model, tokenizer, part_notes, context)

            if self._stop:
                self.chunk.emit("\n\n*(cancelled)*")
            else:
                self.progress.emit(100, self._tokens, self._last_tps)
            self.chunk.emit("\n")
            self.completed.emit()
        except Exception as exc:
            diagnostics.log_exception("AI notes", exc, model=self.model_name,
                                      transcript_chars=len(self.transcript))
            self.failed.emit(f"{exc}\n\nDetails: {diagnostics.log_path()}")
        finally:
            end_download_reporting(self.download_progress.emit)
