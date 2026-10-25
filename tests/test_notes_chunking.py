"""Transcript splitting and combine batching for AI Notes.

A D&D session is long and often one speaker at a time, so these bounds are
what stop a notes run from silently overflowing the model's context window.
"""

import sys
import types

import pytest

import llm


class WordTokenizer:
    """Stand-in tokenizer: one token per space-separated word.

    Real tokenizers need a model download; the splitting logic only cares
    that encode/decode round-trip and that counts are consistent.
    """

    def encode(self, text):
        return text.split(" ")

    def decode(self, ids):
        return " ".join(ids)


class ChatTokenizer(WordTokenizer):
    """WordTokenizer with a chat template whose role markers are words too.

    Renders the thinking switch the way Qwen3's template does (an empty think
    block when enable_thinking is False), so a count taken without the
    request's own template policy comes out different.
    """

    bos_token = "<s>"

    def __init__(self, bos_in_template=True):
        self.bos_in_template = bos_in_template

    def encode(self, text, add_special_tokens=True):
        ids = text.split(" ")
        return [self.bos_token] + ids if add_special_tokens else ids

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        assert not tokenize
        rendered = " ".join(f"<|{m['role']}|> {m['content']} <|end|>" for m in messages)
        if kwargs.get("enable_thinking") is False:
            rendered += " <think> </think>"
        if add_generation_prompt:
            rendered += " <|assistant|>"
        return f"{self.bos_token} {rendered}" if self.bos_in_template else rendered


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_is_one_part():
    tok = WordTokenizer()
    assert llm.split_text(words(50), tok, 1_000) == [words(50)]


def test_a_single_oversized_line_is_split():
    # The audit's case: coalesce() can emit one 70k-token line for a
    # monologue, and line-boundary splitting alone never divided it.
    tok = WordTokenizer()
    parts = llm.split_text(words(70_000), tok, 5_000)
    assert len(parts) == 14
    assert all(len(tok.encode(p)) <= 5_000 for p in parts)


def test_split_respects_the_limit_with_mixed_line_lengths():
    tok = WordTokenizer()
    text = "\n".join([words(10), words(40_000), words(10), words(3_000)])
    parts = llm.split_text(text, tok, 5_000)
    assert all(len(tok.encode(p)) <= 5_000 for p in parts)
    assert len(parts) > 1


def test_split_loses_no_words():
    tok = WordTokenizer()
    text = "\n".join(words(30_000, "a") for _ in range(2))
    joined = " ".join(llm.split_text(text, tok, 4_000))
    assert joined.count("a0") == 2
    assert len(joined.split()) == len(text.split())


def test_budget_leaves_room_for_the_whole_request_and_the_output():
    tok = ChatTokenizer()
    system = words(300, "p")
    request = len(llm._request(tok, "llama-3.1-8b", system, ""))
    # The template, role headers and generation prompt are charged, not just
    # the system text.
    assert request > 300
    budget = llm.input_budget(tok, "llama-3.1-8b", 32_768, system)
    assert budget == (32_768 - llm.MAX_NOTES_TOKENS - llm.PROMPT_MARGIN_TOKENS
                      - request)


def test_the_template_policy_is_in_the_count():
    # Qwen3 runs with enable_thinking=False, which its template renders into
    # the prompt; the count goes through the same builder, so it sees that.
    tok = ChatTokenizer()
    qwen = llm.input_budget(tok, "qwen3-4b", 32_768, "notes")
    plain = llm.input_budget(tok, "llama-3.1-8b", 32_768, "notes")
    assert plain - qwen == 2


@pytest.mark.parametrize("bos_in_template", [True, False])
def test_the_request_is_tokenized_as_stream_generate_would(bos_in_template):
    # mlx-lm adds special tokens to a string prompt unless it already starts
    # with BOS; the ids handed to stream_generate must follow the same rule.
    tok = ChatTokenizer(bos_in_template)
    ids = llm._request(tok, "llama-3.1-8b", "notes", "hello there")
    assert ids[0] == "<s>"
    assert ids.count("<s>") == 1


def test_instructions_that_leave_no_room_are_refused():
    # Once a floor that granted room the window didn't have.
    worker = llm.NotesWorker("", words(40_000, "i"), "qwen3-4b")
    with pytest.raises(ValueError) as caught:
        worker._budgets(ChatTokenizer(), 32_768)
    message = str(caught.value)
    assert "Shorten the AI instructions" in message
    assert "room for 0 tokens" in message
    assert f"{llm.MIN_PROMPT_TOKENS:,}" in message


@pytest.mark.parametrize("room, refused", [(llm.MIN_PROMPT_TOKENS + 1, False),
                                           (llm.MIN_PROMPT_TOKENS, False),
                                           (llm.MIN_PROMPT_TOKENS - 1, True)])
def test_the_minimum_is_met_exactly_or_refused(room, refused):
    tok = ChatTokenizer()
    worker = llm.NotesWorker("", "Keep track of the loot.", "qwen3-4b")
    # Budgets move one for one with the window, so pick the window that
    # leaves exactly `room` in the tightest pass.
    context = 100_000 - (min(worker._budgets(tok, 100_000)) - room)
    if refused:
        with pytest.raises(ValueError):
            worker._budgets(tok, context)
    else:
        assert min(worker._budgets(tok, context)) == room


def test_a_bigger_window_takes_bigger_instructions():
    tok = ChatTokenizer()
    worker = llm.NotesWorker("", words(28_000, "i"), "qwen3-4b")
    with pytest.raises(ValueError):
        worker._budgets(tok, 32_768)
    assert min(worker._budgets(tok, 65_536)) >= llm.MIN_PROMPT_TOKENS


def test_combine_batches_stay_inside_the_limit():
    tok = WordTokenizer()
    # 12 part summaries at the generation cap: one unlimited final prompt
    # would be 72k tokens.
    notes = [words(llm.MAX_NOTES_TOKENS) for _ in range(12)]
    batches = llm.batch_by_tokens(notes, tok, 20_000, llm.NotesWorker.JOINER)
    assert len(batches) > 1
    for batch in batches:
        assert len(tok.encode(llm.NotesWorker.JOINER.join(batch))) <= 20_000
    assert sum(len(b) for b in batches) == 12


def test_an_oversized_summary_is_split_before_batching():
    # Batching alone cannot help when one summary is bigger than a whole
    # pass; _combine pre-splits so every batch provably fits.
    tok = WordTokenizer()
    notes = [words(50_000)]
    pieces = [p for n in notes for p in llm.split_text(n, tok, 10_000)]
    batches = llm.batch_by_tokens(pieces, tok, 10_000, llm.NotesWorker.JOINER)
    for batch in batches:
        assert len(tok.encode(llm.NotesWorker.JOINER.join(batch))) <= 10_000


def test_batching_keeps_order():
    tok = WordTokenizer()
    notes = [f"note{i} " + words(1_000) for i in range(6)]
    flat = [n for batch in llm.batch_by_tokens(notes, tok, 2_500, "\n") for n in batch]
    assert flat == notes


def test_context_window_falls_back_when_unknown():
    class Bare:
        pass

    assert llm.context_window(Bare(), Bare()) == llm.DEFAULT_CONTEXT_TOKENS


def test_context_window_reads_the_model_and_rejects_nonsense():
    class Args:
        max_position_embeddings = 40_960

    class Model:
        args = Args()

    class Silly:
        max_position_embeddings = 1_000_000_000

    class SillyModel:
        args = Silly()

    assert llm.context_window(Model(), object()) == 40_960
    # transformers uses a sentinel of 1e30 for "unknown"; never trust it.
    assert llm.context_window(SillyModel(), object()) == llm.DEFAULT_CONTEXT_TOKENS


def test_only_the_final_pass_asks_for_the_tail_sections():
    # A middle part's ending is not the session's ending.
    for section in ("Where we left off", "Unresolved threads"):
        assert section in llm.DEFAULT_PROMPT
        assert section in llm.COMBINE_PROMPT
        assert section not in llm.PART_PROMPT
        assert section not in llm.MERGE_PROMPT


# ----- whole runs, with the model load and generation stubbed out -----

CONTEXT = 16_000


class FakeModel:
    def __init__(self, context):
        self.args = types.SimpleNamespace(max_position_embeddings=context)


class RecordingWorker(llm.NotesWorker):
    """Runs the real run(); _generate records the ids it would send."""

    def __init__(self, *args, note_words=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.note_words = note_words
        self.prompts = []
        self.chunks = []
        self.failures = []
        self.chunk.connect(self.chunks.append)
        self.failed.connect(self.failures.append)

    def _generate(self, model, tokenizer, prompt, stream):
        self.prompts.append(prompt)
        return words(self.note_words, "n")


@pytest.fixture
def offline_run(monkeypatch):
    stub = types.ModuleType("transcriber")
    stub.begin_download_reporting = stub.end_download_reporting = lambda callback: None
    stub.free_asr_models = lambda: None
    monkeypatch.setitem(sys.modules, "transcriber", stub)
    monkeypatch.setattr(llm.diagnostics, "log_exception", lambda *a, **k: None)

    def run(worker, context=CONTEXT):
        tok = ChatTokenizer()
        monkeypatch.setattr(llm, "_get_llm",
                            lambda name, status: (FakeModel(context), tok))
        worker.run()
        return tok

    return run


def test_a_short_transcript_sends_exactly_the_ids_it_counted(offline_run):
    worker = RecordingWorker(words(500), "", "qwen3-4b")
    tok = offline_run(worker)
    assert not worker.failures
    assert worker.prompts == [llm._request(tok, "qwen3-4b", worker._system_text(),
                                           worker.transcript)]
    assert worker.chunks[0].startswith("## AI Notes")


def test_the_direct_prompt_is_counted_whole(offline_run):
    tok = ChatTokenizer()
    system = llm.NotesWorker("", "", "qwen3-4b")._system_text()
    limit = llm.prompt_limit(CONTEXT)
    fits = limit - len(llm._request(tok, "qwen3-4b", system, "")) + 1
    assert len(llm._request(tok, "qwen3-4b", system, words(fits))) == limit
    exact = RecordingWorker(words(fits), "", "qwen3-4b")
    offline_run(exact)
    assert len(exact.prompts) == 1
    over = RecordingWorker(words(fits + 1), "", "qwen3-4b")
    offline_run(over)
    assert len(over.prompts) > 1  # parts, then the combining pass


@pytest.mark.parametrize("note_words", [50, 4_000])
def test_every_pass_of_a_long_run_fits_the_window(offline_run, note_words):
    # 4,000-word notes don't all fit one combining prompt, so that case goes
    # through merge rounds as well.
    transcript = "\n".join(words(40) for _ in range(2_000))
    worker = RecordingWorker(transcript, "Track every NPC by name.", "qwen3-4b",
                             note_words=note_words)
    offline_run(worker)
    assert not worker.failures
    assert len(worker.prompts) > 2
    assert all(len(p) <= llm.prompt_limit(CONTEXT) for p in worker.prompts)
    rendered = [" ".join(p) for p in worker.prompts]
    merges = [r for r in rendered if "covering this stretch of the session" in r]
    assert bool(merges) == (note_words > 50)
    assert "Unresolved threads" in rendered[-1]  # the final pass


def test_oversized_instructions_never_reach_generation(offline_run):
    worker = RecordingWorker(words(500), words(20_000, "i"), "qwen3-4b")
    offline_run(worker)
    assert worker.prompts == []
    assert worker.chunks == []  # not even the header
    assert "Shorten the AI instructions" in worker.failures[0]


def test_a_huge_window_is_cut_to_the_pass_limit(offline_run):
    # Qwen3.5 claims 262,144 tokens; one pass still holds MAX_PASS_TOKENS at
    # most, so its KV cache stays within a 16 GB Mac.
    transcript = "\n".join(words(40) for _ in range(2_500))  # 100,000 tokens
    worker = RecordingWorker(transcript, "", "qwen3.5-9b")
    offline_run(worker, context=262_144)
    assert not worker.failures
    assert len(worker.prompts) > 2
    assert all(len(p) <= llm.prompt_limit(llm.MAX_PASS_TOKENS) for p in worker.prompts)


def test_a_typical_session_is_one_pass_on_qwen3_5(offline_run):
    transcript = "\n".join(words(40) for _ in range(1_250))  # 50,000 tokens
    worker = RecordingWorker(transcript, "", "qwen3.5-9b")
    offline_run(worker, context=262_144)
    assert len(worker.prompts) == 1


def test_gpt_oss_gets_the_smaller_pass(offline_run):
    transcript = "\n".join(words(40) for _ in range(1_250))
    worker = RecordingWorker(transcript, "", "gpt-oss-20b")
    offline_run(worker, context=131_072)
    assert not worker.failures
    assert len(worker.prompts) > 2
    assert all(len(p) <= llm.prompt_limit(32_768) for p in worker.prompts)
