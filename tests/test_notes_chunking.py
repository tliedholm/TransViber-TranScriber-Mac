"""Transcript splitting and combine batching for AI Notes.

A D&D session is long and often one speaker at a time, so these bounds are
what stop a notes run from silently overflowing the model's context window.
"""

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


def test_budget_leaves_room_for_the_prompt_and_the_output():
    tok = WordTokenizer()
    budget = llm.input_budget(tok, 32_768, words(300, "p"))
    assert budget == 32_768 - 300 - llm.MAX_NOTES_TOKENS - llm.PROMPT_MARGIN_TOKENS
    # Even an absurd prompt cannot drive the budget negative.
    assert llm.input_budget(tok, 4_096, words(10_000, "p")) >= 1_000


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
