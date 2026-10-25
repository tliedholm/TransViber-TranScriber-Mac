"""The notes model catalog and the per-family handling it relies on.

No model is loaded: mlx-lm is replaced by stubs, so these check what the app
asks mlx-lm for (template switches, sampling, pass size) and what it does with
the text that comes back (reasoning kept out of the notes).
"""

import sys
import types

import pytest

import llm


# ----- catalog -----

def test_every_listed_model_has_a_repo_and_the_default_is_listed():
    listed = [name for _, names in llm.NOTES_MODEL_GROUPS for name in names]
    assert len(listed) == len(set(listed))
    assert set(listed) == set(llm.LLM_REPOS)
    assert llm.DEFAULT_NOTES_MODEL in listed


def test_each_tier_starts_with_its_newest_pick():
    firsts = {label: names[0] for label, names in llm.NOTES_MODEL_GROUPS}
    assert firsts["Max Detail (Large Tier)"] == "qwen3.5-9b"
    assert firsts["Quick Review (Small Tier)"] == "qwen3.5-4b"
    assert "gemma-4-12b" in dict(llm.NOTES_MODEL_GROUPS)["Max Detail (Large Tier)"]
    assert dict(llm.NOTES_MODEL_GROUPS)["Experimental (24 GB+ Mac)"] == ["gpt-oss-20b"]


def test_older_models_stay_loadable_for_saved_settings():
    # A saved "qwen3-14b" must still map to its repo, not be tried as one.
    for name in ("qwen3-14b", "qwen3-4b", "gemma-3-12b", "llama-3.1-8b"):
        assert llm.LLM_REPOS[name].startswith("mlx-community/")


# ----- sampling and pass size -----

@pytest.mark.parametrize("name, expected", [
    ("qwen3.5-9b", {"temp": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}),
    ("mlx-community/Qwen3.5-4B-MLX-4bit",
     {"temp": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}),
    ("qwen3-14b", {"temp": 0.7, "top_p": 0.8, "top_k": 20}),
    ("gemma-4-12b", {"temp": 1.0, "top_p": 0.95, "top_k": 64}),
    ("lmstudio-community/gemma-4-12B-it-MLX-4bit", {"temp": 1.0, "top_p": 0.95, "top_k": 64}),
    ("someone/gemma4-e4b-it", {"temp": 1.0, "top_p": 0.95, "top_k": 64}),
    ("gpt-oss-20b", {"temp": 1.0, "top_p": 1.0}),
    # Families without a recommendation keep greedy decoding.
    ("llama-3.1-8b", {}),
    ("gemma-3-12b", {}),
])
def test_sampling_follows_the_model_card(name, expected):
    assert llm.sampling(name) == expected


def test_sampling_hands_out_a_copy():
    llm.sampling("qwen3.5-9b").pop("presence_penalty")
    assert "presence_penalty" in llm.sampling("qwen3.5-9b")


def test_pass_limits():
    assert llm.pass_limit("qwen3.5-9b") == llm.MAX_PASS_TOKENS
    assert llm.pass_limit("llama-3.1-8b") == llm.MAX_PASS_TOKENS
    assert llm.pass_limit("gpt-oss-20b") == 32_768
    assert llm.pass_limit("mlx-community/gpt-oss-20b-MXFP4-Q4") == 32_768


def test_context_window_reads_a_multimodal_checkpoints_text_model():
    # Qwen3.5 and Gemma 4 load as a wrapper whose own args hold only
    # text_config; the text model's args carry the window.
    model = types.SimpleNamespace(
        args=types.SimpleNamespace(text_config={}),
        language_model=types.SimpleNamespace(
            args=types.SimpleNamespace(max_position_embeddings=262_144)))
    assert llm.context_window(model, object()) == 262_144


# ----- models too big for this Mac's GPU -----

M1_PRO_16GB = 12_713_115_648  # what MLX lets the GPU use on Trenton's Mac
MAC_24GB = 17_179_869_184     # the least a 24 GB Mac allows


@pytest.mark.parametrize("name", ["gpt-oss-20b", " gpt-oss-20b ",
                                  "mlx-community/gpt-oss-20b-MXFP4-Q4"])
def test_gpt_oss_is_refused_on_a_16_gb_mac(name):
    reason = llm.gpu_shortfall(name, M1_PRO_16GB)
    assert "24 GB" in reason and "12.7 GB" in reason
    assert "qwen3.5-9b" in reason


@pytest.mark.parametrize("name, budget", [
    ("gpt-oss-20b", MAC_24GB),
    ("gpt-oss-20b", None),           # no MLX to ask: never refused
    ("qwen3.5-9b", M1_PRO_16GB),
    ("gemma-4-12b", M1_PRO_16GB),
    ("someone/unknown-model", M1_PRO_16GB),
])
def test_everything_else_is_allowed(name, budget):
    assert llm.gpu_shortfall(name, budget) is None


def test_no_mlx_means_no_budget(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert llm.gpu_budget() is None


def test_a_refused_model_fails_before_anything_loads(monkeypatch):
    monkeypatch.setattr(llm, "gpu_budget", lambda: M1_PRO_16GB)

    def must_not_load(*args):
        raise AssertionError("loaded a model that cannot fit")

    monkeypatch.setattr(llm, "_get_llm", must_not_load)
    monkeypatch.setattr(llm, "free_llm", must_not_load)
    worker = llm.NotesWorker("Alice: hello", "", "gpt-oss-20b")
    failures, chunks = [], []
    worker.failed.connect(failures.append)
    worker.chunk.connect(chunks.append)
    worker.run()
    assert failures == [llm.gpu_shortfall("gpt-oss-20b", M1_PRO_16GB)]
    assert chunks == []  # not even the notes header


METAL_OOM = ("[METAL] Command buffer execution failed: Insufficient Memory "
             "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)")


@pytest.fixture
def failing_load(monkeypatch):
    """Run a worker whose model load raises `error`; returns the failure text."""
    stub = types.ModuleType("transcriber")
    stub.begin_download_reporting = stub.end_download_reporting = lambda callback: None
    stub.free_asr_models = lambda: None
    monkeypatch.setitem(sys.modules, "transcriber", stub)
    monkeypatch.setattr(llm.diagnostics, "log_exception", lambda *a, **k: None)
    monkeypatch.setattr(llm, "gpu_budget", lambda: M1_PRO_16GB)
    freed = []
    monkeypatch.setattr(llm, "free_llm", lambda: freed.append(True))

    def run(error):
        def load(name, status):
            raise error
        monkeypatch.setattr(llm, "_get_llm", load)
        worker = llm.NotesWorker("Alice: hello", "", "qwen3.5-9b")
        failures = []
        worker.failed.connect(failures.append)
        worker.run()
        return failures[0], bool(freed)

    return run


def test_running_out_of_gpu_memory_says_so_and_frees_the_model(failing_load):
    message, freed = failing_load(RuntimeError(METAL_OOM))
    assert message.startswith("qwen3.5-9b ran out of GPU memory (12.7 GB on this Mac).")
    assert METAL_OOM in message  # the original error is kept
    assert freed


def test_other_failures_keep_the_model_loaded(failing_load):
    message, freed = failing_load(RuntimeError("instructions too long"))
    assert message.startswith("instructions too long")
    assert not freed


# ----- chat template switches -----

class CapturingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        self.calls.append((messages, kwargs))
        return "prompt"


@pytest.mark.parametrize("name, kwargs, system_role", [
    ("qwen3.5-9b", {"enable_thinking": False}, True),
    ("qwen3-4b", {"enable_thinking": False}, True),
    ("gemma-4-12b", {"enable_thinking": False}, True),
    ("gemma-3-12b", {}, False),
    ("gpt-oss-20b", {"reasoning_effort": "low"}, True),
    ("llama-3.1-8b", {}, True),
])
def test_prompt_switches_per_family(name, kwargs, system_role):
    tok = CapturingTokenizer()
    llm._build_prompt(tok, name, "SYSTEM", "TRANSCRIPT")
    messages, sent = tok.calls[0]
    assert sent == kwargs
    assert (messages[0]["role"] == "system") == system_role
    assert "SYSTEM" in messages[0]["content"]
    assert "TRANSCRIPT" in messages[-1]["content"]


# ----- reasoning kept out of the notes -----

GEMMA = "<|channel>thought\n<channel|>## Summary\nThe party rested."
HARMONY = ("<|channel|>analysis<|message|>We need notes. Keep it short."
           "<|end|><|start|>assistant<|channel|>final<|message|>"
           "## Summary\nThe party rested.")
HARMONY_FINAL_ONLY = "<|channel|>final<|message|>## Summary\nThe party rested."
QWEN = "<think>\nPlan the notes.\n</think>\n\n## Summary\nThe party rested."


def filtered(pieces):
    f = llm.ReasoningFilter()
    out = "".join(f.feed(p) for p in pieces) + f.finish()
    return out, f


@pytest.mark.parametrize("text", [GEMMA, HARMONY, HARMONY_FINAL_ONLY, QWEN,
                                  "\n" + QWEN])
@pytest.mark.parametrize("piece", [1, 3, 1_000])
def test_reasoning_is_dropped_however_the_stream_is_cut(text, piece):
    out, f = filtered([text[i:i + piece] for i in range(0, len(text), piece)])
    assert out == "## Summary\nThe party rested."
    assert not f.unfinished


def test_plain_notes_pass_through_at_once():
    f = llm.ReasoningFilter()
    assert f.feed("## Summary") == "## Summary"
    assert f.feed(" <think> is just text now") == " <think> is just text now"


def test_text_that_only_looked_like_a_marker_is_released_whole():
    assert filtered(["<", "b>bold</b> start"])[0] == "<b>bold</b> start"
    assert filtered(["<|ch"])[0] == "<|ch"  # stream ended undecided


def test_a_block_that_never_ends_hides_everything_and_says_so():
    out, f = filtered(["<|channel|>analysis<|message|>", "thinking and thinking"])
    assert out == ""
    assert f.unfinished


# ----- one generation pass, with mlx-lm stubbed out -----

@pytest.fixture
def fake_mlx_lm(monkeypatch):
    """mlx_lm stand-in: stream_generate replays `calls["reply"]` in pieces."""
    calls = {"reply": "", "stream_kwargs": None, "sampler": None, "processors": None}

    def stream_generate(model, tokenizer, prompt, **kwargs):
        calls["stream_kwargs"] = kwargs
        for piece in calls["reply"]:
            yield types.SimpleNamespace(text=piece, generation_tps=20.0)

    def make_sampler(**kwargs):
        calls["sampler"] = kwargs
        return "sampler"

    def make_logits_processors(**kwargs):
        calls["processors"] = kwargs
        return ["processors"]

    package = types.ModuleType("mlx_lm")
    package.stream_generate = stream_generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = make_sampler
    sample_utils.make_logits_processors = make_logits_processors
    package.sample_utils = sample_utils
    monkeypatch.setitem(sys.modules, "mlx_lm", package)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    return calls


def one_pass(model_name):
    worker = llm.NotesWorker("", "", model_name)
    chunks = []
    worker.chunk.connect(chunks.append)
    return worker, chunks


def test_a_pass_samples_as_the_card_says(fake_mlx_lm):
    fake_mlx_lm["reply"] = ["## Notes"]
    worker, _ = one_pass("qwen3.5-9b")
    worker._generate(None, None, [1, 2, 3], stream=False)
    assert fake_mlx_lm["sampler"] == {"temp": 0.7, "top_p": 0.8, "top_k": 20}
    assert fake_mlx_lm["processors"] == {"presence_penalty": 1.5}
    assert fake_mlx_lm["stream_kwargs"] == {"max_tokens": llm.MAX_NOTES_TOKENS,
                                            "sampler": "sampler",
                                            "logits_processors": ["processors"]}


def test_an_unlisted_family_still_decodes_greedily(fake_mlx_lm):
    fake_mlx_lm["reply"] = ["## Notes"]
    worker, _ = one_pass("llama-3.1-8b")
    worker._generate(None, None, [1], stream=False)
    assert fake_mlx_lm["sampler"] == {}  # make_sampler()'s default is greedy
    assert fake_mlx_lm["processors"] == {"presence_penalty": None}


@pytest.mark.parametrize("stream", [True, False])
def test_reasoning_reaches_neither_the_editor_nor_the_combine_pass(fake_mlx_lm, stream):
    fake_mlx_lm["reply"] = [HARMONY[i:i + 4] for i in range(0, len(HARMONY), 4)]
    worker, chunks = one_pass("gpt-oss-20b")
    notes = worker._generate(None, None, [1], stream=stream)
    assert notes == "## Summary\nThe party rested."
    assert "".join(chunks) == (notes if stream else "")
    # Hidden tokens still count towards progress.
    assert worker._tokens == len(fake_mlx_lm["reply"])


def test_a_pass_that_never_stops_reasoning_fails_instead_of_writing_nothing(fake_mlx_lm):
    fake_mlx_lm["reply"] = ["<|channel|>analysis<|message|>", "still going"]
    worker, chunks = one_pass("gpt-oss-20b")
    with pytest.raises(RuntimeError, match="stopped before it finished reasoning"):
        worker._generate(None, None, [1], stream=True)
    assert chunks == []


def test_a_cancel_while_reasoning_is_not_a_failure(fake_mlx_lm):
    fake_mlx_lm["reply"] = ["<think>", "plan", "more plan"]
    worker, _ = one_pass("qwen3.5-9b")
    worker.progress.connect(lambda *_: worker.stop())  # cancel after one token
    assert worker._generate(None, None, [1], stream=True) == ""
    assert worker._tokens == 1
