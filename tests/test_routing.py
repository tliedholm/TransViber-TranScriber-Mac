"""Model routing and backend resolution.

These decide which engine runs and what the user is told ran, and getting
either wrong is invisible until a transcript comes out wrong.
"""

import pytest

import transcriber
from transcriber import Backend, engine_for, fallback_backend, resolve_backend


@pytest.mark.parametrize("name, engine", [
    ("large-v3-turbo", "whisper"),
    ("distil-large-v3.5-ct2", "whisper"),
    ("tiny.en", "whisper"),
    ("parakeet-tdt-0.6b-v3", "onnx"),
    ("parakeet", "onnx"),
    ("canary-1b-v2", "onnx"),
    ("nemo-parakeet-tdt-0.6b-v2", "onnx"),
    ("qwen3-asr-1.7b", "qwen"),
    ("qwen3-asr", "qwen"),
])
def test_engine_routing(name, engine):
    assert engine_for(name) == engine


def test_routing_is_case_insensitive():
    assert engine_for("Parakeet-TDT-0.6B-v3") == "onnx"
    assert engine_for("QWEN3-ASR-1.7B") == "qwen"


def test_misspelled_parakeet_still_routes_to_onnx():
    # onnx-asr's "unknown model" error beats faster-whisper hunting for a repo.
    assert engine_for("parakeet-tdt-0.6b-v9") == "onnx"


def test_qwen_ignores_the_device_setting(monkeypatch):
    monkeypatch.setattr(transcriber, "mlx_available", lambda: True)
    for device in ("mlx", "cpu", "cuda"):
        assert resolve_backend("qwen3-asr-1.7b", device) == Backend("qwen", "metal")


def test_onnx_is_the_cpu_ep_on_mac(monkeypatch):
    monkeypatch.setattr(transcriber, "mlx_available", lambda: True)
    # Asking for the GPU does not give onnx-asr one on Apple silicon.
    assert resolve_backend("parakeet-tdt-0.6b-v3", "mlx") == Backend("onnx", "cpu")
    assert resolve_backend("parakeet-tdt-0.6b-v3", "cuda") == Backend("onnx", "cuda")


def test_whisper_falls_to_ct2_without_mlx(monkeypatch):
    monkeypatch.setattr(transcriber, "mlx_available", lambda: False)
    assert resolve_backend("large-v3", "mlx") == Backend("whisper-ct2", "cpu")
    monkeypatch.setattr(transcriber, "mlx_available", lambda: True)
    assert resolve_backend("large-v3", "mlx") == Backend("whisper-mlx", "metal")


def test_fallback_only_when_a_real_alternative_exists():
    # mlx-whisper -> faster-whisper is a genuinely different runtime.
    assert fallback_backend(Backend("whisper-mlx", "metal")) == Backend("whisper-ct2", "cpu")
    # Qwen is MLX-only and onnx on a Mac is already the CPU EP: no fallback,
    # so the worker reports the failure instead of claiming a retry on CPU.
    assert fallback_backend(Backend("qwen", "metal")) is None
    assert fallback_backend(Backend("onnx", "cpu")) is None
    assert fallback_backend(Backend("whisper-ct2", "cpu")) is None
    # CUDA paths do have a CPU sibling.
    assert fallback_backend(Backend("onnx", "cuda")) == Backend("onnx", "cpu")
    assert fallback_backend(Backend("whisper-ct2", "cuda")) == Backend("whisper-ct2", "cpu")


def test_fallback_chain_terminates():
    backend = Backend("whisper-mlx", "metal")
    seen = set()
    while backend is not None:
        assert backend not in seen, "fallback loop"
        seen.add(backend)
        backend = fallback_backend(backend)


def test_backend_labels_are_human_readable():
    assert Backend("whisper-mlx", "metal").label == "mlx-whisper on Metal GPU"
    assert Backend("onnx", "cpu").label == "onnx-asr on CPU"
