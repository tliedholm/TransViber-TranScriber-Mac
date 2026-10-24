"""Frozen-bundle self-test.

Built as a console PyInstaller app from the SAME analysis as the real app, so
it proves the shipped bundle can actually load and run every engine — in
particular that excluding torch is safe and that mlx.metallib, numba (word
timestamps), tiktoken, onnxruntime and transformers/nagisa all survived
freezing.

Two modes, both used as release gates by packaging/build_app.sh:

  selftest --imports-only   every engine module and shipped resource loads.
                            No audio, no downloads, runs anywhere.
  selftest <audio.wav>      the above plus a real transcription per engine.
                            Needs a Metal GPU and the models in the HF cache.
"""

import multiprocessing
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QCoreApplication

multiprocessing.freeze_support()  # mirrors main.py; without it children re-run this script

import transcriber
from models import Track

IMPORTS_ONLY = "--imports-only" in sys.argv[1:]
_positional = [a for a in sys.argv[1:] if not a.startswith("-")]
AUDIO = Path(_positional[0]) if _positional else None

results = []


def check_bundle() -> list[str]:
    """Load every engine module and resource the frozen app depends on."""
    problems = []
    modules = [
        "av", "numpy", "psutil", "faster_whisper", "faster_whisper.vad",
        "ctranslate2", "mlx.core", "mlx_whisper", "mlx_qwen3_asr", "mlx_lm",
        "onnx_asr", "onnxruntime", "transformers", "huggingface_hub",
        "tqdm.std", "certifi", "tiktoken", "numba",
        "main_window", "llm", "stats", "audio", "merger", "exporters",
        "diagnostics",
    ]
    for name in modules:
        try:
            __import__(name)
        except Exception as exc:
            problems.append(f"import {name}: {type(exc).__name__}: {exc}")

    # mlx.metallib is a data file, so freezing can drop it while mlx itself
    # still imports — the failure would then only show up on a user's GPU.
    try:
        import mlx

        roots = [Path(p) for p in mlx.__path__]
        if not [f for root in roots for f in root.rglob("*.metallib")]:
            problems.append("mlx.metallib missing from the bundle — no Metal kernels")
    except Exception as exc:
        problems.append(f"mlx.metallib check: {exc}")
    try:
        import certifi

        if not Path(certifi.where()).is_file():
            problems.append("certifi CA bundle missing — model downloads would fail")
    except Exception as exc:
        problems.append(f"certifi check: {exc}")
    return problems


def run_engine(label: str, model_name: str, device: str) -> None:
    """Transcribe one file and report what actually happened."""
    segments = []
    status = []
    failures = []
    notes = []

    worker = transcriber.TranscribeWorker(
        tracks=[Track(path=AUDIO, speaker="Alice")],
        indices=[0],
        model_name=model_name,
        device=device,
        language="en",
        hotwords="",
    )
    worker.segment_ready.connect(lambda i, s: segments.append(s))
    worker.model_status.connect(status.append)
    worker.track_failed.connect(lambda i, m: failures.append(m))
    worker.track_note.connect(lambda i, m: notes.append(m))  # decoding's timing notes

    expected = transcriber.resolve_backend(model_name, device)
    t0 = time.time()
    try:
        worker.run()  # run synchronously in this thread, no event loop needed
    except Exception:
        failures.append(traceback.format_exc())
    elapsed = time.time() - t0

    # Compare the backend that actually ran against the one this (model,
    # device) pair should resolve to. Checking worker.device instead used to
    # pass an onnx run "on the GPU" that was always the CPU EP, and to miss a
    # Whisper run that quietly landed on faster-whisper.
    actual = worker.backend
    wrong_backend = actual is not None and actual != expected
    text = " ".join(s.text.strip() for s in segments)
    ok = bool(text) and not failures and not wrong_backend

    results.append((label, ok, elapsed, text, failures, expected, actual, status, notes))
    transcriber.free_asr_models()


print(f"frozen={getattr(sys, 'frozen', False)}  audio={AUDIO}")
print("torch importable? ", end="")
try:
    import torch  # noqa: F401
    print("YES — it was NOT excluded (bundle is bigger than it needs to be)")
except ImportError:
    print("no (excluded as intended)")
print(f"mlx_available={transcriber.mlx_available()}")
print("-" * 70)

bundle_problems = check_bundle()
if bundle_problems:
    print("BUNDLE CHECK FAILED:")
    for problem in bundle_problems:
        print(f"  !! {problem}")
else:
    print("bundle check OK — every engine module and resource loaded")

if IMPORTS_ONLY or AUDIO is None:
    if AUDIO is None and not IMPORTS_ONLY:
        print("\nNo audio file given — skipping the live engine tests.")
        print("Run:  selftest /path/to/sample.wav")
    print("=" * 70)
    print("BUNDLE OK" if not bundle_problems else "BUNDLE INCOMPLETE")
    sys.exit(1 if bundle_problems else 0)

print("-" * 70)
QCoreApplication(sys.argv)

# One per engine path in transcriber.py, using models already in the HF cache.
run_engine("faster-whisper (CPU/CTranslate2)", "tiny", "cpu")
run_engine("mlx-whisper (Metal GPU)", "tiny", "mlx")
run_engine("onnx-asr parakeet (CPU EP, by design)", "parakeet-tdt-0.6b-v3", "mlx")
run_engine("mlx-qwen3-asr + aligner (Metal GPU)", "qwen3-asr-0.6b", "mlx")

print("-" * 70)
print("Notes LLM (mlx-lm): ", end="", flush=True)
notes_ok = False
try:
    import llm

    from mlx_lm import load
    model, tok = load(llm.LLM_REPOS["qwen3-4b"])
    notes_ok = model is not None and tok is not None
    context = llm.context_window(model, tok)
    # Sizes every pass through the real chat template, as a notes run does.
    room = min(llm.NotesWorker("", "", "qwen3-4b")._budgets(tok, context))
    print(f"loaded qwen3-4b OK (context {context}, {room:,} tokens of "
          f"transcript per pass)")
    del model, tok
except Exception:
    print("FAILED")
    traceback.print_exc()

print("=" * 70)
all_ok = notes_ok and not bundle_problems
for label, ok, elapsed, text, failures, expected, actual, status, notes in results:
    all_ok = all_ok and ok
    print(f"[{'PASS' if ok else 'FAIL'}] {label}  ({elapsed:.1f}s)")
    print(f"       backend: {actual.label if actual else 'none'}")
    print(f"       text: {text[:110]!r}")
    if actual is not None and actual != expected:
        print(f"       !! expected {expected.label}, got {actual.label}")
    for f in failures:
        print(f"       !! {f.splitlines()[-1] if f else f}")
    for n in notes:
        print(f"       note: {n}")
    for s in status:
        if "fail" in s.lower() or "error" in s.lower():
            print(f"       !! status: {s}")
print(f"[{'PASS' if notes_ok else 'FAIL'}] Notes LLM (mlx-lm)")
print(f"[{'PASS' if not bundle_problems else 'FAIL'}] Bundle contents")
print("=" * 70)
print("ALL ENGINES OK" if all_ok else "SOME ENGINES FAILED")
sys.exit(0 if all_ok else 1)
