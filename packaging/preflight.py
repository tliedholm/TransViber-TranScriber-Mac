"""Pre-build dependency check, run by build_app.sh before PyInstaller.

Catches a broken environment in seconds instead of at the end of a ten-minute
build — or, worse, after a signed bundle has already been handed to someone.
Every engine the app offers must import here, and the resources that are easy
to lose when freezing (mlx's Metal kernels, certifi's CA bundle) must exist.

Run standalone too:  .venv/bin/python packaging/preflight.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# (import name, what breaks without it)
REQUIRED_IMPORTS = [
    ("PySide6.QtWidgets", "the entire UI"),
    ("av", "all audio decoding (no ffmpeg CLI is shipped)"),
    ("numpy", "audio buffers"),
    ("psutil", "the CPU/RAM readout"),
    ("faster_whisper", "the CPU Whisper engine"),
    ("faster_whisper.vad", "Silero VAD, which Qwen needs for sparse tracks"),
    ("ctranslate2", "faster-whisper's runtime"),
    ("mlx.core", "every Metal path"),
    ("mlx_whisper", "Whisper on the GPU"),
    ("mlx_qwen3_asr", "Qwen3-ASR"),
    ("mlx_lm", "AI Notes"),
    ("onnx_asr", "Parakeet and Canary"),
    ("onnxruntime", "onnx-asr's runtime"),
    ("transformers", "tokenizers for several engines"),
    ("huggingface_hub", "model downloads"),
    ("tqdm.std", "the download-progress hook"),
    ("certifi", "TLS verification for model downloads"),
]

APP_DIR = Path(__file__).resolve().parent.parent / "app"
# The app's own modules, imported the way the frozen bundle does (flat).
APP_MODULES = sorted(p.stem for p in APP_DIR.glob("*.py"))


def _check_imports(names) -> list[str]:
    problems = []
    for entry in names:
        name, why = entry if isinstance(entry, tuple) else (entry, "")
        try:
            importlib.import_module(name)
        except Exception as exc:
            detail = f" — needed for {why}" if why else ""
            problems.append(f"{name}: {type(exc).__name__}: {exc}{detail}")
    return problems


def _check_resources() -> list[str]:
    problems = []
    try:
        import mlx.core  # noqa: F401

        # mlx is a namespace package, so it has __path__ but no __file__.
        roots = [Path(p) for p in sys.modules["mlx"].__path__]
        metallib = [f for root in roots for f in root.rglob("*.metallib")]
        if not metallib:
            problems.append(
                "mlx ships no .metallib — the GPU kernels are missing and "
                "every Metal engine would fail at runtime.")
    except Exception as exc:
        problems.append(f"mlx resource check failed: {exc}")
    try:
        import certifi

        if not Path(certifi.where()).is_file():
            problems.append("certifi's CA bundle is missing; downloads would fail.")
    except Exception as exc:
        problems.append(f"certifi check failed: {exc}")
    return problems


def _check_notes_models() -> list[str]:
    """Every notes model offered in the UI must map to a repo."""
    import llm

    return [f"notes model {model!r} has no repo mapping"
            for _tier, models in llm.NOTES_MODEL_GROUPS
            for model in models if model not in llm.LLM_REPOS]


def main() -> int:
    sys.path.insert(0, str(APP_DIR))
    problems = _check_imports(REQUIRED_IMPORTS)
    problems += _check_imports(APP_MODULES)
    if not problems:  # only meaningful once the modules import
        problems += _check_resources()
        problems += _check_notes_models()

    if problems:
        print("PREFLIGHT FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nFix the environment (bash setup_mac.command) and build again.")
        return 1
    print(f"preflight OK — {len(REQUIRED_IMPORTS)} dependencies, "
          f"{len(APP_MODULES)} app modules, resources and notes models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
