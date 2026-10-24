"""Shared PyInstaller analysis inputs for TransViber.spec and selftest.spec.

Both specs need the identical set of collected packages — that is the whole
point of the self-test, which only proves anything about the real bundle if it
was frozen from the same analysis. They used to be two hand-maintained copies.

The important behaviour here is that collecting a REQUIRED package is not
allowed to fail quietly. The old `except Exception: continue` around every
collect_all() meant an engine could go missing from the bundle and the build
would still succeed, sign, and ship.
"""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

# Every engine the app advertises. mlx ships mlx.metallib (the GPU kernels) —
# without it every Metal path dies at runtime, inside the user's copy.
REQUIRED_PACKAGES = (
    "mlx",
    "mlx_whisper",
    "mlx_qwen3_asr",
    "mlx_lm",
    "onnx_asr",
    "onnxruntime",
    "faster_whisper",
    "ctranslate2",
    "av",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "numba",
    "llvmlite",
    "tiktoken",
    "tiktoken_ext",
    "regex",
)

# Present in the current environment as transitive dependencies; the app still
# runs without them, so a missing one is not a build failure.
OPTIONAL_PACKAGES = (
    "nagisa",
    "sklearn",
    "scipy",
)

HIDDEN_IMPORTS = [
    # Flat modules in app/ that are only reached through main.py.
    "cuda_setup",
    "main_window",
    "transcriber",
    "llm",
    "stats",
    "audio",
    "align",
    "merger",
    "exporters",
    "models",
    "diagnostics",
    # Imported lazily inside functions, so static analysis misses them.
    "mlx.core",
    "mlx.nn",
    "mlx.utils",
    "tqdm.std",
    "huggingface_hub.utils.tqdm",
    "faster_whisper.utils",
    "faster_whisper.vad",
    "encodings.idna",
]

# Dead weight. torch is only pulled in by mlx_whisper's Torch->MLX conversion
# helper and mlx_qwen3_asr's diarization path; this app uses neither, and it
# is 500 MB. The other excludes are unused Qt/dev/plotting stacks.
EXCLUDES = [
    "torch",
    "torchaudio",
    "torchvision",
    "tensorflow",
    "flax",
    "jax",
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "setuptools._distutils",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtQuick",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia",
    "PySide6.QtDesigner",
    "PySide6.QtBluetooth",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtWebSockets",
    "PySide6.QtWebChannel",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtHelp",
    "PySide6.QtSerialPort",
    "PySide6.QtNfc",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSpatialAudio",
    "PySide6.QtTextToSpeech",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
]


class MissingPackage(RuntimeError):
    """A package the shipped app needs could not be collected."""


def collect(app_dir: Path) -> tuple[list, list, list]:
    """(datas, binaries, hiddenimports) for either spec.

    Raises MissingPackage if a required engine package cannot be collected,
    so the build stops here rather than producing a signed, incomplete .app.
    """
    datas = [(str(app_dir / "assets" / "icon.png"), "assets")]
    binaries: list = []
    hiddenimports: list = []

    for pkg in REQUIRED_PACKAGES:
        try:
            d, b, h = collect_all(pkg)
        except Exception as exc:
            raise MissingPackage(
                f"Required package {pkg!r} could not be collected: "
                f"{type(exc).__name__}: {exc}\n"
                "The bundle would ship without it. Fix the environment "
                "(bash setup_mac.command) and build again."
            ) from exc
        if not (d or b or h):
            raise MissingPackage(
                f"Required package {pkg!r} collected nothing — it is probably "
                "not installed in this .venv."
            )
        datas += d
        binaries += b
        hiddenimports += h

    for pkg in OPTIONAL_PACKAGES:
        try:
            d, b, h = collect_all(pkg)
        except Exception as exc:
            print(f"spec: optional package {pkg!r} not collected ({exc})")
            continue
        datas += d
        binaries += b
        hiddenimports += h

    # Certificate bundle for model downloads from Hugging Face. Required: the
    # first run of every engine downloads weights over TLS.
    certifi_data = collect_data_files("certifi")
    if not certifi_data:
        raise MissingPackage(
            "certifi data files not found — the bundle could not verify "
            "Hugging Face's certificate and no model would ever download.")
    datas += certifi_data

    return datas, binaries, hiddenimports + HIDDEN_IMPORTS
