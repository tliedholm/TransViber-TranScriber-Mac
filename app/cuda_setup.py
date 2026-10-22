"""Make pip-installed NVIDIA CUDA DLLs (cuBLAS, cuDNN) visible to CTranslate2.

Must be imported before ctranslate2 / faster_whisper.
"""

import os
import sys
from pathlib import Path


def add_nvidia_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not base.is_dir():
        return
    for sub in base.iterdir():
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


add_nvidia_dll_dirs()
