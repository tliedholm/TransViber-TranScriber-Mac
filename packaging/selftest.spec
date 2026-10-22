# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the frozen engine self-test.

Built and run by packaging/build_app.sh as a release gate. Shares
packaging/analysis_inputs.py with TransViber.spec on purpose: a self-test
frozen from a different analysis would prove nothing about the shipped app.
"""

import sys
from pathlib import Path

sys.path.insert(0, SPECPATH)
import analysis_inputs

PROJECT = Path(SPECPATH).parent
APP_DIR = PROJECT / "app"

datas, binaries, hiddenimports = analysis_inputs.collect(APP_DIR)
excludes = analysis_inputs.EXCLUDES

a = Analysis(
    [str(PROJECT / "packaging" / "selftest.py")],
    pathex=[str(APP_DIR), str(PROJECT / "packaging")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="selftest",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="selftest",
)

