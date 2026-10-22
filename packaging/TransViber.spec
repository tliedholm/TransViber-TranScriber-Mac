# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the distributable TransViber TransScriber.app.

Build with packaging/build_app.sh (which generates the .icns first).

The collected packages, hidden imports and excludes live in
packaging/analysis_inputs.py, shared with selftest.spec so the self-test is
frozen from the same analysis as the app it is testing.
"""

import sys
from pathlib import Path

sys.path.insert(0, SPECPATH)
import analysis_inputs

PROJECT = Path(SPECPATH).parent
APP_DIR = PROJECT / "app"

# Raises MissingPackage — and so fails the build — if an engine is absent.
datas, binaries, hiddenimports = analysis_inputs.collect(APP_DIR)
excludes = analysis_inputs.EXCLUDES

a = Analysis(
    [str(APP_DIR / "main.py")],
    pathex=[str(APP_DIR)],
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
    name="TransViber TransScriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,  # lets Finder "Open With" / drag-onto-icon pass files
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
    name="TransViber TransScriber",
)

app = BUNDLE(
    coll,
    name="TransViber TransScriber.app",
    icon=str(PROJECT / "packaging" / "build" / "icon.icns"),
    bundle_identifier="com.transviber.transscriber",
    version="1.1.0",
    info_plist={
        "CFBundleName": "TransViber TransScriber",
        "CFBundleDisplayName": "TransViber TransScriber",
        "CFBundleShortVersionString": "1.1.0",
        "CFBundleVersion": "1.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHumanReadableCopyright": "Local, offline transcription. Nothing is uploaded.",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Audio file",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": [
                    "public.audio",
                    "public.mpeg-4-audio",
                    "com.microsoft.waveform-audio",
                    "org.xiph.flac",
                    "public.mp3",
                    "com.apple.m4a-audio",
                    "org.matroska.mkv",
                ],
            }
        ],
    },
)
