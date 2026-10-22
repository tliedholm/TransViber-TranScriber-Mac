#!/bin/bash
# Builds the distributable, self-contained TransViber TransScriber.app and
# zips it for sharing. Run from anywhere:  bash packaging/build_app.sh
#
# Requires the project's .venv (setup_mac.command) plus pyinstaller in it.
#
# Release gates, in order — the build stops at the first failure so a bundle
# with a missing engine can never be signed and shipped:
#   1. unit tests            (tests/, fast, no models)
#   2. dependency preflight  (packaging/preflight.py)
#   3. spec collection       (analysis_inputs.py raises on a missing engine)
#   4. frozen bundle check   (selftest --imports-only, inside the .app's env)
#   5. frozen engine test    (selftest <audio>, only with a sample to hand)
#
# Environment:
#   SELFTEST_AUDIO=/path/to/sample.wav   also run the live per-engine test
#   SKIP_SELFTEST=1                      skip gates 4 and 5 (not for releases)
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$PROJECT/packaging"
BUILD="$PKG/build"
DIST="$PKG/dist"
PY="$PROJECT/.venv/bin/python"
APP_NAME="TransViber TransScriber"
SELFTEST_AUDIO="${SELFTEST_AUDIO:-}"
SKIP_SELFTEST="${SKIP_SELFTEST:-0}"

[ -x "$PY" ] || { echo "No .venv — run: bash setup_mac.command"; exit 1; }
"$PY" -c "import PyInstaller" 2>/dev/null || "$PY" -m pip install --quiet pyinstaller

echo "==> Unit tests"
if "$PY" -c "import pytest" 2>/dev/null; then
    "$PY" -m pytest -q "$PROJECT/tests"
else
    echo "    pytest not installed — installing it (it is excluded from the bundle)"
    "$PY" -m pip install --quiet pytest
    "$PY" -m pytest -q "$PROJECT/tests"
fi

echo "==> Dependency preflight"
"$PY" "$PKG/preflight.py"

echo "==> Cleaning previous build"
rm -rf "$BUILD" "$DIST"
mkdir -p "$BUILD"

echo "==> Generating icon.icns"
ICONSET="$BUILD/icon.iconset"
mkdir -p "$ICONSET"
SRC="$PROJECT/app/assets/icon.png"
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$SRC" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
done
# Retina variants: icon_NxN@2x.png is a 2N-pixel image.
cp "$ICONSET/icon_32x32.png"   "$ICONSET/icon_16x16@2x.png"
cp "$ICONSET/icon_64x64.png"   "$ICONSET/icon_32x32@2x.png"
cp "$ICONSET/icon_256x256.png" "$ICONSET/icon_128x128@2x.png"
cp "$ICONSET/icon_512x512.png" "$ICONSET/icon_256x256@2x.png"
rm -f "$ICONSET/icon_64x64.png"
iconutil -c icns "$ICONSET" -o "$BUILD/icon.icns"

echo "==> Running PyInstaller (this takes several minutes)"
"$PY" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$DIST" \
    --workpath "$BUILD/work" \
    "$PKG/TransViber.spec"

APP="$DIST/$APP_NAME.app"
[ -d "$APP" ] || { echo "Build failed: no $APP"; exit 1; }

# COLLECT leaves a loose copy of the same payload next to the .app.
rm -rf "${APP%.app}"

echo "==> Ad-hoc signing (required for Apple Silicon to run the bundle)"
# --deep is deprecated but is still the only practical way to sign the
# hundreds of nested .so/.dylib files PyInstaller collects.
codesign --force --deep --sign - --timestamp=none "$APP"
codesign --verify --deep --strict "$APP" && echo "    signature OK"

if [ "$SKIP_SELFTEST" = "1" ]; then
    echo "==> !! Self-test SKIPPED (SKIP_SELFTEST=1) — do not ship this build"
else
    echo "==> Building the frozen self-test (same analysis as the app)"
    "$PY" -m PyInstaller \
        --noconfirm \
        --distpath "$DIST" \
        --workpath "$BUILD/work-selftest" \
        "$PKG/selftest.spec"

    SELFTEST="$DIST/selftest/selftest"
    [ -x "$SELFTEST" ] || { echo "Self-test build failed: no $SELFTEST"; exit 1; }

    echo "==> Self-test: bundle contents"
    "$SELFTEST" --imports-only

    if [ -n "$SELFTEST_AUDIO" ]; then
        echo "==> Self-test: live engines ($SELFTEST_AUDIO)"
        [ -f "$SELFTEST_AUDIO" ] || { echo "No such file: $SELFTEST_AUDIO"; exit 1; }
        "$SELFTEST" "$SELFTEST_AUDIO"
    else
        echo "==> Self-test: live engine run skipped (set SELFTEST_AUDIO=/path/to.wav)"
        echo "    Metal inference is only proven by that run — do it before a release."
    fi
fi

echo "==> Staging the shareable folder"
STAGE="$DIST/stage/TransViber TransScriber"
rm -rf "$DIST/stage"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/$APP_NAME.app"
cp "$PKG/READ ME FIRST.txt" "$STAGE/READ ME FIRST.txt"
ln -s /Applications "$STAGE/Applications"

echo "==> Zipping"
ZIP="$DIST/TransViber-TransScriber-mac.zip"
rm -f "$ZIP"
# ditto keeps symlinks, resource forks and the ad-hoc signature intact;
# `zip -r` would corrupt the bundle.
ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$ZIP"

echo
echo "=== Done ==="
echo "App:  $APP"
echo "Zip:  $ZIP"
du -sh "$APP" "$ZIP"
