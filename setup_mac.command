#!/bin/bash
# TransViber TransScriber — one-time Mac setup.
# First run: open Terminal, drag this file in (or cd to the folder), and run:
#   bash setup_mac.command
set -e
cd "$(dirname "$0")"

echo "=== TransViber TransScriber — Mac setup ==="

# Find a suitable Python (3.10+). Prefer newer Homebrew installs.
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo ""
    echo "No Python 3.10+ found. Install one with Homebrew and re-run this script:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  brew install python@3.12"
    exit 1
fi

echo "Using $($PY --version) at $(command -v $PY)"

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    "$PY" -m venv .venv
fi

echo "Installing dependencies (a few hundred MB, one time)..."
./.venv/bin/python -m pip install --upgrade pip --quiet
# Pinned in requirements-mac.txt: these eight ML/UI projects move fast, and an
# unpinned install would give this machine a different environment from the
# one the app was tested and packaged on.
./.venv/bin/python -m pip install -r requirements-mac.txt

echo "Checking the installation..."
./.venv/bin/python packaging/preflight.py

chmod +x "TransViber TransScriber.command" setup_mac.command

echo ""
echo "=== Setup complete ==="
echo "Launch the app by double-clicking 'TransViber TransScriber.command'."
echo "(If macOS blocks it the first time: right-click it, choose Open, then Open again.)"
