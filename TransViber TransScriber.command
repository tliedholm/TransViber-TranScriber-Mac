#!/bin/bash
# Launches TransViber TransScriber. Run setup_mac.command once first.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Setup has not been run yet. Run:  bash setup_mac.command"
    read -r -p "Press Enter to close..."
    exit 1
fi
exec ./.venv/bin/python app/main.py
