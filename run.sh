#!/usr/bin/env zsh
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python main.py "$@"
