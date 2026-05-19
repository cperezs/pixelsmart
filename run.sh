#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh — Launch the Pixel-Level Annotator
#
# Usage:
#   ./run.sh
#
# The viewer backend (OpenGL or Qt software renderer) is configured in
# config.toml.  See README.md for details.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec python main.py "$@"
