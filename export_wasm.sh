#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec uv run python -m marimo export html-wasm app.py -o build --mode run -f
