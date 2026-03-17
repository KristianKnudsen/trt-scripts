#!/bin/bash
# Usage: run-build.sh --config <path> --paths <path>
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mpirun --allow-run-as-root -n 1 python "$SCRIPT_DIR/build_engine.py" "$@"
