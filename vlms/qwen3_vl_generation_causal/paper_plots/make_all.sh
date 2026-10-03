#!/usr/bin/env bash
# Regenerate every paper panel into figures/ (PDF for LaTeX, PNG for preview). No model inference.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
for f in fig*.py appx_*.py; do "$PY" "$f"; done
