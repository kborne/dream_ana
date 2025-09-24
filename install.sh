#!/usr/bin/env bash
set -euo pipefail

# Remove previous build artifacts (if you had any)
echo "Cleaning up build/, dist/, and dream.egg-info/…"
rm -rf build dist dream.egg-info

# Install in “editable” (development) mode so you can use the code as-is.
echo "Installing dream in editable mode (pip install -e .)…"
pip install --force-reinstall -e .

echo "Done!"

