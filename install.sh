set -euo pipefail

echo "Cleaning up build/, dist/, and dream.egg-info/…"
rm -rf build dist dream.egg-info

echo "Pre-installing build dependencies…"
pip install setuptools wheel

echo "Installing dream in editable mode (no build isolation)…"
pip install --no-build-isolation -e .

echo "Done!"

