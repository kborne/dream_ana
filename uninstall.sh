#!/usr/bin/env bash
set -euo pipefail

# Uninstall the dream package
echo "Uninstalling dream…"
pip uninstall --yes dream

# Remove any locally built artifacts
echo "Cleaning up build/, dist/, and dream.egg-info/…"
rm -rf build dist dream.egg-info

echo "Done!"

