build_and_install.sh
# Cleans previous builds, builds a wheel, and installs it locally.

set -euo pipefail

# Remove previous build artifacts
echo "Cleaning up build/, dist/, and dream.egg-info/…"
rm -rf build dist dream.egg-info

# Build a binary wheel
echo "Building wheel with setup.py..."
python setup.py bdist_wheel

# Build a wheel package without dependencies or isolation
echo "Generating wheel in dist/…"
pip wheel . --no-deps --no-build-isolation -w dist/

# Install the newly built wheel
echo "Installing dream-0.1.0-py3-none-any.whl…"
cd dist
pip install --force-reinstall dream-0.1.0-py3-none-any.whl

echo "Done!"
