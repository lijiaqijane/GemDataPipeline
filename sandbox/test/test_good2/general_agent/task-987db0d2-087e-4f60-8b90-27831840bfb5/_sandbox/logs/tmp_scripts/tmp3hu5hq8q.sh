set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
python -c "import tools; print('TOOLS_IMPORT_OK')"
tar -czf _output.tar.gz --warning=no-file-changed --warning=no-file-removed --ignore-failed-read --exclude=_output.tar.gz --exclude=_input.tar.gz .