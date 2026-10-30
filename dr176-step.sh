if [ "$EVENT" != "pull_request" ]; then
  echo "code=true" >> "$GITHUB_OUTPUT"
  echo "native=true" >> "$GITHUB_OUTPUT"
  exit 0
fi
files=$(git diff --name-only "origin/$BASE...HEAD")
echo "$files"
other=$(echo "$files" | grep -vE '^docs/|\.md$' || true)  # files beyond the docs
if [ -n "$other" ] || echo "$files" | grep -qx 'README.md'; then
  echo "code=true" >> "$GITHUB_OUTPUT"
else
  echo "code=false" >> "$GITHUB_OUTPUT"
fi
# The native code and the build: the extension sources, setup.py, the project
# metadata, the sdist manifest, the lockfile and the workflows themselves.
native_re='^(src/fse/_native|setup\.py$|pyproject\.toml$|MANIFEST\.in$|uv\.lock$|\.github/workflows/)|\.(pyx|pxd|pxi|h)$'
if echo "$files" | grep -qE "$native_re"; then
  echo "native=true" >> "$GITHUB_OUTPUT"
else
  echo "native=false" >> "$GITHUB_OUTPUT"
fi
