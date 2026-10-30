full=$(jq -nc '[{os: "ubuntu-latest", "python-version": ("3.11", "3.12", "3.13", "3.14", "3.15"), deps: "latest"}, {os: "macos-latest", "python-version": ("3.11", "3.12", "3.13", "3.14", "3.15"), deps: "latest"}, {os: "ubuntu-latest", "python-version": "3.11", deps: "min"}]')
if [ "$EVENT" != "pull_request" ]; then
  {
    for name in code native deps st mlx sdist; do echo "$name=true"; done
    echo "test_matrix={\"include\":$full}"
  } >> "$GITHUB_OUTPUT"
  exit 0
fi
files=$(git diff --name-only "origin/$BASE...HEAD")
echo "$files"
# Whether a changed file matches an extended regular expression.
touches() { echo "$files" | grep -qE "$1"; }
# Output NAME is true when a file matches the regular expression or a workflow changed.
flag() {
  if touches "$2" || touches '^\.github/workflows/'; then echo "$1=true"; else echo "$1=false"; fi
}
other=$(echo "$files" | grep -vE '^docs/|\.md$' || true)  # files beyond the docs
if [ -n "$other" ] || touches '^README\.md$'; then code=true; else code=false; fi
{
  echo "code=$code"
  # The extension sources, setup.py, the project metadata, the sdist manifest,
  # the lockfile.
  flag native '^src/fse/_native|^(setup\.py|pyproject\.toml|MANIFEST\.in|uv\.lock)$|\.(pyx|pxd|pxi|h)$'
  flag deps '^(pyproject\.toml|uv\.lock)$'
  # The converter, its tests and helpers (tests/support/converted.py builds the
  # models of the conversion tests), and the pins of the st group.
  flag st '^src/fse/_encoder/_sentence_transformers|^tests/test_encoder_sentence_transformers|^tests/support/(sentence_transformers|converted)\.py$|^(pyproject\.toml|uv\.lock)$'
  # The backend and its tests, the backend registry, the device conformance
  # tests with their helpers, and everything native (the kernels it mirrors).
  flag mlx '^src/fse/_backends/(_mlx/|__init__\.py$)|^tests/test_backends_mlx|^tests/test_conformance_device|^tests/support/(fake_mlx|conformance)\.py$|^src/fse/_native|^(setup\.py|pyproject\.toml|MANIFEST\.in|uv\.lock)$|\.(pyx|pxd|pxi|h)$'
  flag sdist '^(setup\.py|pyproject\.toml|MANIFEST\.in|uv\.lock)$|^src/fse/[^/]*\.(pyx|pxd|pxi|h)$'
} >> "$GITHUB_OUTPUT"
# Linux on 3.13, plus macOS when the native code changes, plus the minimum
# versions when the pins change.
matrix=$(jq -nc --argjson native "$(grep -qx 'native=true' "$GITHUB_OUTPUT" && echo true || echo false)" --argjson deps "$(grep -qx 'deps=true' "$GITHUB_OUTPUT" && echo true || echo false)" '[{os: "ubuntu-latest", "python-version": "3.13", deps: "latest"}] + (if $native then [{os: "macos-latest", "python-version": "3.13", deps: "latest"}] else [] end) + (if $deps then [{os: "ubuntu-latest", "python-version": "3.11", deps: "min"}] else [] end)')
echo "test_matrix={\"include\":$matrix}" >> "$GITHUB_OUTPUT"
