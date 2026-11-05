"""Write tests/test_data/v2/store_golden.json (never rewrite it once committed).

    uv run python tests/test_data/v2/make_store_golden.py

It pins what ``ShardSink`` writes for the fixed runs of ``tests/support/store_golden.py``:
the complete bytes (``.npy`` headers, manifests and ``store.json``) of every file as an
xxh3-128, the fingerprints, the entries and the log events. tests/test_store_golden.py
compares every build against it, so a refactoring of the store cannot change a byte. The
file was generated once on the code before the shared shard base (DR-162, Scores und
Vorhersagen dauerhaft speichern); this refuses to overwrite it.
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "support"))
sys.path.insert(0, str(HERE.parents[1]))

TARGET = HERE / "store_golden.json"


def main() -> None:
    from support.store_golden import observe, scenarios

    if TARGET.exists():
        raise SystemExit(f"{TARGET} exists; the goldens are never rewritten")
    pinned = {}
    for name, spec in scenarios().items():
        with tempfile.TemporaryDirectory() as tmp:
            pinned[name] = observe(Path(tmp), spec)
    TARGET.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(pinned)} scenarios to {TARGET}")


if __name__ == "__main__":
    main()
