"""A leader whose descendant announces readiness before outliving it."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    if sys.argv[2] == "leader":
        child = subprocess.Popen([sys.executable, __file__, str(root), "descendant", sys.argv[3]])
        while not (root / "ready.json").exists():
            time.sleep(0.01)
        (root / "descendant.pid").write_text(str(child.pid))
        return
    if sys.argv[3] == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / "ready.json").write_text(json.dumps({"ready": True}))
    while not (root / "release").exists():
        time.sleep(0.01)


if __name__ == "__main__":
    main()
