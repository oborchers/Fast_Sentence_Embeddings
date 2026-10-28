import subprocess
import sys


def test_importing_the_runtime_starts_no_thread():
    code = (
        "import threading, fse._runtime, fse._runtime._executor; "
        "assert threading.active_count() == 1, threading.enumerate()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
