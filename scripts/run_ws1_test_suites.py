"""Run all dependency-light WS1 suites and persist exact output."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from finalize_ws1 import resolve_config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # non-UTF-8 consoles vs non-ASCII paths
        sys.stdout.reconfigure(errors="replace")
    _, run_dir = resolve_config()
    commands = [
        [sys.executable, "-m", "compileall", "-q", "mist", "scripts", "tests"],
        [sys.executable, "tests/test_core.py"],
        [sys.executable, "tests/test_ws1.py"],
        [sys.executable, "tests/test_ws1_finalization.py"],
    ]
    output = []
    for command in commands:
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8"
        )
        output.append(f"$ {' '.join(command)}")
        output.append(completed.stdout.rstrip())
        if completed.stderr:
            output.append(completed.stderr.rstrip())
        output.append(f"exit_code={completed.returncode}")
        if completed.returncode:
            (run_dir / "test_results.txt").write_text(
                "\n".join(output) + "\n", encoding="utf-8"
            )
            raise SystemExit(completed.returncode)
    (run_dir / "test_results.txt").write_text(
        "\n".join(output) + "\n", encoding="utf-8"
    )
    print(f"ALL {len(commands)} TEST COMMANDS PASS")
    print(run_dir / "test_results.txt")


if __name__ == "__main__":
    main()
