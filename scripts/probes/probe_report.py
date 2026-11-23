"""Shared evidence-report scaffolding for the standalone environment probes."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback


def run_probe(output, initial, check, *, error_keys, dll_directory=False):
    """Write an exclusive JSON report, print it, and return a process exit code.

    error_keys preserves each probe's historical field insertion order.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    dll_handle = None
    if dll_directory and os.name == "nt":
        dll_handle = os.add_dll_directory(str(Path(sys.prefix) / "Library/bin"))
    with output.open("x", encoding="utf-8") as handle:
        report = dict(initial)
        try:
            report.update(check())
            report["passed"] = True
        except Exception as exc:
            fields = {"error": str(exc), "error_type": type(exc).__name__,
                      "traceback": traceback.format_exc()}
            for key in error_keys:
                report[key] = fields[key]
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if dll_handle:
        dll_handle.close()
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["passed"] else 1


def checked_at_utc():
    return datetime.now(timezone.utc).isoformat()