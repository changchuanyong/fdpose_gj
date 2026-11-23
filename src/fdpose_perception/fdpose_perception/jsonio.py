"""Shared JSON output so every entry point writes evidence the same way."""

import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")