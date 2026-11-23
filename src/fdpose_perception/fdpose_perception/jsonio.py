"""Shared JSON output so every entry point writes evidence the same way."""

import json
from pathlib import Path


def write_json(path, value, *, ensure_ascii=True, allow_nan=True, exclusive=False):
    text = json.dumps(value, indent=2, ensure_ascii=ensure_ascii, allow_nan=allow_nan) + "\n"
    path = Path(path)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")